"""Bounded developmental capability, overfit, and exact-data SFT diagnostics.

All paths are explicit; no downloads, resumption, checkpoint selection or test
set evaluation. The historical evaluator and training objective stay frozen.
"""
import argparse
import collections
from contextlib import contextmanager
import gc
import json
import math
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch
from safetensors.torch import save_file

from .benchmark import (PREFIX, SUFFIX, _cuda_memory, _gpu_metadata, _json,
    _synchronize, _versions, backward_sample, prepare_row, sha256_file)
from .data import read_jsonl, validate_rows
from .diagnostics_data import INTERPRETATION, SPLITS, freeze_suite, load_suite
from .eval_metrics import LABELS, output_summary
from .evaluation import (RUNTIME_SOURCES, canonical_hash, evaluate_rows,
                         load_eval_model, now, prepare_input)
from .model import load_icae

ORIGINAL_TRAIN_SHA256 = '882c179ceec6d832c5136181eeef6eaf9fbf76912c4dc9808bbfb63200994763'
OLD_SFT_SHA256 = 'ff7459d91dcd13cf5eab8f9c96a15fde34639b44dee7136fffd55b2eaba86f7e'
SOURCES = tuple(dict.fromkeys(RUNTIME_SOURCES + ('diagnostics_data.py', 'diagnostics.py')))
TRAIN_DEV_SPLITS = ('dev_d0', 'dev_trainstyle_d0', 'dev_d12', 'dev_unknown')


def parse_eval_steps(text, total):
    if not isinstance(total, int) or total < 1:
        raise ValueError('Training steps must be positive')
    try:
        steps = [int(part) for part in text.split(',')]
    except (TypeError, ValueError, AttributeError) as error:
        raise ValueError('Evaluation steps must be comma-separated integers') from error
    if len(steps) != len(set(steps)) or any(step < 0 or step > total for step in steps):
        raise ValueError('Duplicate or out-of-range evaluation steps')
    return sorted(set(steps) | {0, total})


def _status(output, **values):
    value = dict(timestamp=now(), **values)
    _json(Path(output) / 'status.json', value)
    print(json.dumps(value, allow_nan=False), flush=True)


def _new_output(path):
    output = Path(path)
    output.mkdir(parents=True, exist_ok=False)
    _status(output, status='running', phase='validation', completed=0, total=0)
    return output


def _fail(output, exc):
    previous = json.loads((output / 'status.json').read_text(encoding='utf-8'))
    previous.pop('timestamp', None)
    previous.update(status='failed', error_type=type(exc).__name__, error=str(exc),
        traceback=traceback.format_exc(), oom=isinstance(exc, torch.cuda.OutOfMemoryError))
    _status(output, **previous)


def _write_config(output, config):
    config['config_sha256'] = canonical_hash(config)
    _json(output / 'config.json', config)


def _base_files(base_path):
    base_path = Path(base_path)
    if base_path.is_file():
        return [base_path]
    files = sorted(p for p in base_path.rglob('*') if p.is_file())
    if not files:
        raise ValueError('Base model directory is absent or empty')
    return files


class InputBinding:
    """Hash all model/data/source inputs once; recheck identity before contrasts.

    Large immutable model files are guarded by size/mtime between contrasts and
    rehashed at completion. Small suite/source files are rehashed every time.
    This avoids reading tens of GB at every training checkpoint.
    """
    def __init__(self, args, checkpoint=None):
        suite = Path(args.suite)
        self.paths = {f'source/{name}': Path(__file__).with_name(name) for name in SOURCES}
        self.paths.update({f'suite/{name}': suite / name
            for name in ['manifest.json'] + [f'{split}.jsonl' for split in SPLITS]})
        self.paths['adapter'] = Path(args.adapter_path)
        base = Path(args.base_path)
        self.paths.update({f'base/{p.relative_to(base) if base.is_dir() else p.name}': p
                           for p in _base_files(base)})
        if getattr(args, 'dataset', None):
            self.paths['dataset'] = Path(args.dataset)
        if checkpoint:
            self.paths['checkpoint'] = Path(checkpoint)
        if getattr(args, 'reference_checkpoint', None):
            self.paths['reference_checkpoint'] = Path(args.reference_checkpoint)
        self.hashes = {name: sha256_file(path) for name, path in self.paths.items()}
        self.stats = {name: (p.stat().st_size, p.stat().st_mtime_ns) for name, p in self.paths.items()}

    def assert_unchanged(self, full=False):
        for name, path in self.paths.items():
            stat = path.stat()
            if (stat.st_size, stat.st_mtime_ns) != self.stats[name]:
                raise ValueError('Compared input changed: ' + name)
            if full or stat.st_size < 8 * 1024 * 1024:
                if sha256_file(path) != self.hashes[name]:
                    raise ValueError('Compared input hash changed: ' + name)


@contextmanager
def evaluation_state(model):
    """Restore all module flags and all stochastic streams even after failure."""
    modes = [(module, module.training) for module in model.modules()]
    py_state, np_state, cpu_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        model.eval()
        with torch.no_grad():
            yield
    finally:
        for module, training in modes:
            module.training = training
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def trainable_reference(model):
    # adapter_state() may alias CPU parameters: clone is essential here.
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}


@torch.no_grad()
def parameter_drift(model, reference):
    groups = {name: dict(squared_delta=0., squared_initial=0., parameters=0)
              for name in ('all', 'memory_embedding', 'lora', 'other')}
    for name, parameter in model.named_parameters():
        if name not in reference:
            continue
        initial = reference[name].float()
        current = parameter.detach().float().cpu()
        delta = float((current - initial).square().sum(dtype=torch.float64))
        norm = float(initial.square().sum(dtype=torch.float64))
        group = 'memory_embedding' if name.startswith('memory_token_embed.') else 'lora' if 'lora_' in name else 'other'
        for key in ('all', group):
            groups[key]['squared_delta'] += delta
            groups[key]['squared_initial'] += norm
            groups[key]['parameters'] += initial.numel()
    for values in groups.values():
        values['l2_delta'] = math.sqrt(values.pop('squared_delta'))
        values['initial_l2'] = math.sqrt(values.pop('squared_initial'))
        values['relative_l2_delta'] = values['l2_delta'] / values['initial_l2'] if values['initial_l2'] else None
    return groups


def tokenizer_audit(tokenizer, row):
    item = prepare_input(row, tokenizer, 'legacy')
    legacy = item['left'] + item['context'][1:] + item['right']
    result = dict(id=row['id'], legacy_full_context_ids=legacy,
        legacy_full_context_decoded=tokenizer.decode(legacy, skip_special_tokens=False),
        native_message_content=row['context'] + row['question'],
        purpose='Illustrative prompt boundary audit only; native IDs never replace legacy prompts. '
                'Different token IDs alone do not establish a causal prompt explanation.',
        label_tokenization={label: dict(with_leading_space=tokenizer.encode(' ' + label, add_special_tokens=False),
            without_leading_space=tokenizer.encode(label, add_special_tokens=False)) for label in LABELS},
        decoded_boundaries=dict(context_to_question=tokenizer.decode(item['context'][-12:] + item['right'][:12], skip_special_tokens=False),
            suffix_to_answer=tokenizer.decode(item['right'][-4:] + tokenizer.encode(' ' + row['answer'], add_special_tokens=False) +
                                             [tokenizer.eos_token_id], skip_special_tokens=False)))
    for tokens in result['label_tokenization'].values():
        tokens['with_leading_space_count'] = len(tokens['with_leading_space'])
        tokens['without_leading_space_count'] = len(tokens['without_leading_space'])
    try:
        native = tokenizer.apply_chat_template([{'role': 'user', 'content': result['native_message_content']}],
                                               tokenize=True, add_generation_prompt=True)
        if hasattr(native, 'tolist'):
            native = native.tolist()
        result.update(supported=True, native_chat_template_ids=native,
                      identical_ids=legacy == native, legacy_token_count=len(legacy), native_token_count=len(native))
    except (AttributeError, ValueError, TypeError, NotImplementedError) as exc:
        result.update(supported=False, reason=f'{type(exc).__name__}: {exc}')
    return result


@torch.no_grad()
def _answer_nll(model, item, answer, arm):
    if arm != 'full_context':
        memory = model.compress_ids(torch.tensor([item['context']], device=model.device))
        value = -model.score_ids(memory, item['left'], item['right'], [answer]).mean()
    else:
        prefix = item['left'] + item['context'][1:] + item['right']
        with model.icae.disable_adapter():
            out = model.base(inputs_embeds=model.embeddings(prefix + answer).unsqueeze(0), use_cache=False)
        logits = out.logits[0, len(prefix)-1:len(prefix)-1+len(answer)].float()
        targets = torch.tensor(answer, device=model.device)
        value = -torch.log_softmax(logits, -1).gather(-1, targets[:, None]).mean()
    return float(value), bool(value.requires_grad)


def evaluate_split(model, tokenizer, rows, arm, output, candidates=False, progress=None):
    """Strict generation plus independent answer/EOS NLL, preserving train state."""
    validate_rows(rows)
    if not rows:
        raise ValueError('Cannot evaluate an empty split')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    _synchronize(model.device)
    started = time.perf_counter()
    with evaluation_state(model):
        metrics = evaluate_rows(model, tokenizer, rows, 'legacy', arm, output,
                                candidates=candidates, max_new_tokens=32, progress=progress)
        nlls = []
        with (output / 'teacher_forced.jsonl').open('x', encoding='utf-8') as losses, \
             (output / 'prompt_audit.jsonl').open('x', encoding='utf-8') as prompts:
            for row in rows:
                item = prepare_input(row, tokenizer, 'legacy')
                answer = tokenizer.encode(' ' + row['answer'].lstrip(), add_special_tokens=False) + [tokenizer.eos_token_id]
                value, requires_grad = _answer_nll(model, item, answer, arm)
                if not math.isfinite(value):
                    raise FloatingPointError('Non-finite evaluation loss')
                nlls.append(value)
                losses.write(json.dumps(dict(id=row['id'], mean_answer_nll=value,
                    answer_token_ids=answer, answer_token_count=len(answer), requires_grad=requires_grad)) + '\n')
                losses.flush()
                prompts.write(json.dumps(dict(id=row['id'], context=row['context'], question=row['question'],
                    context_token_ids=item['context'], left_token_ids=item['left'], right_token_ids=item['right'],
                    context_token_count=len(item['context']), prompt_variant='legacy',
                    boundary='full: left+context_without_bos+right; compressed: left+128_memory+learned_ft+right',
                    actual_memory_slots=model.memory_size)) + '\n')
                prompts.flush()
    predictions = read_jsonl(output / 'predictions.jsonl')
    metrics['by_family_metrics'] = {family: output_summary(
        [row for row in rows if row['family'] == family],
        [p for p in predictions if p['id'] in {r['id'] for r in rows if r['family'] == family}])
        for family in sorted({row['family'] for row in rows})}
    metrics.update(teacher_forced_mean_answer_nll=sum(nlls) / len(nlls),
        teacher_forced_definition='mean over examples of mean answer/EOS token negative log probability',
        output_label_frequencies=dict(collections.Counter(p['format']['exact_label'] or '<malformed>' for p in predictions)),
        raw_output_frequencies=dict(collections.Counter(p['prediction'] for p in predictions)),
        stop_reason_counts=dict(collections.Counter(p['generation']['stop_reason'] for p in predictions)),
        diagnostic_only=True)
    _synchronize(model.device)
    metrics['wall_seconds'] = time.perf_counter() - started
    metrics.update(_cuda_memory(model.device))
    _json(output / 'summary.json', metrics)
    return metrics


def _common_config(args, binding, stage):
    return dict(stage=stage, arguments=vars(args), input_hashes=binding.hashes,
        source_hashes={name: binding.hashes[f'source/{name}'] for name in SOURCES},
        versions=_versions(), prompt_variant='legacy', max_new_tokens=32,
        max_context_tokens=512, candidate_labels=list(LABELS),
        base_dtype='bfloat16', trainable_dtype='float32', interpretation=INTERPRETATION,
        comparison_identity='All contrasts share these hashes; small files rehashed before each contrast, '
                            'large inputs size/mtime guarded and rehashed at completion.')


def run_capability(args):
    output = _new_output(args.output)
    try:
        suite = load_suite(args.suite)
        binding = InputBinding(args, args.checkpoint)
        config = _common_config(args, binding, 'capability')
        config.update(arms=['full_context', 'original_icae', 'sft'],
                      checkpoint_matches_historical_hash=binding.hashes['checkpoint'] == OLD_SFT_SHA256,
                      historical_sft_sha256=OLD_SFT_SHA256)
        _write_config(output, config)
        splits = tuple(split for split in SPLITS if split != 'overfit')
        total = 3 * sum(len(suite[split]) for split in splits)
        completed, summaries = 0, {}
        _status(output, status='running', phase='load_original', completed=completed, total=total)
        model, tokenizer = load_eval_model(args.base_path, args.adapter_path, device=args.device)
        _json(output / 'tokenizer_audit.json', tokenizer_audit(tokenizer, suite['dev_d12'][0]))
        for arm in ('full_context', 'original_icae', 'sft'):
            binding.assert_unchanged()
            if arm == 'sft':
                # Delete every reference before loading the trained model.
                del model, tokenizer
                gc.collect()
                if args.device.startswith('cuda'):
                    torch.cuda.empty_cache()
                _status(output, status='running', phase='load_sft', completed=completed, total=total)
                model, tokenizer = load_eval_model(args.base_path, args.adapter_path, args.checkpoint, args.device)
            summaries[arm] = {}
            for split in splits:
                prior = completed
                progress = lambda n, rid: _status(output, status='running', phase='generation',
                    arm=arm, split=split, completed=prior+n, total=total, last_id=rid)
                summaries[arm][split] = evaluate_split(model, tokenizer, suite[split], arm,
                    output / arm / split, candidates=True, progress=progress)
                completed += len(suite[split])
                _json(output / 'partial_summary.json', summaries)
        binding.assert_unchanged(full=True)
        result = dict(status='success', metrics=summaries, config_sha256=config['config_sha256'],
                      interpretation=INTERPRETATION)
        _json(output / 'summary.json', result)
        _status(output, status='success', phase='complete', completed=completed, total=total)
        return result
    except BaseException as exc:
        _fail(output, exc)
        raise


def _seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_train(args, model=None, tokenizer=None):
    output = _new_output(args.output)
    started = time.perf_counter()
    try:
        if args.regime not in ('overfit', 'replay'):
            raise ValueError('Unsupported training regime')
        if not math.isfinite(args.learning_rate) or args.learning_rate < 0:
            raise ValueError('Learning rate must be finite and nonnegative')
        checkpoints = parse_eval_steps(args.eval_steps, args.steps)
        suite = load_suite(args.suite)
        binding = InputBinding(args)
        config = _common_config(args, binding, 'train')
        config.update(eval_steps=checkpoints, seed_before_load=args.seed,
            method='sft', microbatch=1, grad_accumulation=1, max_grad_norm=1.,
            optimizer='AdamW', optimizer_settings=dict(lr=args.learning_rate, weight_decay=0.,
            betas=[.9, .999], eps=1e-8, amsgrad=False, foreach=None, fused=None),
            sample_order='random.Random(seed).shuffle(indices), cyclic without reshuffling',
            candidate_diagnostics=bool(args.candidate_diagnostics),
            historical_sft_sha256=OLD_SFT_SHA256, eval_perturbs_training_rng=False)
        _write_config(output, config)
        if args.regime == 'replay':
            if binding.hashes.get('dataset') != args.expected_dataset_hash:
                raise ValueError('Replay dataset hash mismatch')
            rows = read_jsonl(args.dataset)
            validate_rows(rows)
            if not rows or any(row['kind'] != 'swap' or row['split'] != 'train' for row in rows):
                # Tiny fixture may use the diagnostic overfit schema but still
                # must consist exclusively of complete known-condition pairs.
                if args.expected_dataset_hash == ORIGINAL_TRAIN_SHA256 or not rows or any(row['kind'] != 'swap' for row in rows):
                    raise ValueError('Replay requires known-condition training swaps only')
        else:
            rows = suite['overfit']
        _seed(args.seed)
        _status(output, status='running', phase='load_model', completed=0, total=args.steps)
        if model is None:
            model, tokenizer = load_icae(args.base_path, args.adapter_path,
                device=args.device, dtype=torch.bfloat16, memory_size=128)
        if tokenizer is None:
            raise ValueError('Tokenizer required')
        model.retrieval_proj.requires_grad_(False)
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        if model.base.is_gradient_checkpointing:
            raise ValueError('Shared adapter checkpointing is forbidden')
        trainables = [p for p in model.parameters() if p.requires_grad]
        reference = trainable_reference(model)
        config.update(memory_slots=model.memory_size,
            lora_ranks=sorted({module.r['default'] for module in model.modules()
                if isinstance(getattr(module, 'r', None), dict) and 'default' in module.r}),
            lora_alpha=32, lora_dropout=.05, lora_target_modules=['q_proj', 'v_proj'],
            trainable_dtypes=sorted({str(p.dtype) for p in trainables}),
            actual_base_dtype=str(model.base.get_input_embeddings().weight.dtype),
            trainable_parameters=sum(p.numel() for p in trainables), gpu=_gpu_metadata(model.device),
            training_rows=len(rows), training_dataset_sha256=(binding.hashes['dataset']
                if args.regime == 'replay' else binding.hashes['suite/overfit.jsonl']))
        config.pop('config_sha256', None)
        _write_config(output, config)
        items = [prepare_row(row, tokenizer) for row in rows]
        order = list(range(len(items)))
        random.Random(args.seed).shuffle(order)
        config['cyclic_row_ids'] = [items[index]['id'] for index in order]
        config.pop('config_sha256', None)
        _write_config(output, config)
        model.train()
        optimizer = torch.optim.AdamW(trainables, **config['optimizer_settings'])
        if model.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(model.device)
        evaluated, trajectories = [], {}

        def checkpoint(step):
            binding.assert_unchanged()
            splits = (('overfit',) if args.regime == 'overfit' else ()) + TRAIN_DEV_SPLITS
            total_eval = sum(len(suite[split]) for split in splits)
            completed_eval, metrics = 0, {}
            for split in splits:
                prior = completed_eval
                progress = lambda n, rid: _status(output, status='running', phase='evaluation',
                    completed=step, total=args.steps, checkpoint_step=step, split=split,
                    eval_completed=prior+n, eval_total=total_eval, last_id=rid)
                metrics[split] = evaluate_split(model, tokenizer, suite[split], 'sft',
                    output / f'step-{step:04d}' / split,
                    candidates=args.candidate_diagnostics, progress=progress)
                completed_eval += len(suite[split])
            entry = dict(step=step, metrics=metrics, parameter_drift=parameter_drift(model, reference),
                         timestamp=now(), **_cuda_memory(model.device))
            trajectories[str(step)] = entry
            _json(output / f'step-{step:04d}' / 'summary.json', entry)
            _json(output / 'trajectory.json', trajectories)
            evaluated.append(step)

        checkpoint(0)
        with (output / 'steps.jsonl').open('x', encoding='utf-8', buffering=1) as log:
            for step in range(1, args.steps + 1):
                _synchronize(model.device)
                step_start = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                item = items[order[(step - 1) % len(order)]]
                values = backward_sample(model, item, method='sft')
                norm = torch.nn.utils.clip_grad_norm_(trainables, 1., error_if_nonfinite=True)
                finite = bool(torch.isfinite(norm)) and all(bool(torch.isfinite(p.grad).all()) for p in trainables if p.grad is not None)
                if not finite or not all(math.isfinite(value) for value in values.values()):
                    raise FloatingPointError('Non-finite loss or gradients')
                optimizer.step()
                _synchronize(model.device)
                record = dict(**values, optimizer_step=step, samples=step, samples_this_step=1,
                    row_ids=[item['id']], finite_gradients=finite,
                    gradient_norm_before_clip=float(norm.detach()), wall_seconds=time.perf_counter()-step_start,
                    timestamp=now(), **_cuda_memory(model.device))
                log.write(json.dumps(record, allow_nan=False) + '\n')
                _status(output, status='running', phase='training', completed=step, total=args.steps, last_id=item['id'])
                if step in checkpoints:
                    checkpoint(step)
        _status(output, status='running', phase='saving_adapter', completed=args.steps, total=args.steps)
        save_file(model.adapter_state(include_projection=False), str(output / 'adapter.safetensors'))
        binding.assert_unchanged(full=True)
        final_hash = sha256_file(output / 'adapter.safetensors')
        comparison = dict(historical_sft_sha256=OLD_SFT_SHA256,
            exact_checkpoint_file_match=final_hash == OLD_SFT_SHA256,
            interpretation='Matching file SHA256 establishes byte identity; differing SHA256 does not establish numerical equivalence. '
                           'The historical 50-step result is not assumed reproduced.')
        if getattr(args, 'reference_checkpoint', None):
            from safetensors.torch import load_file
            previous = load_file(args.reference_checkpoint, device='cpu')
            current = model.adapter_state(include_projection=False)
            same_keys = set(previous) == set(current)
            comparison.update(reference_sha256=binding.hashes['reference_checkpoint'], same_parameter_keys=same_keys)
            if same_keys:
                comparison['exact_tensor_match'] = all(torch.equal(previous[n], current[n]) for n in previous)
                comparison['max_abs_tensor_difference'] = max(float((previous[n].float()-current[n].float()).abs().max()) for n in previous)
        result = dict(status='success', evaluated_steps=evaluated, optimizer_steps=args.steps,
            final_adapter_sha256=final_hash, historical_comparison=comparison,
            config_sha256=config['config_sha256'], parameter_drift=parameter_drift(model, reference),
            run_total_wall_seconds=time.perf_counter()-started, **_cuda_memory(model.device),
            interpretation=INTERPRETATION)
        _json(output / 'summary.json', result)
        _status(output, status='success', phase='complete', completed=args.steps, total=args.steps)
        return result
    except BaseException as exc:
        _fail(output, exc)
        raise


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='stage', required=True)
    data = subs.add_parser('data')
    data.add_argument('--output', required=True)
    data.add_argument('--seed', type=int, default=173)
    for name in ('capability', 'train'):
        sub = subs.add_parser(name)
        for field in ('suite', 'base-path', 'adapter-path', 'output'):
            sub.add_argument('--' + field, required=True)
        sub.add_argument('--device', default='cuda')
        if name == 'capability':
            sub.add_argument('--checkpoint', required=True)
        else:
            sub.add_argument('--dataset', required=True)
            sub.add_argument('--regime', required=True, choices=('overfit', 'replay'))
            sub.add_argument('--learning-rate', required=True, type=float)
            sub.add_argument('--steps', required=True, type=int)
            sub.add_argument('--eval-steps', required=True)
            sub.add_argument('--seed', type=int, default=73)
            sub.add_argument('--expected-dataset-hash', default=ORIGINAL_TRAIN_SHA256)
            sub.add_argument('--candidate-diagnostics', action='store_true')
            sub.add_argument('--reference-checkpoint')
    return parser


def main():
    args = argument_parser().parse_args()
    if args.stage == 'data':
        output = Path(args.output)
        if output.exists():
            raise FileExistsError('Diagnostic output already exists')
        try:
            manifest = freeze_suite(output, args.seed)
            _json(output / 'config.json', dict(stage='data', seed=args.seed,
                suite_path=str(output), source_hashes=manifest['source_hashes'],
                manifest_sha256=sha256_file(output / 'manifest.json')))
            _status(output, status='success', phase='complete', completed=sum(v['rows'] for v in manifest['splits'].values()),
                    total=sum(v['rows'] for v in manifest['splits'].values()))
        except BaseException as exc:
            if not (output / 'status.json').exists():
                _status(output, status='running', phase='data', completed=0, total=0)
            _fail(output, exc)
            raise
    elif args.stage == 'capability':
        run_capability(args)
    else:
        run_train(args)


if __name__ == '__main__':
    main()
