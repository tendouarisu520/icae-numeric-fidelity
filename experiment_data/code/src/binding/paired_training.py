"""Protocol-v2 five-row scenario updates and strict developmental stability gates.

Fresh CLI entry configures CUBLAS before importing Torch. Runtime policy is an
explicit required choice; unsupported deterministic operations fail closed.
Final-test inputs are never opened by this baseline runner.
"""
import argparse
import collections
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import time
import traceback

from .paired_data import LEVELS, audit_suite, canonical_hash, load_suite
from .repro_probe import configure_environment, configure_torch, _attention_context, runtime_settings, runtime_snapshot, tensor_snapshot

PROFILES = ('det_auto', 'det_math')
METHODS = ('sft', 'hard_negative', 'binding')


@dataclass
class TrainConfig:
    suite: str
    base_path: str
    adapter_path: str
    output: str
    profile: str
    method: str = 'sft'
    seed: int = 73
    steps: int = 128
    learning_rate: float = 1e-5
    eval_steps: str | None = None
    device: str = 'cuda'


def optimizer_settings(learning_rate):
    return dict(lr=learning_rate, weight_decay=0., betas=(.9, .999), eps=1e-8,
                amsgrad=False, foreach=None, fused=None)


def seed_sample(training_seed, unit_index, row_id):
    """Independent SHA256 row seed isolates primary dropout from auxiliary draws."""
    import numpy as np
    import torch
    seed = int(canonical_hash([int(training_seed), int(unit_index), str(row_id)])[:16], 16) % (2**63)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def build_schedule(rows, seed, steps):
    if not 1 <= steps <= 224:
        raise ValueError('Schedule steps must be in 1..224')
    audit_suite({'train': rows}, splits=('train',))
    groups = collections.defaultdict(list)
    for row in rows:
        groups[row['scenario_id']].append(row)
    all_ids = sorted(groups)
    d0 = [sid for sid in all_ids if groups[sid][0]['difficulty'] == 0]
    random.Random(canonical_hash([seed, 'curriculum'])).shuffle(d0)
    order = [d0[i % len(d0)] for i in range(32)]
    for epoch in range(2):
        mixed = list(all_ids)
        random.Random(canonical_hash([seed, 'mixed', epoch])).shuffle(mixed)
        order.extend(mixed)
    schedule = []
    for index, scenario in enumerate(order[:steps]):
        row_ids = sorted(r['id'] for r in groups[scenario])
        random.Random(canonical_hash([seed, 'row_order', index, scenario])).shuffle(row_ids)
        schedule.append(dict(unit_index=index, scenario_id=scenario, row_ids=row_ids,
                             phase='curriculum_d0' if index < 32 else 'mixed_epoch_' + str((index-32)//96+1)))
    return schedule


def train_unit(model, items, optimizer, method, seed, unit_index):
    """One original clipped AdamW update on the exact five-example mean.

    Timing includes seed resets, forward/backward, finite checks, clipping and
    AdamW, bounded by device synchronization. No evaluation/hash/save/log I/O.
    """
    import torch
    from .benchmark import backward_sample, _synchronize
    if method not in METHODS or len(items) != 5 or len({i['id'] for i in items}) != 5:
        raise ValueError('A complete five-row update and valid method are required')
    if sum(i['kind'] == 'unknown' for i in items) != 1:
        raise ValueError('An update must contain four known rows and one UNKNOWN')
    trainables = [p for p in model.parameters() if p.requires_grad]
    _synchronize(model.device)
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    losses = dict(loss=0., sft_loss=0., auxiliary_loss=0.)
    for item in items:
        seed_sample(seed, unit_index, item['id'])
        values = backward_sample(model, item, method, accumulation=5)
        for key in losses:
            losses[key] += values[key] / 5
    norm = torch.nn.utils.clip_grad_norm_(trainables, 1., error_if_nonfinite=True)
    finite = math.isfinite(float(norm)) and all(bool(torch.isfinite(p.grad).all()) for p in trainables if p.grad is not None)
    if not finite or not all(math.isfinite(v) for v in losses.values()):
        raise FloatingPointError('Non-finite loss or gradients')
    optimizer.step()
    if not all(bool(torch.isfinite(p).all()) for p in trainables):
        raise FloatingPointError('Non-finite parameters after update')
    _synchronize(model.device)
    return dict(optimizer_step=unit_index+1, samples_this_step=5, row_ids=[i['id'] for i in items],
        losses=losses, finite_gradients=finite, gradient_norm_before_clip=float(norm),
        training_seconds=time.perf_counter()-started)


def _indexed(rows, predictions):
    from .metrics import _index
    indexed = _index(rows, predictions)
    if len(indexed) != len(rows):
        raise ValueError('Complete predictions required')
    return indexed


def _subset_summary(rows, predictions):
    from .metrics import evaluate_predictions, normalize
    from .eval_metrics import diagnose_output
    indexed = _indexed(rows, predictions)
    result = evaluate_predictions(rows, predictions)
    known = [r for r in rows if r['kind'] == 'swap']
    unknown = [r for r in rows if r['kind'] == 'unknown']
    correct = lambda r: normalize(indexed[r['id']]['prediction']) == normalize(r['answer'])
    result.update(known_rows=len(known), unknown_rows=len(unknown), scenarios=len({r['scenario_id'] for r in rows}),
        known_correct=sum(correct(r) for r in known), unknown_correct=sum(correct(r) for r in unknown),
        known_accuracy=sum(correct(r) for r in known)/len(known) if known else None,
        known_format_valid=sum(diagnose_output(indexed[r['id']]['prediction'], indexed[r['id']].get('generation', {}).get('stop_reason') == 'max_new_tokens')['category'] == 'exact_single_label' for r in known),
        unknown_format_valid=sum(diagnose_output(indexed[r['id']]['prediction'], indexed[r['id']].get('generation', {}).get('stop_reason') == 'max_new_tokens')['category'] == 'exact_single_label' for r in unknown))
    return result


def stability_gate(rows, predictions, finite):
    """Engineering readiness only, using known-only d0 and separate format rates."""
    from .metrics import normalize, pair_scores
    from .eval_metrics import diagnose_output
    audit_suite({'dev': rows}, splits=('dev',))
    indexed = _indexed(rows, predictions)
    known = [r for r in rows if r['kind'] == 'swap']
    unknown = [r for r in rows if r['kind'] == 'unknown']
    d0 = [r for r in known if r['difficulty'] == 0]
    checks = {}
    def check(name, numerator, denominator, threshold):
        checks[name] = dict(numerator=numerator, denominator=denominator,
            value=numerator/denominator, threshold=threshold, passed=numerator/denominator >= threshold)
    def accuracy(name, subset, threshold):
        check(name, sum(normalize(indexed[r['id']]['prediction']) == normalize(r['answer']) for r in subset), len(subset), threshold)
    pairs = pair_scores(d0, [indexed[r['id']] for r in d0])
    check('d0_pair_accuracy', sum(pairs.values()), len(pairs), .875)
    accuracy('d0_known_accuracy', d0, .90)
    accuracy('unknown_accuracy', unknown, .75)
    for level in (0, 12):
        accuracy(f'd{level}_unknown_accuracy', [r for r in unknown if r['difficulty'] == level], .75)
    for name, subset in (('known_format', known), ('unknown_format', unknown)):
        # Recompute from raw text with the frozen parser; never trust edited
        # diagnostic metadata as a more permissive primary prediction.
        check(name, sum(diagnose_output(indexed[r['id']]['prediction'],
            indexed[r['id']].get('generation', {}).get('stop_reason') == 'max_new_tokens')['category'] == 'exact_single_label' for r in subset), len(subset), .95)
    return dict(passed=bool(finite) and all(c['passed'] for c in checks.values()), finite_gradients=bool(finite), checks=checks,
        interpretation='Engineering stability gate for this seed, not evidence of scientific efficacy.')


def evaluate_checkpoint(model, tokenizer, rows, output, finite, method='sft', evaluate_fn=None):
    """Strict frozen evaluator, raw outputs and exact level/gate denominators."""
    from .diagnostics import evaluate_split, evaluation_state
    from .benchmark import _json, sha256_file
    from .data import read_jsonl
    audit_suite({'dev': rows}, splits=('dev',))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    evaluate_fn = evaluate_split if evaluate_fn is None else evaluate_fn
    before = runtime_snapshot(model)
    before_weights = tensor_snapshot(model)
    result = dict(by_level={}, evaluator_summaries={}, prediction_hashes={})
    predictions = []
    try:
        with evaluation_state(model):
            for level in LEVELS:
                name = f'd{level}'
                chosen = [r for r in rows if r['difficulty'] == level]
                destination = output / name
                result['evaluator_summaries'][name] = evaluate_fn(model, tokenizer, chosen, method, destination, candidates=False)
                saved = read_jsonl(destination / 'predictions.jsonl')
                result['by_level'][name] = _subset_summary(chosen, saved)
                result['prediction_hashes'][name] = sha256_file(destination / 'predictions.jsonl')
                predictions.extend(saved)
    finally:
        after = runtime_snapshot(model)
        after_weights = tensor_snapshot(model)
        state = dict(before=before, after=after, weights_before=before_weights, weights_after=after_weights,
                     preserved=before == after and before_weights == after_weights)
        _json(output / 'evaluation_state_audit.json', state)
        if not state['preserved']:
            raise RuntimeError('Evaluation changed training state or parameters')
    result.update(pooled=_subset_summary(rows, predictions), gate=stability_gate(rows, predictions, finite),
                  checkpoint_tensor_sha256=before_weights['sha256'], evaluation_state_preserved=True)
    _json(output / 'summary.json', result)
    return result


def _checkpoint_identity(state):
    from .repro_probe import _tensor_bytes
    entries = {}
    for name, tensor in sorted(state.items()):
        entries[name] = dict(shape=list(tensor.shape), dtype=str(tensor.dtype), sha256=hashlib.sha256(_tensor_bytes(tensor)).hexdigest())
    return dict(sha256=canonical_hash(entries), tensors=entries)


def _input_files(config):
    from .diagnostics import _base_files
    from .evaluation import RUNTIME_SOURCES
    names = tuple(dict.fromkeys((*RUNTIME_SOURCES, 'diagnostics.py', 'diagnostics_data.py', 'repro_probe.py', 'paired_data.py', 'paired_training.py')))
    files = {f'source/{name}': Path(__file__).with_name(name) for name in names}
    files.update({f'suite/{name}': Path(config.suite)/name for name in ('manifest.json', 'semantic_audit.json', 'train.jsonl', 'dev.jsonl')})
    files['initial_adapter'] = Path(config.adapter_path)
    base = Path(config.base_path)
    files.update({f'base/{p.relative_to(base) if base.is_dir() else p.name}': p for p in _base_files(base)})
    return files


def _eval_steps(config, injected):
    values = ([128] if config.steps == 128 else [128, 224]) if config.eval_steps is None else [int(v) for v in config.eval_steps.split(',')]
    if len(values) != len(set(values)) or any(v < 0 or v > config.steps for v in values):
        raise ValueError('Duplicate or out-of-range evaluation steps')
    if not injected and (config.steps not in (128, 224) or any(v not in (0, 128, 224) for v in values)):
        raise ValueError('Baseline requires a fresh 128- or 224-step trajectory')
    return sorted(set(values) | {config.steps})


def run_train(config, model=None, tokenizer=None):
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=False)
    environment = configure_environment(config.profile)
    import numpy as np
    import torch
    from safetensors.torch import save_file
    from . import benchmark
    write = benchmark._json
    started = time.perf_counter()
    status = dict(status='running', phase='validation', optimizer_steps=0, samples=0)
    metadata = dict(protocol_version=2, arguments=asdict(config), environment=environment, injected_model=model is not None)
    write(output/'status.json', status)
    write(output/'config.json', metadata)
    try:
        if config.profile not in PROFILES or config.method not in METHODS:
            raise ValueError('Explicit strict profile and valid method required')
        if not math.isfinite(config.learning_rate) or config.learning_rate <= 0:
            raise ValueError('Learning rate must be finite and positive')
        # A CPU fixture may share a host/process with prior CUDA users. Inspect
        # its actual device; only a CUDA-target run needs fresh initialization.
        target_device = torch.device(model.device if model is not None else config.device)
        if target_device.type == 'cuda' and environment['torch_preimported'] and torch.cuda.is_initialized():
            raise RuntimeError('Deterministic CUDA training requires a fresh CLI process before Torch import')
        checkpoints = _eval_steps(config, model is not None)
        configure_torch(config.profile)
        suite = load_suite(config.suite, splits=('train', 'dev'))
        schedule = build_schedule(suite['train'], config.seed, config.steps)
        files = _input_files(config)
        hashes = {name: benchmark.sha256_file(path) for name, path in files.items()}
        stats = {name: (path.stat().st_size, path.stat().st_mtime_ns) for name, path in files.items()}
        def unchanged(full=False):
            for name, path in files.items():
                if (path.stat().st_size, path.stat().st_mtime_ns) != stats[name]:
                    raise ValueError('Input changed: ' + name)
                if (full or path.stat().st_size < 8*1024*1024) and benchmark.sha256_file(path) != hashes[name]:
                    raise ValueError('Input hash changed: ' + name)
        source_hashes = {k[7:]: v for k, v in hashes.items() if k.startswith('source/')}
        data_hashes = {k[6:]: v for k, v in hashes.items() if k.startswith('suite/')}
        metadata.update(input_hashes=hashes, source_hashes=source_hashes, data_hashes=data_hashes,
            versions=benchmark._versions(), eval_steps=checkpoints, order_sha256=canonical_hash(schedule),
            unit_order_sha256=canonical_hash([u['scenario_id'] for u in schedule]),
            row_order_sha256=canonical_hash([u['row_ids'] for u in schedule]),
            optimizer_settings=optimizer_settings(config.learning_rate), microbatch=1, accumulation=5, max_grad_norm=1.,
            auxiliary_weight=.1, binding_margin=.5, infonce_temperature=.07,
            stochastic_policy='v2: seed Python/NumPy/Torch CPU/CUDA before load; reset each to SHA256([training_seed,zero_based_unit_index,row_id]) first 64 bits modulo 2^63 (NumPy modulo 2^32) before each original backward_sample; primary SFT dropout matched across methods; auxiliary RNG cannot perturb next row.',
            timing='Sum CUDA-synchronized five-row updates only; seed reset/backward/finite checks/clip/AdamW included; load/eval/save/hash/log excluded.',
            scenario_cluster='One five-row scenario has two pairs; use one scenario cluster.',
            test_policy='Only train/dev JSONL opened; final test remains reserved.', prompt_variant='legacy', max_context_tokens=512,
            memory_slots_required=128, base_dtype_required='bfloat16', trainable_dtype_required='float32',
            projection_difference='Projection trainable/saved exclusively for hard_negative; extra compute is reported, not claimed equivalent.')
        write(output/'schedule.json', schedule)
        random.seed(config.seed); np.random.seed(config.seed % (2**32)); torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        status.update(phase='load_model'); write(output/'status.json', status)
        if model is None:
            model, tokenizer = benchmark.load_icae(config.base_path, config.adapter_path,
                device=config.device, dtype=torch.bfloat16, memory_size=128)
        if tokenizer is None:
            raise ValueError('Tokenizer required')
        model.retrieval_proj.requires_grad_(config.method == 'hard_negative')
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        if model.base.is_gradient_checkpointing:
            raise ValueError('Shared adapter checkpointing forbidden')
        if not metadata['injected_model'] and (model.memory_size != 128 or model.base.get_input_embeddings().weight.dtype != torch.bfloat16):
            raise ValueError('Production requires 128 memory slots and BF16 base')
        trainables = [p for p in model.parameters() if p.requires_grad]
        prepared = {r['id']: benchmark.prepare_row(r, tokenizer, 512) for r in suite['train']}
        # Preflight *all* development contexts too: never silently truncate.
        dev_items = [benchmark.prepare_row(r, tokenizer, 512) for r in suite['dev']]
        model.train()
        core = _checkpoint_identity(model.adapter_state(include_projection=False))
        metadata.update(initial_core_sha256=core['sha256'], initial_tensor_identity=_checkpoint_identity(model.adapter_state(include_projection=config.method == 'hard_negative')),
            initial_state=runtime_snapshot(model), initial_trainables=tensor_snapshot(model), gpu=benchmark._gpu_metadata(model.device),
            memory_slots=model.memory_size, actual_base_dtype=str(model.base.get_input_embeddings().weight.dtype),
            trainable_dtypes=sorted({str(p.dtype) for p in trainables}), trainable_parameters=sum(p.numel() for p in trainables),
            total_parameters=sum(p.numel() for p in model.parameters()), prepared_train_sha256=canonical_hash(list(prepared.values())),
            prepared_dev_sha256=canonical_hash(dev_items), maximum_context_tokens=max(len(i['context']) for i in list(prepared.values())+dev_items),
            attention_implementation=getattr(model.base.config, '_attn_implementation', None),
            attention_modules={n: type(m).__name__ for n, m in model.named_modules() if n.endswith('self_attn')},
            dropout_probabilities={n: m.p for n, m in model.named_modules() if isinstance(m, torch.nn.Dropout)},
            lora_ranks=sorted({m.r['default'] for m in model.modules() if isinstance(getattr(m, 'r', None), dict) and 'default' in m.r}),
            lora_alpha=32, lora_dropout=.05, lora_target_modules=['q_proj', 'v_proj'])
        optimizer = torch.optim.AdamW(trainables, **optimizer_settings(config.learning_rate))
        evaluations, seconds, finite = {}, [], True
        if model.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(model.device)
        with _attention_context(config.profile):
            metadata['runtime'] = runtime_settings()
            metadata['config_sha256'] = canonical_hash(metadata)
            write(output/'config.json', metadata)
            if 0 in checkpoints:
                evaluations['0'] = evaluate_checkpoint(model, tokenizer, suite['dev'], output/'evaluation-step-0000', finite, config.method)
                unchanged()
            with (output/'steps.jsonl').open('x', encoding='utf-8', buffering=1) as log:
                for unit in schedule:
                    record = train_unit(model, [prepared[rid] for rid in unit['row_ids']], optimizer, config.method, config.seed, unit['unit_index'])
                    seconds.append(record['training_seconds']); finite = finite and record['finite_gradients']
                    step = unit['unit_index']+1
                    record.update(scenario_id=unit['scenario_id'], phase=unit['phase'], samples=step*5,
                        cumulative_training_seconds=math.fsum(seconds), **benchmark._cuda_memory(model.device))
                    log.write(json.dumps(record, allow_nan=False)+'\n')
                    status.update(phase='training', optimizer_steps=step, samples=step*5,
                        cumulative_training_seconds=math.fsum(seconds), elapsed_seconds=time.perf_counter()-started)
                    write(output/'status.json', status)
                    print(json.dumps(record, allow_nan=False), flush=True)
                    if step in checkpoints:
                        unchanged()
                        status.update(phase='evaluation', evaluation_step=step); write(output/'status.json', status)
                        evaluations[str(step)] = evaluate_checkpoint(model, tokenizer, suite['dev'], output/f'evaluation-step-{step:04d}', finite, config.method)
                        write(output/'evaluations.json', evaluations)
                        unchanged()
        status.update(phase='save_final_adapter'); write(output/'status.json', status)
        state = model.adapter_state(include_projection=config.method == 'hard_negative')
        checkpoint_identity = _checkpoint_identity(state)
        save_file(state, str(output/'adapter.safetensors'))
        unchanged(full=True)
        result = dict(status='success', protocol_version=2, method=config.method, seed=config.seed, profile=config.profile,
            optimizer_steps=len(schedule), samples=len(schedule)*5, cumulative_training_seconds=math.fsum(seconds),
            training_step_seconds=seconds, finite_gradients=finite, evaluations=evaluations,
            final_gate=evaluations[str(config.steps)]['gate'], config_sha256=metadata['config_sha256'],
            source_hashes=source_hashes, data_hashes=data_hashes, input_hashes=hashes, runtime=metadata['runtime'],
            initial_core_sha256=core['sha256'], initial_tensor_identity=metadata['initial_tensor_identity'],
            order_sha256=metadata['order_sha256'], unit_order_sha256=metadata['unit_order_sha256'], row_order_sha256=metadata['row_order_sha256'],
            adapter_sha256=benchmark.sha256_file(output/'adapter.safetensors'), adapter_tensor_identity=checkpoint_identity,
            run_wall_seconds=time.perf_counter()-started, **benchmark._cuda_memory(model.device),
            interpretation='Baseline developmental readiness only; no final-test evaluation or scientific efficacy claim.')
        write(output/'summary.json', result)
        status.update(status='success', phase='complete'); write(output/'status.json', status)
        return result
    except BaseException as exc:
        status.update(status='failed', error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        write(output/'status.json', status)
        write(output/'summary.json', status)
        (output/'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    freeze = commands.add_parser('freeze')
    freeze.add_argument('--output', required=True); freeze.add_argument('--seed', type=int, default=73)
    train = commands.add_parser('train')
    for name in ('suite', 'base-path', 'adapter-path', 'output'):
        train.add_argument('--'+name, required=True)
    train.add_argument('--profile', choices=PROFILES, required=True)
    train.add_argument('--method', choices=METHODS, default='sft')
    train.add_argument('--seed', type=int, default=73)
    train.add_argument('--steps', type=int, choices=(128,224), default=128)
    train.add_argument('--learning-rate', type=float, default=1e-5)
    train.add_argument('--eval-steps')
    train.add_argument('--device', default='cuda')
    return parser


def main():
    args = vars(argument_parser().parse_args())
    command = args.pop('command')
    if command == 'freeze':
        from .paired_data import freeze_suite
        result = freeze_suite(**args)
    else:
        result = run_train(TrainConfig(**args))
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
