"""Local-only ICAE throughput benchmark; 50 steps are not an efficacy study.

Run each method in a fresh process against the same frozen JSONL and initial
adapter. No downloads are allowed. See --help for the required local paths.
"""
import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import sys
import time
import traceback

import torch
from safetensors.torch import save_file

from .data import read_jsonl, validate_rows, write_jsonl
from .metrics import evaluate_predictions
from .model import load_icae
from .objectives import binding_loss, hard_negative_loss

PREFIX = [1, 733, 16289, 28793]
SUFFIX = [733, 28748, 16289, 28793]
METHODS = ('sft', 'hard_negative', 'binding')


@dataclass
class BenchmarkConfig:
    method: str
    dataset: str
    base_path: str
    adapter_path: str
    output: str
    steps: int = 50
    seed: int = 73
    grad_accumulation: int = 1
    max_context_tokens: int = 512
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    device: str = 'cuda'
    preflight: bool = False
    dev_dataset: str | None = None
    preflight_pairs: int = 20
    max_new_tokens: int = 16


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def prepare_row(row, tokenizer, max_context_tokens=512):
    """Encode only input fields; never truncate or infer answers from metadata."""
    if not 1 <= max_context_tokens <= 512:
        raise ValueError('max_context_tokens must be between 1 and 512')
    if tokenizer.bos_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError('Tokenizer requires BOS and EOS')
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    context = [tokenizer.bos_token_id] + encode(row['context'])
    if len(context) > max_context_tokens:
        raise ValueError(f"context {row['id']} has {len(context)} tokens; limit {max_context_tokens}")
    unknown = row['kind'] == 'unknown'
    item = dict(id=row['id'], kind=row['kind'], context=context, left=list(PREFIX),
                right=encode(row['question']) + SUFFIX,
                query=[tokenizer.bos_token_id] + encode(row['question']),
                answer=encode(' ' + row['answer'].lstrip()) + [tokenizer.eos_token_id],
                foil=encode(' ' + row['foil'].lstrip()) + [tokenizer.eos_token_id])
    for key in ('support', 'hard_negative'):
        item[key] = None if unknown else [tokenizer.bos_token_id] + encode(row[key])
        if item[key] is not None and len(item[key]) > max_context_tokens:
            raise ValueError(f'{key} exceeds context token limit')
    return item


def _rng_state(device):
    return (torch.get_rng_state(), torch.cuda.get_rng_state(device) if device.type == 'cuda' else None)


def _restore_rng(state, device):
    torch.set_rng_state(state[0])
    if state[1] is not None:
        torch.cuda.set_rng_state(state[1], device)


def _sequential_auxiliary(branches, objective, device, scale):
    """Exact first-order chain rule with stochastic branch replay.

    Evaluate small branch outputs without graphs, differentiate the objective on
    those outputs, then replay and backprop each branch separately. RNG replay
    retains LoRA dropout masks. No optimizer update occurs between replays.
    """
    values, states = [], []
    for branch in branches:
        states.append(_rng_state(device))
        with torch.no_grad():
            values.append(branch().detach().requires_grad_())
    final_rng = _rng_state(device)
    auxiliary = objective(*values)
    gradients = torch.autograd.grad(auxiliary * scale, values)
    loss_value = float(auxiliary.detach())
    for branch, state, gradient in zip(branches, states, gradients):
        _restore_rng(state, device)
        result = branch()
        result.backward(gradient)
        del result
    _restore_rng(final_rng, device)
    return loss_value


def backward_sample(model, item, method, accumulation=1):
    if method not in METHODS or accumulation < 1:
        raise ValueError('Invalid method or gradient accumulation')
    ids = lambda tokens: torch.tensor([tokens], device=model.device, dtype=torch.long)

    def score(answer):
        memory = model.compress_ids(ids(item['context']))
        return model.score_ids(memory, item['left'], item['right'], [answer])

    sft = -score(item['answer']).mean()
    sft_value = float(sft.detach())
    (sft / accumulation).backward()
    del sft
    auxiliary_value = 0.0
    if item['kind'] != 'unknown' and method != 'sft':
        if method == 'binding':
            # Both answers condition on exactly the same stochastic memory.
            # Recompute the foil encoder under the correct encoder's RNG, but
            # preserve the live stream: logically the encoder ran only once.
            encoder_rng = _rng_state(model.device)

            def foil_score():
                devices = [model.device] if model.device.type == 'cuda' else []
                with torch.random.fork_rng(devices=devices):
                    _restore_rng(encoder_rng, model.device)
                    memory = model.compress_ids(ids(item['context']))
                return model.score_ids(memory, item['left'], item['right'], [item['foil']])

            branches = [lambda: score(item['answer']), foil_score]
            objective = binding_loss
        else:
            branches = [lambda: model.query_vector(ids(item['query'])),
                        lambda: model.memory_vector(model.compress_ids(ids(item['support']))),
                        lambda: model.memory_vector(model.compress_ids(ids(item['hard_negative'])))]
            objective = hard_negative_loss
        auxiliary_value = _sequential_auxiliary(branches, objective, model.device, .1 / accumulation)
    return {'sft_loss': sft_value, 'auxiliary_loss': auxiliary_value,
            'loss': sft_value + .1 * auxiliary_value}


def throughput_summary(step_seconds, samples_per_step, full_wall_seconds, warmup_steps=5):
    steady = step_seconds[warmup_steps:]
    count = len(step_seconds) * samples_per_step
    steady_samples = len(steady) * samples_per_step
    return {'optimizer_steps': len(step_seconds), 'samples': count,
            'samples_per_optimizer_step': samples_per_step,
            'warmup_optimizer_steps': min(warmup_steps, len(step_seconds)),
            'steady_optimizer_steps': len(steady), 'steady_samples': steady_samples,
            'steady_wall_seconds': sum(steady), 'full_wall_seconds': full_wall_seconds,
            'steady_samples_per_second': steady_samples / sum(steady) if steady and sum(steady) > 0 else None,
            'steady_seconds_per_optimizer_step': sum(steady) / len(steady) if steady else None,
            'full_samples_per_second': count / full_wall_seconds if full_wall_seconds > 0 else None}


def _synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _cuda_memory(device):
    if device.type != 'cuda':
        return {'peak_allocated_mib': 0., 'peak_reserved_mib': 0.}
    return {'peak_allocated_mib': torch.cuda.max_memory_allocated(device) / 2**20,
            'peak_reserved_mib': torch.cuda.max_memory_reserved(device) / 2**20}


def _gpu_metadata(device):
    if device.type != 'cuda':
        return {'device': str(device), 'name': 'CPU', 'cuda_available': torch.cuda.is_available()}
    props = torch.cuda.get_device_properties(device)
    free, total = torch.cuda.mem_get_info(device)
    return {'device': str(device), 'name': props.name, 'total_mib': total / 2**20,
            'baseline_free_mib': free / 2**20,
            'baseline_allocated_mib': torch.cuda.memory_allocated(device) / 2**20,
            'baseline_reserved_mib': torch.cuda.memory_reserved(device) / 2**20,
            'compute_capability': [props.major, props.minor], 'torch_cuda': torch.version.cuda}


def _versions():
    result = {'python': sys.version}
    for package in ('torch', 'transformers', 'peft', 'safetensors', 'huggingface-hub', 'sentencepiece', 'numpy'):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


@torch.no_grad()
def _full_context_generate(model, item, eos_id, max_new_tokens):
    # BOS belongs to the official instruction prefix, so remove context's BOS.
    tokens = item['left'] + item['context'][1:] + item['right']
    inputs = torch.tensor([tokens], device=model.device)
    generated, cache = [], None
    with model.icae.disable_adapter():
        for _ in range(max_new_tokens):
            output = model.base(input_ids=inputs, past_key_values=cache, use_cache=True)
            token = int(output.logits[0, -1].argmax())
            if token == eos_id:
                break
            generated.append(token)
            cache = output.past_key_values
            inputs = torch.tensor([[token]], device=model.device)
    return generated


def run_preflight(model, tokenizer, rows, output, max_context_tokens=512, pairs=20, max_new_tokens=16):
    validate_rows(rows)
    if pairs < 20:
        raise ValueError('Preflight requires at least 20 complete dev pairs')
    pair_ids = list(dict.fromkeys(r['pair_id'] for r in rows if r['pair_id'] is not None))
    if len(pair_ids) < pairs or any(r['split'] != 'dev' for r in rows):
        raise ValueError('Preflight requires at least 20 complete dev pairs from dev data')
    selected = set(pair_ids[:pairs])
    chosen = [r for r in rows if r['pair_id'] in selected]
    items = [prepare_row(r, tokenizer, max_context_tokens) for r in chosen]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    predictions, compressed_predictions, saved = [], [], []
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for row, item in zip(chosen, items):
                full = _full_context_generate(model, item, tokenizer.eos_token_id, max_new_tokens)
                memory = model.compress_ids(torch.tensor([item['context']], device=model.device))
                compressed = model.generate_ids(memory, item['left'], item['right'], tokenizer.eos_token_id, max_new_tokens)
                full_text = tokenizer.decode(full, skip_special_tokens=True)
                compressed_text = tokenizer.decode(compressed, skip_special_tokens=True)
                predictions.append({'id': row['id'], 'prediction': full_text})
                compressed_predictions.append({'id': row['id'], 'prediction': compressed_text})
                saved.append({'id': row['id'], 'answer': row['answer'], 'foil': row['foil'],
                              'full_context': full_text, 'icae': compressed_text})
    finally:
        model.train(was_training)
    metrics = {'full_context': evaluate_predictions(chosen, predictions),
               'icae': evaluate_predictions(chosen, compressed_predictions),
               'interpretation': 'Generation diagnostic only. Low accuracy does not prevent a throughput benchmark; no effectiveness claim.'}
    write_jsonl(output / 'preflight_predictions.jsonl', saved)
    _json(output / 'preflight_metrics.json', metrics)
    return metrics


def run_benchmark(config, model=None, tokenizer=None):
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    status = {'status': 'partial', 'optimizer_steps': 0, 'samples': 0, 'phase': 'setup'}
    _json(output / 'status.json', status)
    metadata = asdict(config)
    _json(output / 'config.json', metadata)
    try:
        if config.method not in METHODS or min(config.steps, config.grad_accumulation) < 1:
            raise ValueError('Positive optimizer steps/accumulation and valid method required')
        if config.learning_rate <= 0 or config.weight_decay < 0 or config.max_grad_norm <= 0:
            raise ValueError('Invalid optimizer settings')
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        metadata.update(dataset_sha256=sha256_file(config.dataset),
                        adapter_sha256=sha256_file(config.adapter_path), versions=_versions())
        _json(output / 'config.json', metadata)
        rows = read_jsonl(config.dataset)
        validate_rows(rows)
        if not rows:
            raise ValueError('Empty frozen dataset')
        if model is None:
            model, tokenizer = load_icae(config.base_path, config.adapter_path, device=config.device, dtype=torch.bfloat16, memory_size=128)
        if tokenizer is None:
            raise ValueError('Tokenizer required')
        items = [prepare_row(row, tokenizer, config.max_context_tokens) for row in rows]
        # Projection participates exclusively in the hard-negative objective.
        model.retrieval_proj.requires_grad_(config.method == 'hard_negative')
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        if model.base.is_gradient_checkpointing:
            raise ValueError('Checkpointing across shared adapter toggles is forbidden')
        trainables = [p for p in model.parameters() if p.requires_grad]
        device = model.device
        metadata.update(trainable_parameters=sum(p.numel() for p in trainables),
                        total_parameters=sum(p.numel() for p in model.parameters()),
                        trainable_dtypes=sorted({str(p.dtype) for p in trainables}),
                        base_dtype=str(model.base.get_input_embeddings().weight.dtype),
                        gpu=_gpu_metadata(device),
                        prompt={'prefix': PREFIX, 'suffix': SUFFIX, 'learned_ft_token': True,
                                'context_bos': True, 'answer_leading_space': True, 'answer_eos': True},
                        memory_slots=model.memory_size,
                        lora_ranks=sorted({p.r['default'] for p in model.modules() if hasattr(p, 'r') and isinstance(p.r, dict) and 'default' in p.r}),
                        microbatch=1, optimizer='AdamW', auxiliary_weight=.1,
                        optimizer_settings={'lr': config.learning_rate,
                            'weight_decay': config.weight_decay, 'betas': [.9, .999],
                            'eps': 1e-8, 'amsgrad': False, 'foreach': None, 'fused': None},
                        binding_margin=.5, infonce_temperature=.07,
                        auxiliary_execution='no-grad branch values, small objective derivatives, sequential branch replays; binding replays identical encoder dropout for both answers',
                        sample_order='seeded shuffle, cyclic, identical across arms',
                        timing='CUDA synchronized optimizer steps including all accumulation microbatches; full wall includes optimizer construction, excludes load/preflight/save',
                        interpretation='Engineering throughput only; 50 steps do not establish scientific improvement.')
        if config.preflight:
            if not config.dev_dataset:
                raise ValueError('--preflight requires --dev-dataset')
            metadata['dev_dataset_sha256'] = sha256_file(config.dev_dataset)
            _json(output / 'config.json', metadata)
            status['phase'] = 'preflight'
            _json(output / 'status.json', status)
            run_preflight(model, tokenizer, read_jsonl(config.dev_dataset), output,
                          config.max_context_tokens, config.preflight_pairs, config.max_new_tokens)
        _json(output / 'config.json', metadata)
        model.train()
        order = list(range(len(items)))
        random.Random(config.seed).shuffle(order)
        _synchronize(device)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        training_start = time.perf_counter()
        optimizer = torch.optim.AdamW(trainables, **metadata['optimizer_settings'])
        step_seconds = []
        with (output / 'steps.jsonl').open('w', encoding='utf-8', buffering=1) as log:
            for step in range(config.steps):
                _synchronize(device)
                step_start = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                losses = dict(loss=0., sft_loss=0., auxiliary_loss=0.)
                row_ids = []
                for micro in range(config.grad_accumulation):
                    index = order[(step * config.grad_accumulation + micro) % len(order)]
                    item = items[index]
                    values = backward_sample(model, item, config.method, config.grad_accumulation)
                    row_ids.append(item['id'])
                    for key, value in values.items():
                        losses[key] += value / config.grad_accumulation
                norm = torch.nn.utils.clip_grad_norm_(trainables, config.max_grad_norm, error_if_nonfinite=True)
                finite = bool(torch.isfinite(norm)) and all(bool(torch.isfinite(p.grad).all()) for p in trainables if p.grad is not None)
                if not finite or not all(torch.isfinite(torch.tensor(x)) for x in losses.values()):
                    raise FloatingPointError('Non-finite loss or gradients')
                optimizer.step()
                _synchronize(device)
                duration = time.perf_counter() - step_start
                step_seconds.append(duration)
                record = {**losses, 'optimizer_step': step + 1,
                          'samples': (step + 1) * config.grad_accumulation,
                          'samples_this_step': config.grad_accumulation, 'row_ids': row_ids,
                          'finite_gradients': finite, 'gradient_norm_before_clip': float(norm.detach()),
                          'wall_seconds': duration, 'warmup': step < 5, **_cuda_memory(device)}
                log.write(json.dumps(record, allow_nan=False) + '\n')
                status.update(optimizer_steps=step + 1, samples=record['samples'], phase='training',
                              elapsed_seconds=time.perf_counter() - start, **_cuda_memory(device))
                _json(output / 'status.json', status)
                print(json.dumps(record), flush=True)
        _synchronize(device)
        full_wall = time.perf_counter() - training_start
        status['phase'] = 'saving_adapter'
        _json(output / 'status.json', status)
        save_file(model.adapter_state(include_projection=config.method == 'hard_negative'), str(output / 'adapter.safetensors'))
        summary = {**throughput_summary(step_seconds, config.grad_accumulation, full_wall),
                   **_cuda_memory(device), 'method': config.method, 'status': 'success',
                   'run_total_wall_seconds': time.perf_counter() - start,
                   'adapter_sha256': sha256_file(output / 'adapter.safetensors'),
                   'interpretation': metadata['interpretation']}
        _json(output / 'summary.json', summary)
        status.update(status='success', phase='complete', elapsed_seconds=time.perf_counter() - start)
        _json(output / 'status.json', status)
        return summary
    except BaseException as error:
        status.update(status='failed', error_type=type(error).__name__, error=str(error),
                      oom=isinstance(error, torch.cuda.OutOfMemoryError), elapsed_seconds=time.perf_counter() - start)
        _json(output / 'status.json', status)
        (output / 'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=METHODS, required=True)
    for name in ('dataset', 'base-path', 'adapter-path', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--steps', type=int, default=50, help='Optimizer steps (not samples)')
    parser.add_argument('--seed', type=int, default=73)
    parser.add_argument('--grad-accumulation', type=int, default=1, help='Samples per optimizer step; microbatch is fixed at 1')
    parser.add_argument('--max-context-tokens', type=int, default=512)
    parser.add_argument('--learning-rate', type=float, default=5e-5)
    parser.add_argument('--weight-decay', type=float, default=0.)
    parser.add_argument('--max-grad-norm', type=float, default=1.)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--dev-dataset')
    parser.add_argument('--preflight-pairs', type=int, default=20)
    parser.add_argument('--max-new-tokens', type=int, default=16)
    return parser


def main():
    config = BenchmarkConfig(**vars(argument_parser().parse_args()))
    try:
        summary = run_benchmark(config)
    except Exception as error:
        print(f'{type(error).__name__}: {error}', file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
