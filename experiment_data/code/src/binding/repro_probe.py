"""Protocol-v2 numerical localization of the original first SFT updates.

This module deliberately imports no Torch-dependent modules at module scope:
the CLI configures CUBLAS before the first Torch import. No downloads, checkpoint
saves, resume, backend fallback, or execution-policy selection occur here.
"""
import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import random
import sys
import traceback


ORIGINAL_TRAIN_SHA256 = '882c179ceec6d832c5136181eeef6eaf9fbf76912c4dc9808bbfb63200994763'
PROFILES = ('default', 'det_auto', 'det_math')
OPTIMIZER_SETTINGS = dict(lr=5e-5, weight_decay=0., betas=(.9, .999), eps=1e-8,
                          amsgrad=False, foreach=None, fused=None)


@dataclass
class ProbeConfig:
    base_path: str
    adapter_path: str
    dataset: str
    output: str
    profile: str = 'default'
    steps: int = 3
    seed: int = 73
    initial_eval_suite: str | None = None
    device: str = 'cuda'
    expected_dataset_hash: str = ORIGINAL_TRAIN_SHA256


def configure_environment(profile):
    if profile not in PROFILES:
        raise ValueError('Unknown execution profile')
    previous = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    preimported = 'torch' in sys.modules
    if profile != 'default':
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    return dict(prior_cublas_workspace_config=previous,
                cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
                torch_preimported=preimported,
                pythonhashseed=os.environ.get('PYTHONHASHSEED'))


def configure_torch(profile):
    import torch
    if profile not in PROFILES:
        raise ValueError('Unknown execution profile')
    if profile != 'default':
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def runtime_settings():
    import torch
    return dict(deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        sdpa_math=torch.backends.cuda.math_sdp_enabled(),
        sdpa_flash=torch.backends.cuda.flash_sdp_enabled(),
        sdpa_mem_efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),
        sdpa_cudnn=torch.backends.cuda.cudnn_sdp_enabled(),
        math_sdpa_allow_fp16_bf16_reduction=torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(),
        cpu_threads=torch.get_num_threads(), cpu_interop_threads=torch.get_num_interop_threads())


def _attention_context(profile):
    if profile == 'det_math':
        from torch.nn.attention import SDPBackend, sdpa_kernel
        return sdpa_kernel(SDPBackend.MATH)
    return nullcontext()


def _canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _tensor_bytes(tensor):
    import torch
    # The view supports BF16 and encodes actual storage bytes without conversion.
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def _group(name):
    return ('memory_embedding' if name.startswith('memory_token_embed.') else
            'lora' if 'lora_' in name else 'other')


def _number(value):
    return value if math.isfinite(value) else str(value)


def tensor_snapshot(model, gradients=False):
    """Sorted, framed name/dtype/shape/bytes hashes; one CPU tensor at a time.

    Absent gradients have an explicit marker. Canonical norms square CPU float64
    values before reduction and combine per-parameter sums using math.fsum.
    Neither model tensors nor gradients are modified or retained by this helper.
    """
    import torch
    digests = {name: hashlib.sha256() for name in ('all', 'memory_embedding', 'lora', 'other')}
    parameters, squares, nonzero, finite, with_grad = {}, [], 0, True, 0
    for name, parameter in sorted(model.named_parameters()):
        if not parameter.requires_grad:
            continue
        tensor = parameter.grad if gradients else parameter
        entry = dict(dtype=str(parameter.dtype), shape=list(parameter.shape), group=_group(name))
        if gradients:
            entry['gradient_dtype'] = str(tensor.dtype) if tensor is not None else None
            entry['gradient_shape'] = list(tensor.shape) if tensor is not None else None
        header = _canonical_bytes(dict(name=name, dtype=str(tensor.dtype) if tensor is not None else None,
            shape=list(tensor.shape) if tensor is not None else None, absent=tensor is None))
        digest = hashlib.sha256()
        for target in (digest, digests['all'], digests[entry['group']]):
            target.update(len(header).to_bytes(8, 'big'))
            target.update(header)
        if tensor is not None:
            cpu = tensor.detach().cpu().contiguous()
            raw = _tensor_bytes(cpu)
            for target in (digest, digests['all'], digests[entry['group']]):
                target.update(len(raw).to_bytes(8, 'big'))
                target.update(raw)
            if gradients:
                entry['finite'] = bool(torch.isfinite(cpu).all())
                entry['nonzero_elements'] = int(torch.count_nonzero(cpu))
                squared = float(cpu.to(torch.float64).square().sum())
                entry['sum_squares_float64'] = _number(squared)
                squares.append(squared)
                nonzero += entry['nonzero_elements']
                finite = finite and entry['finite']
                with_grad += 1
            del cpu, raw
        entry['sha256'] = digest.hexdigest()
        parameters[name] = entry
    result = dict(sha256=digests['all'].hexdigest(),
        groups={key: digest.hexdigest() for key, digest in digests.items() if key != 'all'},
        parameters=parameters)
    if gradients:
        squared = math.fsum(squares)
        result.update(canonical_l2_norm=_number(math.sqrt(squared)),
            sum_squares_float64=_number(squared), finite=finite, nonzero_elements=nonzero,
            parameters_with_grad=with_grad, nonzero_gradient_parameters=sum(
                entry.get('nonzero_elements', 0) > 0 for entry in parameters.values()))
    return result


def runtime_snapshot(model):
    """Hash all stochastic streams and save every module/parameter mode flag."""
    import numpy as np
    import torch
    rng = dict(python=hashlib.sha256(pickle.dumps(random.getstate(), protocol=4)).hexdigest(),
        numpy=hashlib.sha256(pickle.dumps(np.random.get_state(), protocol=4)).hexdigest(),
        torch_cpu=hashlib.sha256(_tensor_bytes(torch.get_rng_state())).hexdigest(),
        torch_cuda=[hashlib.sha256(_tensor_bytes(state)).hexdigest()
                    for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [])
    flags = dict(module_training={name: module.training for name, module in model.named_modules()},
        requires_grad={name: parameter.requires_grad for name, parameter in model.named_parameters()})
    return dict(rng=rng, rng_sha256=hashlib.sha256(_canonical_bytes(rng)).hexdigest(),
        flags=flags, flags_sha256=hashlib.sha256(_canonical_bytes(flags)).hexdigest())


def initial_evaluation(model, tokenizer, suite, output, record, evaluate_fn=None):
    from .diagnostics import TRAIN_DEV_SPLITS, evaluate_split, evaluation_state
    evaluate_fn = evaluate_split if evaluate_fn is None else evaluate_fn
    record.update(before=runtime_snapshot(model), trainables_before=tensor_snapshot(model), metrics={})
    try:
        # The outer existing wrapper also restores state if a split fails before
        # its own wrapper is entered. Each original evaluate_split uses it too.
        with evaluation_state(model):
            for split in TRAIN_DEV_SPLITS:
                record['metrics'][split] = evaluate_fn(model, tokenizer, suite[split], 'sft',
                    Path(output) / 'initial-evaluation' / split, candidates=False)
    finally:
        record.update(after=runtime_snapshot(model), trainables_after=tensor_snapshot(model))


def _input_files(config):
    from .diagnostics import SOURCES, _base_files
    from .diagnostics_data import SPLITS
    files = {f'source/{name}': Path(__file__).with_name(name) for name in (*SOURCES, 'repro_probe.py')}
    files.update(dataset=Path(config.dataset), adapter=Path(config.adapter_path))
    base = Path(config.base_path)
    files.update({f'base/{path.relative_to(base) if base.is_dir() else path.name}': path
                  for path in _base_files(base)})
    if config.initial_eval_suite:
        suite = Path(config.initial_eval_suite)
        files.update({f'suite/{name}': suite / name for name in
                      ['manifest.json'] + [f'{split}.jsonl' for split in SPLITS]})
    return files


def run_probe(config, model=None, tokenizer=None):
    """Run three short original SFT updates; injection is for local CPU fixtures."""
    # Refuse collisions before touching process execution flags.
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=False)
    environment = configure_environment(config.profile)
    import numpy as np
    import torch
    from . import benchmark
    from .data import read_jsonl, validate_rows
    from .diagnostics_data import load_suite

    status = dict(status='running', phase='setup', completed_steps=0)
    metadata = dict(protocol_version=2, arguments=asdict(config), environment=environment,
        optimizer_settings=OPTIMIZER_SETTINGS, method='sft', microbatch=1,
        accumulation=1, max_grad_norm=1., versions=benchmark._versions(),
        seed_policy='Python, Torch CPU/CUDA before load; NumPy additionally seeded for trace identity (unused by original training)',
        injected_model=model is not None, hash_format='sorted named tensors; length-framed JSON header and CPU raw bytes; SHA256')
    trace = dict(initial={}, steps=[])
    write = benchmark._json
    write(output / 'config.json', metadata)
    write(output / 'status.json', status)
    write(output / 'trace.json', trace)
    try:
        if config.steps < 1:
            raise ValueError('Positive steps required')
        # Prior CUDA initialization only invalidates a CUDA-target probe. CPU
        # fixtures may share a process with callers that previously used CUDA.
        target_device = torch.device(model.device if model is not None else config.device)
        if (config.profile != 'default' and target_device.type == 'cuda'
                and environment['torch_preimported'] and torch.cuda.is_initialized()):
            raise RuntimeError('Deterministic CUDA probe requires a fresh CLI process before importing Torch')
        metadata['prior_runtime'] = runtime_settings()
        configure_torch(config.profile)
        metadata['runtime'] = runtime_settings()
        files = _input_files(config)
        metadata['input_hashes'] = {name: benchmark.sha256_file(path) for name, path in files.items()}
        input_stats = {name: (path.stat().st_size, path.stat().st_mtime_ns) for name, path in files.items()}
        write(output / 'config.json', metadata)
        if metadata['input_hashes']['dataset'] != config.expected_dataset_hash:
            raise ValueError('Original training dataset hash mismatch')
        rows = read_jsonl(config.dataset)
        validate_rows(rows)
        if not rows or any(row['kind'] != 'swap' or row['split'] != 'train' for row in rows):
            raise ValueError('Original known-condition training swaps required')
        suite = load_suite(config.initial_eval_suite) if config.initial_eval_suite else None
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        if model is None:
            model, tokenizer = benchmark.load_icae(config.base_path, config.adapter_path,
                device=config.device, dtype=torch.bfloat16, memory_size=128)
        if tokenizer is None:
            raise ValueError('Tokenizer required')
        items = [benchmark.prepare_row(row, tokenizer, 512) for row in rows]
        model.retrieval_proj.requires_grad_(False)
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        if model.base.is_gradient_checkpointing:
            raise ValueError('Shared adapter checkpointing is forbidden')
        trainables = [parameter for parameter in model.parameters() if parameter.requires_grad]
        order = list(range(len(items)))
        random.Random(config.seed).shuffle(order)
        metadata.update(cyclic_row_ids=[items[index]['id'] for index in order],
            prepared_items_sha256=hashlib.sha256(_canonical_bytes(items)).hexdigest(),
            prepared_rows=len(items), gpu=benchmark._gpu_metadata(model.device),
            base_dtype=str(model.base.get_input_embeddings().weight.dtype),
            memory_slots=model.memory_size,
            trainable_parameters=sum(parameter.numel() for parameter in trainables),
            attention_config=model.base.config.to_dict(),
            attention_implementation=getattr(model.base.config, '_attn_implementation', None),
            attention_modules={name: type(module).__name__ for name, module in model.named_modules()
                               if name.endswith('self_attn')},
            dropout_probabilities={name: module.p for name, module in model.named_modules()
                                   if isinstance(module, torch.nn.Dropout)})
        write(output / 'config.json', metadata)
        model.train()
        optimizer = torch.optim.AdamW(trainables, **OPTIMIZER_SETTINGS)
        with _attention_context(config.profile):
            metadata['effective_runtime'] = runtime_settings()
            write(output / 'config.json', metadata)
            trace['initial'] = dict(trainables=tensor_snapshot(model), state=runtime_snapshot(model))
            write(output / 'trace.json', trace)
            if suite is not None:
                status.update(phase='initial_evaluation')
                write(output / 'status.json', status)
                trace['initial_evaluation'] = {}
                try:
                    initial_evaluation(model, tokenizer, suite, output, trace['initial_evaluation'])
                finally:
                    write(output / 'trace.json', trace)
                evaluation = trace['initial_evaluation']
                if evaluation['before'] != evaluation['after'] or evaluation['trainables_before'] != evaluation['trainables_after']:
                    raise RuntimeError('Initial evaluation changed training state or weights')
            for step in range(config.steps):
                status.update(phase='training', active_step=step + 1)
                write(output / 'status.json', status)
                optimizer.zero_grad(set_to_none=True)
                item = items[order[step % len(order)]]
                record = dict(optimizer_step=step + 1, row_id=item['id'], before_forward=runtime_snapshot(model))
                trace['steps'].append(record)
                write(output / 'trace.json', trace)
                losses = benchmark.backward_sample(model, item, method='sft', accumulation=1)
                record['finite_losses'] = all(math.isfinite(value) for value in losses.values())
                record['losses'] = {name: _number(value) for name, value in losses.items()}
                benchmark._synchronize(model.device)
                record['raw_gradients'] = tensor_snapshot(model, gradients=True)
                write(output / 'trace.json', trace)
                norm = torch.nn.utils.clip_grad_norm_(trainables, 1., error_if_nonfinite=True)
                record['clip_returned_norm'] = _number(float(norm.detach()))
                record['post_clip_gradients'] = tensor_snapshot(model, gradients=True)
                write(output / 'trace.json', trace)
                if not record['raw_gradients']['finite'] or not math.isfinite(float(norm)) or not record['finite_losses']:
                    raise FloatingPointError('Non-finite loss or gradients')
                optimizer.step()
                benchmark._synchronize(model.device)
                record['post_step'] = tensor_snapshot(model)
                record['after_step_state'] = runtime_snapshot(model)
                write(output / 'trace.json', trace)
                status['completed_steps'] = step + 1
                write(output / 'status.json', status)
        for name, path in files.items():
            if (path.stat().st_size, path.stat().st_mtime_ns) != input_stats[name]:
                raise ValueError('Input changed during probe: ' + name)
            if path.stat().st_size < 8 * 1024 * 1024 and benchmark.sha256_file(path) != metadata['input_hashes'][name]:
                raise ValueError('Input hash changed during probe: ' + name)
        summary = dict(status='success', protocol_version=2, profile=config.profile,
            optimizer_steps=config.steps, row_ids=[record['row_id'] for record in trace['steps']],
            initial_trainables_sha256=trace['initial']['trainables']['sha256'],
            final_trainables_sha256=trace['steps'][-1]['post_step']['sha256'],
            trace_sha256=benchmark.sha256_file(output / 'trace.json'),
            interpretation='Localization evidence only; repeatability and policy choice require separate fresh-process comparisons.')
        write(output / 'summary.json', summary)
        status.update(status='success', phase='complete')
        write(output / 'status.json', status)
        return summary
    except BaseException as exc:
        failure = traceback.format_exc()
        status.update(status='failed', error_type=type(exc).__name__, error=str(exc), traceback=failure)
        write(output / 'trace.json', trace)
        write(output / 'status.json', status)
        write(output / 'summary.json', dict(status='failed', profile=config.profile,
            optimizer_steps=status['completed_steps'], error_type=type(exc).__name__, error=str(exc)))
        (output / 'failure.txt').write_text(failure, encoding='utf-8')
        raise


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base-path', 'adapter-path', 'dataset', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--profile', choices=PROFILES, default='default')
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--seed', type=int, default=73)
    parser.add_argument('--initial-eval-suite')
    parser.add_argument('--device', default='cuda')
    return parser


def main():
    args = argument_parser().parse_args()
    run_probe(ProbeConfig(**vars(args)))


if __name__ == '__main__':
    main()
