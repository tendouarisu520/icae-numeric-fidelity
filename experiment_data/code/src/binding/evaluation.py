"""Calibrate, freeze, evaluate, and summarize the fixed ICAE pilot.

No training and no downloads. Run separate evaluate processes for the five arms.
Primary predictions are unrestricted greedy generation; public-vocabulary
likelihood rankings are strictly diagnostic and never replace those predictions.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

import torch

from .benchmark import PREFIX, SUFFIX, _json, _versions, sha256_file
from .data import read_jsonl, validate_rows
from .eval_metrics import LABELS, diagnose_output, output_summary, scenario_bootstrap
from .model import load_icae
from .metrics import normalize

VARIANTS = ('legacy', 'delimited', 'structured')
ARMS = ('full_context', 'original_icae', 'sft', 'hard_negative', 'binding')
PROMPT_VERSION = 1
RUNTIME_SOURCES = ('evaluation.py', 'eval_metrics.py', 'model.py', 'metrics.py',
                   'data.py', 'objectives.py', 'benchmark.py')
TASK_INSTRUCTION = ('Find the record matching both the exact team and the exact condition in the question. '
                    'Use only that matching record. If no matching record exists, answer UNKNOWN. '
                    'Return exactly one label, without explanation, from: ' + ', '.join(LABELS) + '.\n')


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def prepare_input(row, tokenizer, variant, max_context_tokens=512):
    """Read only context/question. Compression IDs exactly match benchmark v1."""
    if variant not in VARIANTS or not 1 <= max_context_tokens <= 512:
        raise ValueError('Invalid prompt variant or context token limit')
    if tokenizer.bos_token_id is None or tokenizer.eos_token_id is None:
        raise ValueError('Tokenizer requires BOS and EOS')
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    context = [tokenizer.bos_token_id] + encode(row['context'])
    if len(context) > max_context_tokens:
        raise ValueError(f"context {row['id']} has {len(context)} tokens; limit {max_context_tokens}; truncation forbidden")
    if variant == 'legacy':
        left, right = list(PREFIX), encode(row['question']) + SUFFIX
    else:
        instruction = TASK_INSTRUCTION if variant == 'structured' else ''
        left = list(PREFIX) + encode(instruction + '\nRecords:\n')
        right = encode('\nEnd records.\nQuestion:\n' + row['question'] + '\n') + SUFFIX
    return {'context': context, 'left': left, 'right': right}


def candidate_ids(tokenizer):
    return {label: tokenizer.encode(' ' + label, add_special_tokens=False) + [tokenizer.eos_token_id]
            for label in LABELS}


def rank_candidates(scores):
    if tuple(scores) != LABELS or not all(math.isfinite(v) for v in scores.values()):
        raise ValueError('Require finite scores for the fixed public vocabulary in declared order')
    maximum = max(scores.values())
    ties = [label for label in LABELS if scores[label] == maximum]
    return {'scores': scores, 'prediction': ties[0], 'ties': ties,
            'tie_policy': 'first_in_public_vocabulary', 'tie_tolerance': 0.0,
            'score_definition': 'mean_answer_token_log_probability_including_eos',
            'diagnostic_only': True}


def primary_error(prediction, row):
    if normalize(prediction) == normalize(row['answer']):
        return 'correct'
    if row['kind'] == 'swap' and normalize(prediction) == normalize(row['foil']):
        return 'foil_misbinding'
    return 'other_public_label' if diagnose_output(prediction)['exact_label'] else 'malformed'


def select_variant(summaries):
    references = {s['variant']: s for s in summaries if s['arm'] == 'full_context'}
    if set(references) != set(VARIANTS):
        raise ValueError('Require all three full-context calibration variants')
    def key(variant):
        summary = references[variant]
        metrics = summary['metrics']
        malformed = summary.get('malformed_rate', metrics.get('malformed_rate'))
        return (metrics['pair_accuracy'], metrics['accuracy'], -malformed, -VARIANTS.index(variant))
    return max(VARIANTS, key=key)


@torch.no_grad()
def greedy_generate(model, prefix, eos_id, max_new_tokens=32):
    """Manual greedy generation preserves EOS and exact max-length metadata."""
    if max_new_tokens < 1:
        raise ValueError('max_new_tokens must be positive')
    from transformers import DynamicCache
    generated, cache, inputs = [], DynamicCache(), prefix
    with model.icae.disable_adapter():
        for _ in range(max_new_tokens):
            out = model.base(inputs_embeds=inputs, past_key_values=cache, use_cache=True)
            next_id = int(out.logits[0, -1].argmax())
            generated.append(next_id)
            if next_id == eos_id:
                return {'token_ids': generated, 'stop_reason': 'eos', 'eos_emitted': True,
                        'generated_token_count': len(generated)}
            cache = out.past_key_values
            inputs = model.embeddings([next_id]).unsqueeze(0)
    return {'token_ids': generated, 'stop_reason': 'max_new_tokens', 'eos_emitted': False,
            'generated_token_count': len(generated)}


@torch.no_grad()
def score_candidates(model, prefix, candidates):
    """One candidate at a time; same decoder and prefix as primary generation."""
    scores = {}
    with model.icae.disable_adapter():
        for label, answer in candidates.items():
            tokens = torch.tensor(answer, device=model.device, dtype=torch.long)
            inputs = torch.cat((prefix, model.embeddings(answer).unsqueeze(0)), dim=1)
            output = model.base(inputs_embeds=inputs, use_cache=False)
            # Logit at prefix[-1] predicts answer[0]. Only answer/EOS positions count.
            logits = output.logits[0, prefix.shape[1] - 1:prefix.shape[1] - 1 + len(answer)].float()
            logp = torch.log_softmax(logits, dim=-1).gather(-1, tokens[:, None]).squeeze(-1)
            scores[label] = float(logp.mean())
            del output, inputs, logits, logp
    return rank_candidates(scores)


def load_eval_model(base_path, adapter_path, checkpoint=None, device='cuda'):
    """Fresh original state per process; retain requires_grad for adapter loader."""
    model, tokenizer = load_icae(base_path, adapter_path, device=device, dtype=torch.bfloat16)
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    if checkpoint:
        from safetensors.torch import load_file
        model.load_adapter_state(load_file(str(checkpoint), device='cpu'))
    # requires_grad is intentionally retained: all execution below uses no_grad.
    # The existing loader validates keys based on requires_grad.
    if any(p.requires_grad and p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError('Trainable adapter parameters must remain FP32')
    model.eval()
    return model, tokenizer


def _validate_dataset(rows, split):
    validate_rows(rows)
    if not rows or any(r['split'] != split for r in rows):
        raise ValueError(f'Require nonempty {split}-only frozen data')
    pairs = {r['pair_id'] for r in rows if r['pair_id'] is not None}
    if split == 'dev' and len(pairs) < 20:
        raise ValueError('Calibration requires at least 20 complete dev pairs')
    for row in rows:
        if row['kind'] not in ('swap', 'unknown') or row['answer'] not in LABELS or row['foil'] not in LABELS:
            raise ValueError('Invalid row kind or public answer label')
        if (row['kind'] == 'unknown') != (row['pair_id'] is None):
            raise ValueError('Only swap examples may have a pair_id')
    groups = {}
    for row in rows:
        if row['pair_id'] is not None:
            old = groups.setdefault(row['pair_id'], row['scenario_id'])
            if old != row['scenario_id']:
                raise ValueError('A pair must belong to one scenario')


def _write_config(output, config):
    config = dict(config)
    config['versions'] = _versions()
    config['source_hashes'] = {name: sha256_file(Path(__file__).with_name(name)) for name in RUNTIME_SOURCES}
    config['config_sha256'] = canonical_hash(config)
    _json(output / 'config.json', config)
    return config


def _read_config(path):
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    if canonical_hash({k:v for k,v in config.items() if k != 'config_sha256'}) != config.get('config_sha256'):
        raise ValueError('Config integrity hash mismatch: ' + str(path))
    return config


def _status(output, **values):
    value = dict(timestamp=now(), **values)
    _json(output / 'status.json', value)
    print(json.dumps(value, allow_nan=False), flush=True)


@torch.no_grad()
def evaluate_rows(model, tokenizer, rows, variant, arm, output, candidates=True,
                  max_new_tokens=32, max_context_tokens=512, progress=None):
    """Stream raw predictions as each example completes; never resume/overwrite."""
    if arm not in ARMS:
        raise ValueError('Unknown evaluation arm')
    items = [prepare_input(row, tokenizer, variant, max_context_tokens) for row in rows]
    candidate_tokens = candidate_ids(tokenizer) if candidates else None
    saved = []
    model.eval()
    with (Path(output) / 'predictions.jsonl').open('x', encoding='utf-8') as handle:
        for index, (row, item) in enumerate(zip(rows, items)):
            started = time.perf_counter()
            if arm == 'full_context':
                prefix = model.embeddings(item['left'] + item['context'][1:] + item['right']).unsqueeze(0)
            else:
                memory = model.compress_ids(torch.tensor([item['context']], device=model.device))
                prefix = model.prefix_ids(memory, item['left'], item['right'])
                del memory
            generation = greedy_generate(model, prefix, tokenizer.eos_token_id, max_new_tokens)
            decoded = tokenizer.decode(generation['token_ids'], skip_special_tokens=True)
            record = {'id': row['id'], 'arm': arm, 'variant': variant, 'prediction': decoded,
                      'raw_output': tokenizer.decode(generation['token_ids'], skip_special_tokens=False),
                      'generation': generation, 'context_token_count': len(item['context']),
                      'prefix_token_count': prefix.shape[1], 'timestamp': now(),
                      'format': diagnose_output(decoded, generation['stop_reason'] == 'max_new_tokens')}
            record['primary_error'] = primary_error(decoded, row)
            if candidates:
                record['candidate_diagnostic'] = score_candidates(model, prefix, candidate_tokens)
            record['elapsed_seconds'] = time.perf_counter() - started
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
            handle.flush()
            saved.append(record)
            if progress:
                progress(index + 1, row['id'])
            del prefix
    return output_summary(rows, saved)


def _new_output(path):
    output = Path(path)
    output.mkdir(parents=True, exist_ok=False)
    _status(output, status='running', phase='validation', completed=0)
    return output


def _fail(output, exc):
    previous = json.loads((output / 'status.json').read_text(encoding='utf-8'))
    _status(output, status='failed', phase=previous.get('phase'), completed=previous.get('completed', 0),
            error_type=type(exc).__name__, error=str(exc),
            oom=isinstance(exc, torch.cuda.OutOfMemoryError), traceback=traceback.format_exc())


def run_calibration(args, model=None, tokenizer=None):
    output = _new_output(args.output)
    try:
        rows = read_jsonl(args.dataset)
        config = _write_config(output, dict(stage='calibrate', dataset_sha256=sha256_file(args.dataset),
            original_adapter_sha256=sha256_file(args.adapter_path), base_path=str(args.base_path),
            dataset=str(args.dataset), adapter_path=str(args.adapter_path), device=args.device,
            variants=list(VARIANTS), arms=list(ARMS[:2]), max_new_tokens=32, max_context_tokens=512,
            candidate_diagnostics=bool(args.candidate_diagnostics), prompt_version=PROMPT_VERSION))
        _validate_dataset(rows, 'dev')
        _status(output, status='running', phase='load_original', completed=0)
        if model is None:
            model, tokenizer = load_eval_model(args.base_path, args.adapter_path, device=args.device)
        summaries = []
        for arm in ARMS[:2]:
            for variant in VARIANTS:
                destination = output / f'{arm}__{variant}'
                destination.mkdir()
                finished = len(summaries) * len(rows)
                progress = lambda n, rid: _status(output, status='running', phase='generation',
                    arm=arm, variant=variant, completed=finished+n, total=6*len(rows), last_id=rid)
                metrics = evaluate_rows(model, tokenizer, rows, variant, arm, destination,
                                        candidates=args.candidate_diagnostics, progress=progress)
                summary = dict(arm=arm, variant=variant, metrics=metrics,
                               malformed_rate=metrics['malformed_rate'])
                _json(destination / 'summary.json', summary)
                summaries.append(summary)
        result = dict(status='success', selected_variant=select_variant(summaries), summaries=summaries,
                      config_sha256=config['config_sha256'], dataset_sha256=config['dataset_sha256'],
                      selection_rule='full_context_pair_accuracy_then_accuracy_then_lower_malformed_then_simpler')
        _json(output / 'summary.json', result)
        _status(output, status='success', phase='complete', completed=6*len(rows), total=6*len(rows))
        return result
    except BaseException as exc:
        _fail(output, exc)
        raise


def _validate_test(rows):
    _validate_dataset(rows, 'test')
    pairs = {r['pair_id'] for r in rows if r['pair_id'] is not None}
    unknown = sum(r['kind'] == 'unknown' for r in rows)
    if len(pairs) != 200 or unknown != 40 or len(rows) != 440:
        raise ValueError('Frozen test must contain exactly 200 complete pairs and 40 unknown controls')
    families = {r['family'] for r in rows}
    if families != {'context', 'time', 'subject', 'combination'}:
        raise ValueError('Frozen test must contain all four condition families')
    counts = [sum(r['family'] == f for r in rows) for f in sorted(families)]
    if len(set(counts)) != 1:
        raise ValueError('Frozen test must balance condition families')


def freeze_protocol(args):
    output = _new_output(args.output)
    try:
        source = Path(args.calibration_dir)
        status = json.loads((source / 'status.json').read_text(encoding='utf-8'))
        summary = json.loads((source / 'summary.json').read_text(encoding='utf-8'))
        config = _read_config(source / 'config.json')
        if status['status'] != 'success' or summary['status'] != 'success':
            raise ValueError('Calibration must complete successfully before freezing')
        if summary['config_sha256'] != config['config_sha256'] or summary['dataset_sha256'] != config['dataset_sha256']:
            raise ValueError('Calibration summary/config mismatch')
        if set(config['source_hashes']) != set(RUNTIME_SOURCES):
            raise ValueError('Calibration must bind all evaluator runtime sources')
        for name, expected in config['source_hashes'].items():
            if sha256_file(Path(__file__).with_name(name)) != expected:
                raise ValueError('Evaluator source changed since calibration: ' + name)
        rows = read_jsonl(args.dataset)
        _validate_test(rows)
        variant = select_variant(summary['summaries'])
        if variant != summary['selected_variant']:
            raise ValueError('Calibration selection mismatch')
        protocol = dict(schema_version=1, prompt_version=PROMPT_VERSION, selected_variant=variant,
            variant_order=list(VARIANTS), task_instruction=TASK_INSTRUCTION,
            candidate_labels=list(LABELS), candidate_score='mean_answer_token_log_probability_including_eos',
            primary='strict_free_greedy_generation', max_new_tokens=32, max_context_tokens=512,
            context_truncation=False, base_dtype='bfloat16', trainable_dtype='float32',
            dev_sha256=config['dataset_sha256'], test_sha256=sha256_file(args.dataset),
            original_adapter_sha256=config['original_adapter_sha256'], base_path=config['base_path'],
            calibration_summary_sha256=sha256_file(source / 'summary.json'),
            calibration_config_sha256=sha256_file(source / 'config.json'),
            evaluator_source_hashes=config['source_hashes'], created_at=now())
        protocol['protocol_sha256'] = canonical_hash(protocol)
        _json(output / 'protocol.json', protocol)
        _status(output, status='success', phase='frozen', completed=0,
                protocol_sha256=protocol['protocol_sha256'], selected_variant=variant)
        return protocol
    except BaseException as exc:
        _fail(output, exc)
        raise


def read_protocol(path, dataset):
    protocol = json.loads(Path(path).read_text(encoding='utf-8'))
    contents = {k:v for k,v in protocol.items() if k != 'protocol_sha256'}
    if canonical_hash(contents) != protocol['protocol_sha256']:
        raise ValueError('Protocol integrity hash mismatch')
    if (protocol['prompt_version'] != PROMPT_VERSION or protocol['task_instruction'] != TASK_INSTRUCTION or
            protocol['candidate_labels'] != list(LABELS) or protocol['max_new_tokens'] != 32 or
            protocol['max_context_tokens'] != 512):
        raise ValueError('Frozen protocol incompatible with evaluator')
    for name, expected in protocol['evaluator_source_hashes'].items():
        if sha256_file(Path(__file__).with_name(name)) != expected:
            raise ValueError('Evaluator source changed since calibration: ' + name)
    if sha256_file(dataset) != protocol['test_sha256']:
        raise ValueError('Frozen test dataset hash mismatch')
    return protocol


def run_evaluation(args, model=None, tokenizer=None):
    output = _new_output(args.output)
    try:
        protocol = read_protocol(args.protocol, args.dataset)
        rows = read_jsonl(args.dataset)
        _validate_test(rows)
        if str(args.base_path) != protocol['base_path']:
            raise ValueError('Base path changed since calibration')
        if (args.arm in ARMS[2:]) != bool(args.checkpoint):
            raise ValueError('Trained arms require --checkpoint; reference arms prohibit it')
        original_hash = sha256_file(args.adapter_path)
        if original_hash != protocol['original_adapter_sha256']:
            raise ValueError('Original adapter changed since calibration')
        config = _write_config(output, dict(stage='evaluate', arm=args.arm, variant=protocol['selected_variant'],
            protocol_sha256=protocol['protocol_sha256'], dataset_sha256=sha256_file(args.dataset),
            original_adapter_sha256=original_hash, checkpoint_sha256=sha256_file(args.checkpoint) if args.checkpoint else None,
            base_path=str(args.base_path), adapter_path=str(args.adapter_path), checkpoint=args.checkpoint,
            dataset=str(args.dataset), device=args.device, max_new_tokens=32, max_context_tokens=512,
            candidate_diagnostics=True, base_dtype='bfloat16', trainable_dtype='float32'))
        _status(output, status='running', phase='load_model', completed=0)
        if model is None:
            model, tokenizer = load_eval_model(args.base_path, args.adapter_path, args.checkpoint, args.device)
        progress = lambda n, rid: _status(output, status='running', phase='generation',
            arm=args.arm, completed=n, total=len(rows), last_id=rid)
        metrics = evaluate_rows(model, tokenizer, rows, protocol['selected_variant'], args.arm, output,
                                candidates=True, progress=progress)
        result = dict(status='success', arm=args.arm, metrics=metrics,
                      protocol_sha256=protocol['protocol_sha256'], config_sha256=config['config_sha256'])
        _json(output / 'summary.json', result)
        _status(output, status='success', phase='complete', completed=len(rows), total=len(rows))
        return result
    except BaseException as exc:
        _fail(output, exc)
        raise


def run_summary(args):
    output = _new_output(args.output)
    try:
        protocol = read_protocol(args.protocol, args.dataset)
        rows = read_jsonl(args.dataset)
        _validate_test(rows)
        paths = dict(value.split('=', 1) for value in args.runs)
        if set(paths) != set(ARMS) or len(args.runs) != len(ARMS):
            raise ValueError('Require exactly one completed run for each of the five arms')
        predictions, metrics, hashes = {}, {}, {}
        for arm in ARMS:
            path = Path(paths[arm])
            config = _read_config(path / 'config.json')
            status = json.loads((path / 'status.json').read_text(encoding='utf-8'))
            if status['status'] != 'success' or config['protocol_sha256'] != protocol['protocol_sha256'] or config['arm'] != arm:
                raise ValueError('Incomplete run or protocol/arm mismatch: ' + arm)
            predictions[arm] = read_jsonl(path / 'predictions.jsonl')
            if any(p.get('arm') != arm or p.get('variant') != protocol['selected_variant'] for p in predictions[arm]):
                raise ValueError('Foreign prediction arm/variant: ' + arm)
            metrics[arm] = output_summary(rows, predictions[arm])
            if metrics[arm]['missing']:
                raise ValueError('Final summary requires all predictions: ' + arm)
            hashes[arm] = {'predictions_sha256': sha256_file(path / 'predictions.jsonl'),
                           'checkpoint_sha256': config['checkpoint_sha256'],
                           'config_sha256': sha256_file(path / 'config.json')}
        deltas = {arm: scenario_bootstrap(rows, predictions['binding'], predictions[arm],
                                         seed=73, repetitions=args.bootstrap_repetitions) for arm in ARMS[:-1]}
        result = dict(status='success', protocol_sha256=protocol['protocol_sha256'],
            metrics=metrics, binding_minus_baseline=deltas, inputs=hashes,
            primary_metric='pair_accuracy', secondary_intervals='exploratory; no multiplicity correction',
            interpretation='Single-seed synthetic 50-step pilot; calibration is dev-only; candidate likelihood is diagnostic only.')
        _json(output / 'summary.json', result)
        _status(output, status='success', phase='complete', completed=5*len(rows))
        return result
    except BaseException as exc:
        _fail(output, exc)
        raise


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='stage', required=True)
    for stage in ('calibrate', 'evaluate'):
        sub = subs.add_parser(stage)
        for field in ('dataset', 'base-path', 'adapter-path', 'output'):
            sub.add_argument('--' + field, required=True)
        sub.add_argument('--device', default='cuda')
        if stage == 'calibrate':
            sub.add_argument('--candidate-diagnostics', action='store_true')
        else:
            sub.add_argument('--protocol', required=True)
            sub.add_argument('--arm', choices=ARMS, required=True)
            sub.add_argument('--checkpoint')
    freeze = subs.add_parser('freeze')
    freeze.add_argument('--calibration-dir', required=True)
    freeze.add_argument('--dataset', required=True)
    freeze.add_argument('--output', required=True)
    summary = subs.add_parser('summarize')
    summary.add_argument('--protocol', required=True)
    summary.add_argument('--dataset', required=True)
    summary.add_argument('--runs', nargs='+', required=True, help='arm=completed_run_directory for each of the five arms')
    summary.add_argument('--bootstrap-repetitions', type=int, default=2000)
    summary.add_argument('--output', required=True)
    return parser


def main():
    args = argument_parser().parse_args()
    functions = dict(calibrate=run_calibration, freeze=freeze_protocol,
                     evaluate=run_evaluation, summarize=run_summary)
    functions[args.stage](args)


if __name__ == '__main__':
    main()
