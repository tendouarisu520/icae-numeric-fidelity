"""Fresh developmental diagnostic cases; labels and metadata are never prompts."""
import collections
import hashlib
import json
from pathlib import Path
import random

from .data import CONDITIONS, VALUES, question, read_jsonl, statement, validate_rows, write_jsonl

SPLITS = ('dev_d0', 'dev_d2', 'dev_d6', 'dev_d12', 'dev_singlefact', 'overfit', 'dev_unknown',
          'dev_trainstyle_d0', 'dev_unrelated_entities_d12')
INTERPRETATION = ('Shared development cases for capability and trajectory diagnosis only. '
                  'Repeated use is not confirmatory evidence; a changed method needs a fresh holdout.')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _scenario(rng, seed, group, index, distractors, unknown=False):
    family = tuple(CONDITIONS)[index % 4]
    conditions = CONDITIONS[family]
    entity = f'Team Diag{seed}{group}{index:03d}'
    template = index % 2 if group == 'T' else 2
    values = (VALUES[index], VALUES[(index + 1) % 8]) if group == 'T' else rng.sample(VALUES, 2)
    target = 0 if group == 'T' else rng.randrange(2)
    qcond = conditions[2] if unknown else conditions[target]
    extras = []
    for j in range(distractors):
        other = f'Team Diag{seed}{group}{index:03d}D{j}' if j % 2 else entity
        cond = conditions[rng.randrange(2)] if j % 2 else f'special assignment {j + 10}'
        extras.append(statement(other, cond, rng.choice(VALUES), template))
    order = list(range(distractors + 2))
    rng.shuffle(order)
    scenario = f'diag{seed}-{group.lower()}-{index:03d}'
    rows = []
    for variant in range(1 if unknown else 2):
        assigned = values if variant == 0 else values[::-1]
        evidence = [statement(entity, conditions[k], assigned[k], template) for k in range(2)]
        parts = evidence + extras
        rows.append(dict(id=f'{scenario}-{variant}', scenario_id=scenario,
            pair_id=None if unknown else scenario, variant=variant,
            split='overfit' if group == 'T' else 'diagnostic_dev', family=family,
            kind='unknown' if unknown else 'swap', entity=entity, template_id=template,
            condition=qcond, context='\n'.join(parts[j] for j in order),
            question=question(entity, qcond), answer='UNKNOWN' if unknown else assigned[target],
            foil=assigned[target] if unknown else assigned[1 - target],
            support=None if unknown else evidence[target], hard_negative=evidence[1 - target]))
    return rows


def make_diagnostic_suite(seed=173):
    """Construct nested distractor probes, balanced overfit pairs, and UNKNOWNs."""
    if not isinstance(seed, int) or seed < 0:
        raise ValueError('Diagnostic seed must be a nonnegative integer')
    rng = random.Random(seed)
    master = [row for i in range(16) for row in _scenario(rng, seed, 'D', i, 12)]
    suite = {}
    for n in (0, 2, 6, 12):
        rows = []
        for row in master:
            lines = row['context'].splitlines()
            evidence = {row['support'], row['hard_negative']}
            extras = [line for line in lines if line not in evidence]
            wanted = evidence | set(extras[:n])
            rows.append(dict(row, context='\n'.join(line for line in lines if line in wanted)))
        suite[f'dev_d{n}'] = rows
    suite['dev_singlefact'] = [dict(row, id=f'diag{seed}-single-{i:03d}',
        scenario_id=f'diag{seed}-single-{i:03d}', pair_id=None, kind='singlefact_probe',
        context=row['support']) for i, row in enumerate(master)]
    suite['overfit'] = [row for i in range(8) for row in _scenario(rng, seed, 'T', i, 0)]
    suite['dev_unknown'] = [row for i in range(8) for row in _scenario(rng, seed, 'U', i, 12, True)]
    suite['dev_trainstyle_d0'] = []
    for index, row in enumerate(suite['dev_d0']):
        template = (index // 2) % 2
        other = next(c for c in CONDITIONS[row['family']][:2] if c != row['condition'])
        support = statement(row['entity'], row['condition'], row['answer'], template)
        negative = statement(row['entity'], other, row['foil'], template)
        lines = [support if line == row['support'] else negative for line in row['context'].splitlines()]
        suite['dev_trainstyle_d0'].append(dict(row, context='\n'.join(lines),
            template_id=template, support=support, hard_negative=negative))
    suite['dev_unrelated_entities_d12'] = []
    for index, row in enumerate(master):
        context = row['context']
        # Longest first prevents D1 from partially replacing D11.
        for j in (11, 9, 7, 5, 3, 1):
            foreign = f"{row['entity']}D{j}"
            replacement = f'Team Diag{seed}F{index // 2:03d}X{j}'
            context = context.replace(foreign, replacement)
        suite['dev_unrelated_entities_d12'].append(dict(row, context=context))
    for rows in suite.values():
        validate_rows(rows)
    return suite


def freeze_suite(output, seed=173):
    """Write a new immutable suite, binding counts, generation, and source hashes."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    suite = make_diagnostic_suite(seed)
    manifest = dict(schema_version=1, generator='binding.diagnostics_data', seed=seed,
        interpretation=INTERPRETATION,
        parameters=dict(master_pairs=16, nested_distractors=[0, 2, 6, 12],
                        overfit_pairs=8, unknown_controls=8, singlefact_queries=32,
                        overfit_target_condition=0, overfit_templates=[0, 1], dev_template=2),
        source_hashes={name: _sha(Path(__file__).with_name(name))
                       for name in ('diagnostics_data.py', 'data.py')}, splits={})
    for split, rows in suite.items():
        file = output / f'{split}.jsonl'
        write_jsonl(file, rows)
        manifest['splits'][split] = dict(rows=len(rows),
            pairs=len({r['pair_id'] for r in rows if r['pair_id'] is not None}),
            label_counts=dict(collections.Counter(r['answer'] for r in rows)),
            family_counts=dict(collections.Counter(r['family'] for r in rows)), sha256=_sha(file))
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding='utf-8')
    return manifest


def load_suite(path):
    """Reject tampering, duplicate IDs, unsupported splits, and incomplete pairs."""
    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1 or set(manifest.get('splits', {})) != set(SPLITS):
        raise ValueError('Unsupported diagnostic manifest')
    for name, digest in manifest.get('source_hashes', {}).items():
        if name not in ('diagnostics_data.py', 'data.py') or _sha(Path(__file__).with_name(name)) != digest:
            raise ValueError('Diagnostic generator source hash mismatch')
    if set(manifest.get('source_hashes', {})) != {'diagnostics_data.py', 'data.py'}:
        raise ValueError('Missing diagnostic generator source hashes')
    expected = make_diagnostic_suite(manifest['seed'])
    suite = {}
    for split in SPLITS:
        file = path / f'{split}.jsonl'
        if _sha(file) != manifest['splits'][split]['sha256']:
            raise ValueError('Diagnostic file hash mismatch: ' + split)
        rows = read_jsonl(file)
        validate_rows(rows)
        counts = manifest['splits'][split]
        if (rows != expected[split] or counts['rows'] != len(rows) or
                counts['pairs'] != len({r['pair_id'] for r in rows if r['pair_id'] is not None}) or
                counts['label_counts'] != dict(collections.Counter(r['answer'] for r in rows)) or
                counts['family_counts'] != dict(collections.Counter(r['family'] for r in rows))):
            raise ValueError('Diagnostic rows/counts disagree with generation parameters: ' + split)
        suite[split] = rows
    return suite
