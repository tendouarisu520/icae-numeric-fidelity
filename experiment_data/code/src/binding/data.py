"""Deterministic intervention pairs; metadata is NEVER a model input."""
import argparse
import collections
import hashlib
import json
import random
from pathlib import Path

VALUES = ('Python', 'Rust', 'Java', 'Ruby', 'Go', 'Swift', 'Kotlin', 'Julia')
CONDITIONS = {
    'context': ('office tasks', 'home tasks', 'school tasks'),
    'time': ('morning sessions', 'evening sessions', 'afternoon sessions'),
    'subject': ('requests from Alice', 'requests from Bob', 'requests from Carol'),
    'combination': ('office tasks on weekdays', 'home tasks on weekends', 'school tasks on holidays'),
}
TEMPLATES = {
    'train': (0, 1), 'dev': (2,), 'test': (3,), 'ood': (4,),
}


def statement(entity, condition, value, template):
    return (
        f'{entity} uses {value} for {condition}.',
        f'For {condition}, {entity} selects {value}.',
        f'The choice of {entity} for {condition} is {value}.',
        f'{entity} has specified {value} as the tool for {condition}.',
        f'When handling {condition}, the tool used by {entity} is {value}.',
    )[template]


def question(entity, condition):
    return f'Which tool does {entity} use for {condition}? Reply with only the tool name, or UNKNOWN if the records do not say.'


def make_split(split, pairs, seed=73, distractors=12, controls=0):
    if split not in TEMPLATES or min(pairs, distractors, controls) < 0:
        raise ValueError('Unknown split or negative size')
    salt = int(hashlib.sha256(split.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed + salt)
    prefix = {'train': 'T', 'dev': 'V', 'test': 'E', 'ood': 'O'}[split]
    rows = []
    for i in range(pairs + controls):
        family = tuple(CONDITIONS)[i % len(CONDITIONS)]
        conditions = CONDITIONS[family]
        if split == 'ood' and family == 'combination':
            conditions = ('office tasks on weekends', 'home tasks on weekdays', 'school tasks on holidays')
        template = rng.choice(TEMPLATES[split])
        entity = f'Team {prefix}{i:05d}'
        values = rng.sample(VALUES, 2)
        target = rng.randrange(2)
        is_control = i >= pairs
        qcond = conditions[2] if is_control else conditions[target]
        q = question(entity, qcond)
        context_parts = []
        # Half of distractors share the subject but have different conditions;
        # the rest share a target condition but belong to another subject.
        for j in range(distractors):
            if j % 2:
                e, c = f'Team {prefix}{i:05d}D{j}', conditions[rng.randrange(2)]
            else:
                e, c = entity, f'special assignment {j + 10}'
            context_parts.append(statement(e, c, rng.choice(VALUES), template))
        order = list(range(distractors + 2))
        rng.shuffle(order)
        scenario = f'{split}-{i:06d}'
        for variant in range(1 if is_control else 2):
            assigned = values if variant == 0 else values[::-1]
            evidence = [statement(entity, conditions[k], assigned[k], template) for k in range(2)]
            parts = evidence + context_parts
            context = '\n'.join(parts[j] for j in order)
            answer = 'UNKNOWN' if is_control else assigned[target]
            foil = assigned[target] if is_control else assigned[1 - target]
            rows.append({
                'id': f'{scenario}-{variant}', 'scenario_id': scenario,
                'pair_id': None if is_control else scenario,
                'variant': variant, 'split': split, 'family': family,
                'kind': 'unknown' if is_control else 'swap', 'entity': entity,
                'template_id': template, 'condition': qcond,
                'context': context, 'question': q, 'answer': answer, 'foil': foil,
                'support': None if is_control else evidence[target],
                'hard_negative': evidence[1 - target],
            })
    validate_rows(rows)
    return rows


def validate_rows(rows):
    ids = [r['id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate row IDs')
    groups = collections.defaultdict(list)
    for row in rows:
        if not row['context'] or not row['question']:
            raise ValueError('Empty model input')
        if row['pair_id'] is not None:
            groups[row['pair_id']].append(row)
    for pair, group in groups.items():
        if len(group) != 2:
            raise ValueError(f'Incomplete intervention pair: {pair}')
        a, b = group
        if (a['question'] != b['question'] or a['answer'] != b['foil'] or
                a['foil'] != b['answer'] or a['answer'] == a['foil'] or
                collections.Counter(a['context'].split()) != collections.Counter(b['context'].split())):
            raise ValueError(f'Invalid intervention: {pair}')


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='\n') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='data/pilot')
    parser.add_argument('--seed', type=int, default=73)
    parser.add_argument('--train-pairs', type=int, default=1000)
    parser.add_argument('--dev-pairs', type=int, default=100)
    parser.add_argument('--test-pairs', type=int, default=200)
    parser.add_argument('--distractors', type=int, default=12)
    args = parser.parse_args()
    destination = Path(args.output)
    if (destination / 'manifest.json').exists():
        raise FileExistsError('Frozen data already exists; use a new output directory')
    manifest = {'schema_version': 1, 'generator': 'binding.data', 'config': vars(args), 'splits': {}}
    for split, count in [('train', args.train_pairs), ('dev', args.dev_pairs), ('test', args.test_pairs), ('ood', args.test_pairs)]:
        rows = make_split(split, count, args.seed, args.distractors, controls=max(1, count // 5))
        path = destination / f'{split}.jsonl'
        write_jsonl(path, rows)
        manifest['splits'][split] = {'rows': len(rows), 'pairs': count, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
