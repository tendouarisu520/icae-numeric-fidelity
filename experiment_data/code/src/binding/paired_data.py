"""Protocol-v2 complete paired/UNKNOWN scenarios, with immutable data freezes.

Both pairs share one independent scenario cluster. Names are opaque equal-length
random tokens (v2); no target name is a prefix of a foreign name. Metadata is
never concatenated into model context or questions.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import random
import re
import string

from .data import CONDITIONS, VALUES, TEMPLATES, statement, question, validate_rows, read_jsonl, write_jsonl

LEVELS = (0, 2, 6, 12)
SPLITS = ('train', 'dev', 'test')
SCENARIOS = dict(train=96, dev=48, test=96)
SUBGROUP_BOUND = (6, 18)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    return {name: sha256_file(Path(__file__).with_name(name)) for name in ('data.py', 'paired_data.py')}


def _label_pairs(specs, rng, balanced):
    # Random blocks of four scenarios expose every label once; assignment order
    # is shuffled independently of condition/template and naming draws. Reject
    # only on the predeclared family/difficulty bound, never on model outcomes.
    for _ in range(10000):
        order = list(range(len(specs)))
        rng.shuffle(order)
        pairs = {}
        for start in range(0, len(order), 4):
            labels = list(VALUES)
            rng.shuffle(labels)
            for j, index in enumerate(order[start:start + 4]):
                pairs[index] = labels[2*j:2*j+2]
        if not balanced:
            return pairs
        counters = [collections.defaultdict(collections.Counter) for _ in range(2)]
        for index, spec in enumerate(specs):
            for axis, key in enumerate(spec[:2]):
                counters[axis][key].update(pairs[index] * 2)
        if all(SUBGROUP_BOUND[0] <= c[v] <= SUBGROUP_BOUND[1]
               for axis in counters for c in axis.values() for v in VALUES):
            return pairs
    raise RuntimeError('Could not satisfy predeclared label subgroup bound')


def make_suite(seed=73):
    rng = random.Random(seed)
    used_names = set()

    def name():
        while True:
            token = 'Team ' + ''.join(rng.choices(string.ascii_lowercase, k=12))
            if token not in used_names:
                used_names.add(token)
                return token

    suite = {}
    for split in SPLITS:
        specs = [(family, level, absent, template) for family in CONDITIONS for level in LEVELS
                 for absent in range(3) for template in TEMPLATES[split]]
        if split == 'test':
            specs = specs * 2
        # A separate stream avoids acceptance-loop effects on entity/context RNG.
        labels = _label_pairs(specs, random.Random(canonical_hash([seed, split, 'labels'])), split == 'train')
        rows = []
        for i, (family, level, absent, template) in enumerate(specs):
            conditions = CONDITIONS[family]
            present = [k for k in range(3) if k != absent]
            values = labels[i]
            entity = name()
            foreign = [name() for _ in range(level // 2)]
            entities = [entity] + foreign
            distractors = []
            for j in range(level):
                e, c = ((foreign[j // 2], rng.choice(conditions)) if j % 2 else
                        (entity, f'special assignment {j + 10}'))
                distractors.append(statement(e, c, rng.choice(VALUES), template))
            order = list(range(level + 2))
            rng.shuffle(order)
            scenario = f'paired-v2-{split}-{i:04d}'
            contexts, evidence_by_variant = [], []
            for variant in range(2):
                assigned = values if variant == 0 else values[::-1]
                evidence = [statement(entity, conditions[k], assigned[j], template) for j, k in enumerate(present)]
                parts = evidence + distractors
                contexts.append('\n'.join(parts[j] for j in order))
                evidence_by_variant.append(evidence)
            shared = dict(scenario_id=scenario, split=split, family=family, difficulty=level,
                entity=entity, entities=entities, template_id=template, absent_index=absent)
            for target, index in enumerate(present):
                for variant in range(2):
                    assigned = values if variant == 0 else values[::-1]
                    rows.append(dict(shared, id=f'{scenario}-q{index}-v{variant}',
                        pair_id=f'{scenario}-q{index}', kind='swap', queried_index=index,
                        variant=variant, condition=conditions[index], context=contexts[variant],
                        question=question(entity, conditions[index]), answer=assigned[target], foil=assigned[1-target],
                        support=evidence_by_variant[variant][target], hard_negative=evidence_by_variant[variant][1-target]))
            replicate = i // 48 if split == 'test' else 0
            variant = (tuple(CONDITIONS).index(family) + LEVELS.index(level) + absent + template + replicate) % 2
            rows.append(dict(shared, id=f'{scenario}-unknown', pair_id=None, kind='unknown',
                queried_index=absent, variant=variant, condition=conditions[absent], context=contexts[variant],
                question=question(entity, conditions[absent]), answer='UNKNOWN', foil=values[variant],
                support=None, hard_negative=None))
        suite[split] = rows
    audit_suite(suite)
    return suite


_PATTERNS = (
    r'(?P<entity>Team [a-z]{12}) uses (?P<value>\w+) for (?P<condition>.+)\.',
    r'For (?P<condition>.+), (?P<entity>Team [a-z]{12}) selects (?P<value>\w+)\.',
    r'The choice of (?P<entity>Team [a-z]{12}) for (?P<condition>.+) is (?P<value>\w+)\.',
    r'(?P<entity>Team [a-z]{12}) has specified (?P<value>\w+) as the tool for (?P<condition>.+)\.',
)


def audit_suite(suite, splits=SPLITS):
    """Parse actual inputs; fail closed on absent/support/swap/leakage violations."""
    namespaces, scenario_names, audit = {}, {}, {}
    if set(suite) != set(splits):
        raise ValueError('Unexpected suite splits')
    for split in splits:
        rows = suite[split]
        if len(rows) != SCENARIOS[split] * 5:
            raise ValueError('Wrong split row count')
        validate_rows(rows)
        groups = collections.defaultdict(list)
        conditional = collections.defaultdict(collections.Counter)
        cells = collections.defaultdict(collections.Counter)
        namespaces[split], scenario_names[split] = set(), set()
        for row in rows:
            if (row['split'] != split or row['family'] not in CONDITIONS or row['difficulty'] not in LEVELS
                    or row['template_id'] not in TEMPLATES[split] or row['kind'] not in ('swap', 'unknown')):
                raise ValueError('Invalid row stratum')
            if row['condition'] != CONDITIONS[row['family']][row['queried_index']] or row['question'] != question(row['entity'], row['condition']):
                raise ValueError('Question metadata disagrees with input')
            if not 0 <= row['absent_index'] < 3:
                raise ValueError('Invalid absent condition')
            names = row['entities']
            if len(set(names)) != len(names) or any(not re.fullmatch(r'Team [a-z]{12}', n) for n in names) or row['entity'] != names[0]:
                raise ValueError('Invalid opaque entity namespace')
            facts = []
            for line in row['context'].splitlines():
                match = re.fullmatch(_PATTERNS[row['template_id']], line)
                if match is None:
                    raise ValueError('Unparseable semantic record')
                fact = match.groupdict()
                if fact['entity'] not in names or fact['value'] not in VALUES:
                    raise ValueError('Invalid input entity/value')
                facts.append((line, fact))
            if len(facts) != row['difficulty'] + 2:
                raise ValueError('Wrong record count')
            supports = [(line, f) for line, f in facts if f['entity'] == row['entity'] and f['condition'] == row['condition']]
            absent_condition = CONDITIONS[row['family']][row['absent_index']]
            if any(f['entity'] == row['entity'] and f['condition'] == absent_condition for _, f in facts):
                raise ValueError('Absent condition is present in input')
            if row['kind'] == 'unknown':
                if (supports or row['answer'] != 'UNKNOWN' or row['support'] is not None or row['hard_negative'] is not None
                        or row['pair_id'] is not None or row['queried_index'] != row['absent_index']):
                    raise ValueError('Invalid UNKNOWN semantics')
            else:
                if (len(supports) != 1 or supports[0][1]['value'] != row['answer'] or supports[0][0] != row['support']
                        or row['answer'] not in VALUES or row['foil'] not in VALUES or row['pair_id'] is None):
                    raise ValueError('Invalid support semantics')
                negative = [(line, f) for line, f in facts if line == row['hard_negative']]
                if (len(negative) != 1 or negative[0][1]['entity'] != row['entity']
                        or negative[0][1]['condition'] == row['condition'] or negative[0][1]['value'] != row['foil']):
                    raise ValueError('Invalid hard-negative semantics')
            groups[row['scenario_id']].append(row)
            conditional[(row['family'], row['difficulty'], row['template_id'], row['condition'])][row['kind']] += 1
            cells[(row['family'], row['difficulty'])][(row['absent_index'], row['template_id'])] += 1
        expected = 30 if split in ('train', 'test') else 15
        if len(cells) != 16 or any(sum(c.values()) != expected for c in cells.values()):
            raise ValueError('Wrong family/difficulty counts')
        repetitions = 10 if split == 'test' else 5
        expected_combinations = {(a, t) for a in range(3) for t in TEMPLATES[split]}
        if any(set(c) != expected_combinations or set(c.values()) != {repetitions} for c in cells.values()):
            raise ValueError('Missing-condition/template balance violated')
        if any(c['swap'] != 4*c['unknown'] or c['unknown'] == 0 for c in conditional.values()):
            raise ValueError('Conditional known:UNKNOWN balance violated')
        unknown_variants = collections.Counter(r['variant'] for r in rows if r['kind'] == 'unknown')
        if unknown_variants[0] != unknown_variants[1]:
            raise ValueError('UNKNOWN context variants are not counterbalanced')
        if split in ('train', 'test'):
            for family, level in cells:
                variants = collections.Counter(r['variant'] for r in rows if r['kind'] == 'unknown'
                                               and r['family'] == family and r['difficulty'] == level)
                if variants != {0: 3, 1: 3}:
                    raise ValueError('UNKNOWN cell context variants are not counterbalanced')
        for scenario, group in groups.items():
            if (len(group) != 5 or sum(r['kind'] == 'unknown' for r in group) != 1
                    or len({r['pair_id'] for r in group if r['pair_id']}) != 2):
                raise ValueError('Incomplete five-row scenario')
            for key in ('entity', 'family', 'difficulty', 'template_id', 'absent_index', 'entities'):
                if any(r[key] != group[0][key] for r in group):
                    raise ValueError('Scenario shared metadata disagrees')
            contexts = {r['variant']: r['context'] for r in group if r['kind'] == 'swap'}
            if set(contexts) != {0, 1} or any(r['context'] != contexts[r['variant']] for r in group):
                raise ValueError('Scenario context structures disagree')
            if {r['queried_index'] for r in group if r['kind'] == 'swap'} != set(range(3)) - {group[0]['absent_index']}:
                raise ValueError('Incomplete present-condition queries')
            names = set(group[0]['entities'])
            if namespaces[split] & names:
                raise ValueError('Entity reused across independent scenarios')
            namespaces[split].update(names)
            scenario_names[split].add(scenario)
        audit[split] = dict(rows=len(rows), scenarios=len(groups), pairs=2*len(groups), known=4*len(groups), unknown=len(groups),
            unknown_context_variant_counts=dict(unknown_variants),
            conditional_counts=[dict(family=k[0], difficulty=k[1], template=k[2], condition=k[3], **dict(v)) for k, v in conditional.items()])
    for i, a in enumerate(splits):
        for b in splits[i+1:]:
            if namespaces[a] & namespaces[b] or scenario_names[a] & scenario_names[b]:
                raise ValueError('Cross-split namespace leakage')
    label_audit = {}
    if 'train' in suite:
        train = [r for r in suite['train'] if r['kind'] == 'swap']
        global_counts = collections.Counter(r['answer'] for r in train)
        if global_counts != collections.Counter(dict.fromkeys(VALUES, 48)):
            raise ValueError('Global train label balance violated')
        label_audit['global'] = dict(global_counts)
        for axis, output in (('family', 'by_family'), ('difficulty', 'by_difficulty')):
            counters = {str(key): dict(collections.Counter(r['answer'] for r in train if r[axis] == key)) for key in sorted({r[axis] for r in train})}
            if any(not SUBGROUP_BOUND[0] <= c.get(v, 0) <= SUBGROUP_BOUND[1] for c in counters.values() for v in VALUES):
                raise ValueError('Train subgroup label bound violated')
            label_audit[output] = counters
    return dict(passed=True, splits=audit, train_label_counts=label_audit, train_subgroup_label_bound=list(SUBGROUP_BOUND),
        cluster='One scenario contains two pairs and one UNKNOWN; never treat its pairs as independent clusters.',
        naming='v2: independent equal-length opaque names, disjoint across every scenario and split; no prefix overlap')


def freeze_suite(output, seed=73):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config = dict(seed=seed, scenarios=SCENARIOS, levels=list(LEVELS), templates={s: list(TEMPLATES[s]) for s in SPLITS},
        train_subgroup_label_bound=list(SUBGROUP_BOUND), protocol_version=2)
    suite = make_suite(seed)
    audit = audit_suite(suite)
    (output / 'semantic_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    manifest = dict(schema_version=2, generator='binding.paired_data', config=config,
        config_sha256=canonical_hash(config), source_hashes=source_hashes(), splits={}, semantic_audit_sha256=sha256_file(output / 'semantic_audit.json'))
    for split, rows in suite.items():
        path = output / f'{split}.jsonl'
        write_jsonl(path, rows)
        manifest['splits'][split] = dict(rows=len(rows), scenarios=SCENARIOS[split], sha256=sha256_file(path))
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def load_suite(output, splits=SPLITS):
    output = Path(output)
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['schema_version'] != 2 or canonical_hash(manifest['config']) != manifest['config_sha256']:
        raise ValueError('Invalid suite config hash/version')
    if sha256_file(output / 'semantic_audit.json') != manifest['semantic_audit_sha256']:
        raise ValueError('Semantic audit hash mismatch')
    suite = {}
    for split in splits:
        path = output / f'{split}.jsonl'
        if sha256_file(path) != manifest['splits'][split]['sha256']:
            raise ValueError('Split hash mismatch: ' + split)
        suite[split] = read_jsonl(path)
    audit_suite(suite, splits)
    return suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=73)
    args = parser.parse_args()
    print(json.dumps(freeze_suite(args.output, args.seed), indent=2))


if __name__ == '__main__':
    main()
