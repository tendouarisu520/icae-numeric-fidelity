"""Strict generated-answer evaluation with pair-level uncertainty."""
import collections
import random
import re


def normalize(text):
    # Never accept a substring match: 'Rust or Python' is not a correct answer.
    return re.sub(r'\s+', ' ', str(text).strip().lower()).strip(' .!\n\r\t`\"\'')


def _index(rows, predictions):
    allowed = {r['id'] for r in rows}
    if len(allowed) != len(rows):
        raise ValueError('Duplicate reference IDs')
    indexed = {}
    for p in predictions:
        if p['id'] not in allowed or p['id'] in indexed:
            raise ValueError('Duplicate or foreign prediction ID: ' + p['id'])
        indexed[p['id']] = p
    return indexed


def _scores(rows, predictions):
    indexed = _index(rows, predictions)
    return {r['id']: int(r['id'] in indexed and normalize(indexed[r['id']]['prediction']) == normalize(r['answer'])) for r in rows}


def pair_scores(rows, predictions):
    scores = _scores(rows, predictions)
    groups = collections.defaultdict(list)
    for row in rows:
        if row['pair_id'] is not None:
            groups[row['pair_id']].append(scores[row['id']])
    if any(len(v) != 2 for v in groups.values()):
        raise ValueError('References must contain complete pairs')
    return {k: min(v) for k, v in groups.items()}


def evaluate_predictions(rows, predictions):
    if not rows:
        raise ValueError('No reference examples')
    indexed = _index(rows, predictions)
    correct = _scores(rows, predictions)
    pairs = pair_scores(rows, predictions)
    swap = [r for r in rows if r['kind'] == 'swap']
    unknown = [r for r in rows if r['kind'] == 'unknown']
    mismatches = sum(normalize(indexed.get(r['id'], {}).get('prediction', '')) == normalize(r['foil']) for r in swap)
    return {
        'n': len(rows), 'n_pairs': len(pairs), 'missing': len(rows) - len(indexed),
        'accuracy': sum(correct.values()) / len(rows),
        'pair_accuracy': sum(pairs.values()) / len(pairs) if pairs else None,
        'misbinding_rate': mismatches / len(swap) if swap else None,
        'unknown_accuracy': sum(correct[r['id']] for r in unknown) / len(unknown) if unknown else None,
        'by_family': {f: sum(correct[r['id']] for r in rows if r['family'] == f) / sum(r['family'] == f for r in rows) for f in sorted({r['family'] for r in rows})},
    }


def paired_bootstrap(rows, predictions_a, predictions_b, seed=73, repetitions=2000):
    a, b = pair_scores(rows, predictions_a), pair_scores(rows, predictions_b)
    if not a or repetitions < 1:
        raise ValueError('Bootstrap requires complete pairs and positive repetitions')
    # Generator has exactly one pair per independent scenario. Group again by
    # scenario so a future multiple-query extension cannot inflate sample size.
    scenario_for_pair = {r['pair_id']: r['scenario_id'] for r in rows if r['pair_id'] is not None}
    clusters = collections.defaultdict(list)
    for pair in a:
        clusters[scenario_for_pair[pair]].append(a[pair] - b[pair])
    cluster_values = [sum(v) / len(v) for v in clusters.values()]
    rng = random.Random(seed)
    draws = sorted(sum(rng.choices(cluster_values, k=len(cluster_values))) / len(cluster_values) for _ in range(repetitions))
    return {'difference': sum(cluster_values) / len(cluster_values),
            'low': draws[int(.025 * (repetitions - 1))], 'high': draws[int(.975 * (repetitions - 1))],
            'clusters': len(clusters), 'seed': seed, 'repetitions': repetitions}
