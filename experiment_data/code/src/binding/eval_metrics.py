"""Strict output diagnostics and paired scenario-cluster uncertainty.

Diagnostic extraction never changes the primary exact-match prediction.
Unknown controls enter accuracy only. Bootstrap draws resample independent
scenarios, then recompute each ratio with its original example/pair denominator.
"""
import collections
import random
import re

from .data import VALUES, validate_rows
from .metrics import _index, _scores, evaluate_predictions, normalize, pair_scores

LABELS = VALUES + ('UNKNOWN',)
_LABEL_PATTERN = re.compile(r'(?<!\w)(' + '|'.join(map(re.escape, LABELS)) + r')(?!\w)', re.I)


def diagnose_output(text, truncated=False):
    occurrences = [match.group() for match in _LABEL_PATTERN.finditer(text)]
    exact = next((label for label in LABELS if normalize(text) == normalize(label)), None)
    category = ('multiple_labels' if len(occurrences) > 1 else
                'exact_single_label' if exact else
                'single_label_explanatory' if occurrences else 'absent_label')
    return {'category': 'truncated' if truncated else category,
            'content_category': category, 'exact_label': exact,
            'label_occurrences': occurrences, 'truncated': bool(truncated)}


def output_summary(rows, predictions):
    result = evaluate_predictions(rows, predictions)
    categories = collections.Counter(p['format']['category'] for p in predictions)
    result['format_counts'] = dict(categories)
    result['malformed_rate'] = 1 - categories['exact_single_label'] / len(rows)
    result['primary_error_counts'] = dict(collections.Counter(p['primary_error'] for p in predictions if 'primary_error' in p))
    if predictions and all('candidate_diagnostic' in p for p in predictions):
        result['candidate_diagnostic_metrics'] = evaluate_predictions(rows, [
            {'id': p['id'], 'prediction': p['candidate_diagnostic']['prediction']} for p in predictions])
    return result


def scenario_bootstrap(rows, predictions_a, predictions_b, seed=73, repetitions=2000):
    """Percentile 95% CIs for A-minus-B, with metric-specific scenario draws."""
    validate_rows(rows)
    if not rows or repetitions < 1:
        raise ValueError('Bootstrap requires references and positive repetitions')
    a, b = _scores(rows, predictions_a), _scores(rows, predictions_b)
    ia, ib = _index(rows, predictions_a), _index(rows, predictions_b)
    pa, pb = pair_scores(rows, predictions_a), pair_scores(rows, predictions_b)
    clusters = {key: collections.defaultdict(lambda: [0, 0])
                for key in ('accuracy', 'pair_accuracy', 'misbinding_rate')}
    seen = set()
    for row in rows:
        scenario = row['scenario_id']
        value = clusters['accuracy'][scenario]
        value[0] += a[row['id']] - b[row['id']]
        value[1] += 1
        if row['kind'] == 'swap':
            value = clusters['misbinding_rate'][scenario]
            value[0] += (int(normalize(ia.get(row['id'], {}).get('prediction', '')) == normalize(row['foil'])) -
                         int(normalize(ib.get(row['id'], {}).get('prediction', '')) == normalize(row['foil'])))
            value[1] += 1
            pair = row['pair_id']
            if pair not in seen:
                seen.add(pair)
                value = clusters['pair_accuracy'][scenario]
                value[0] += pa[pair] - pb[pair]
                value[1] += 1
    result = {}
    for metric, groups in clusters.items():
        values = list(groups.values())
        if not values:
            result[metric] = None
            continue
        rng = random.Random(seed)
        ratio = lambda sample: sum(x[0] for x in sample) / sum(x[1] for x in sample)
        draws = sorted(ratio(rng.choices(values, k=len(values))) for _ in range(repetitions))
        result[metric] = {'difference': ratio(values),
                          'low': draws[int(.025 * (repetitions - 1))],
                          'high': draws[int(.975 * (repetitions - 1))],
                          'clusters': len(values), 'denominator': sum(v[1] for v in values),
                          'seed': seed, 'repetitions': repetitions,
                          'method': 'paired_scenario_cluster_percentile_95'}
    return result
