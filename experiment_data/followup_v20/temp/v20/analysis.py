"""V20 analysis, fixed before any V20 GPU work. Pure Python, CPU only.

It is run on the host by collect.py and again offline on the evidence archive:
    python analysis.py <root> [out.json]
where <root> holds the files under the same relative paths as on the host.

Definitions
  exact            normalised exact match used by the original metrics (binding.v11_pipeline.em_f1)
  choice           the two candidate strings are the two target values of the unit in the current context;
                   mean token log probability; an exact tie is an error
  T                all four target answers correct in the original and swapped contexts (per unit)
  U                both unaffected answers correct in all four contexts (per unit)
  absent code      a four-digit output that is not a value of the current record
  digit advantage  positions an absent code shares with the correct code minus its mean over other record values
  same-row substitution  a wrong output equal to another field of the queried row
  slot rule        badge or route question -> first rendered value of the row; locker question -> second
  intervals        percentile intervals from 10,000 cluster bootstrap resamples (ratio of totals), RNG 20261113;
                   clusters are source scenes, or articles for the development original cohort
"""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import math
import random
import re
import string
import sys

RESAMPLES = 10000
RNG = 20261113
OLD_SEEDS = (73, 97)
NEW_SEEDS = (101, 131, 151)
T975 = {1: 12.7062047362, 2: 4.3026527297, 3: 3.1824463053, 4: 2.7764451052, 5: 2.5705818366}
SFT_LABELS = ('main-compressed-sft-seed73-step512', 'main-compressed-sft-seed97-step512',
              'main-full-sft-seed73-step512', 'main-full-sft-seed97-step512')
COHORTS = {
    'development_crossed': 'data/v16-role-balance-20261106/development.jsonl',
    'new_scene_crossed': 'data/v19-prospective-20261108/crossed.jsonl',
    'new_scene_original': 'data/v19-prospective-20261108/original.jsonl',
    'development_original': 'data/v13-answerable-relations-20261026/development.jsonl',
}
ORIGINAL_RUNS = {
    'sft': 'runs/v13-answerable-reading-20261026/main-compressed-sft-seed{seed}',
    'bottleneck': 'runs/v14-bottleneck-reading-20261026/main-compressed-bottleneck-seed{seed}',
}
V20_RESULTS = 'results/v20-seeds-crossed-choice-20261113'
V20_RUNS = 'runs/v20-seed-extension-20261113'
VIEW = ('semantic_rotation', 'layout_rotation', 'target_team_position')
BONFERRONI_TESTS = 6  # three new seeds by two contrasts on the new scenes


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(4194304), b''):
            h.update(block)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf8'))


def rows(p):
    return [json.loads(s) for s in Path(p).read_text(encoding='utf8').splitlines() if s.strip()]


def normalize(value):
    value = value.casefold().translate(str.maketrans('', '', string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', value).split())


def exact(prediction, gold):
    return normalize(prediction) == normalize(gold)


def hamming(a, b):
    return sum(x != y for x, y in zip(a, b))


def quantile(a, p):
    x = (len(a) - 1) * p
    i = int(x)
    return a[i] * (1 - (x - i)) + a[min(i + 1, len(a) - 1)] * (x - i)


def ratio_bootstrap(clusters, level=.95, resamples=RESAMPLES, seed=RNG):
    """clusters: one (numerator, denominator) per cluster. Statistic: sum of numerators / sum of denominators."""
    def calc(sample):
        d = sum(c[1] for c in sample)
        return sum(c[0] for c in sample) / d if d else None
    point = calc(clusters)
    rng = random.Random(seed)
    k = len(clusters)
    draws = []
    for _ in range(resamples):
        v = calc([clusters[rng.randrange(k)] for _ in range(k)])
        if v is not None:
            draws.append(v)
    # Undefined resamples are never dropped silently: the interval is then reported as undefined.
    valid = point is not None and len(draws) == resamples
    draws.sort()
    lo = (1 - level) / 2
    per = [c[0] / c[1] for c in clusters if c[1]]
    return dict(estimate=point, interval=[quantile(draws, lo), quantile(draws, 1 - lo)] if valid else [None, None],
                level=level, clusters=k, valid_resamples=len(draws), numerator=sum(c[0] for c in clusters),
                denominator=sum(c[1] for c in clusters), per_cluster_min=min(per) if per else None,
                per_cluster_max=max(per) if per else None)


def decide(row):
    """The choice from the stored candidate means (None is an exact tie), checked against the token scores."""
    scores = row['candidate_scores']
    assert len(scores) == 2 and len(scores[0]['token_ids']) == len(scores[1]['token_ids'])
    means = [s['mean'] for s in scores]
    assert all(math.isfinite(m) for m in means)
    decision = None if means[0] == means[1] else int(means[1] > means[0])
    assert decision == row['choice'], 'Stored choice differs from the stored candidate means'
    again = [math.fsum(s['token_logp']) / len(s['token_logp']) for s in scores]
    assert all(abs(a - b) <= 1e-4 for a, b in zip(again, means)), 'Token scores do not reproduce the stored means'
    return decision, means


def row_fields(unit, i, key):
    line = next(s for s in unit['contexts'][i].splitlines() if s.startswith('Team %s, %s mode: ' % (key[0], key[1])))
    return re.findall(r'\b(badge|locker|route) (\d{4})\b', line)


def classify(unit, i, j, prediction):
    key = unit['query_keys'][j]
    gold = unit['answers'][i][j]
    v = prediction.strip()
    fields = row_fields(unit, i, key)
    out = dict(label=key[2], value_slot=[f for f, _ in fields].index(key[2]) + 1, rule_slot=2 if key[2] == 'locker' else 1)
    values = [f['value'] for f in unit['facts'][i]]
    if exact(prediction, gold):
        out['category'] = 'correct'
    else:
        hits = [pos + 1 for pos, (_, code) in enumerate(fields) if code == v]
        if hits:
            out.update(category='same_row', predicted_slot=hits[0], follows_rule=hits[0] == out['rule_slot'])
        elif j < 2 and v == unit['answers'][i][1 - j]:
            out['category'] = 'other_candidate'
        elif v in values:
            out['category'] = 'other_record'
        elif re.fullmatch('[0-9]{4}', v):
            out['category'] = 'absent_code'
        else:
            out['category'] = 'other'
    return out


def joint(ok, questions, contexts):
    return sum(all(v[i, j] for i in contexts for j in questions) for v in ok.values())


def original_cell(data, units, cluster_key, level=.95):
    """Original-layout cohort: choice against generation on target rows, absent codes, joint successes."""
    z = defaultdict(Counter)
    digit = defaultdict(float)
    ok_unit = defaultdict(dict)
    choice_unit = defaultdict(dict)
    scored = False
    for r in data:
        u = units[r['unit_id']]
        i, j = r['context_index'], r['question_index']
        gold = u['answers'][i][j]
        assert gold == r['gold']
        c = z[u[cluster_key]]
        ok = exact(r['prediction'], gold)
        v = r['prediction'].strip()
        c['rows'] += 1
        c['correct'] += ok
        c['strip_disagreement'] += ok != (v == gold)
        ok_unit[u['id']][i, j] = ok
        if j < 2 and 'candidate_scores' in r:
            scored = True
            decision, _ = decide(r)
            good = decision == j
            choice_unit[u['id']][i, j] = good
            c['target'] += 1
            c['choice'] += good
            c['generation'] += ok
            c['tie'] += decision is None
            c['failed'] += not ok
            c['rescued'] += good and not ok
            c['lost'] += ok and not good
        values = [f['value'] for f in u['facts'][i]]
        if re.fullmatch('[0-9]{4}', v) and v not in values:
            others = [hamming(v, x) for x in values if x != gold]
            c['absent'] += 1
            digit[u[cluster_key]] += sum(others) / len(others) - hamming(v, gold)
    keys = sorted(z)
    total = Counter()
    for k in keys:
        total.update(z[k])
    out = dict(units=len(ok_unit), clusters=len(keys), cluster_key=cluster_key, rows=total['rows'], correct=total['correct'],
               errors=total['rows'] - total['correct'], absent=total['absent'], strip_disagreements=total['strip_disagreement'],
               target_joint=joint(ok_unit, (0, 1), (0, 1)), unaffected_joint=joint(ok_unit, (2, 3), (0, 1, 2, 3)),
               digit_advantage=ratio_bootstrap([(digit[k], z[k]['absent']) for k in keys], level))
    if scored:
        out.update(
            target_rows=total['target'], choice_correct=total['choice'], generation_correct=total['generation'],
            ties=total['tie'], failed_generations=total['failed'], choice_correct_on_failed=total['rescued'],
            generation_correct_choice_wrong=total['lost'], choice_target_joint=joint(choice_unit, (0, 1), (0, 1)),
            gap=ratio_bootstrap([(z[k]['choice'] - z[k]['generation'], z[k]['target']) for k in keys], level),
            gap_95=ratio_bootstrap([(z[k]['choice'] - z[k]['generation'], z[k]['target']) for k in keys], .95),
            digit_advantage_95=ratio_bootstrap([(digit[k], z[k]['absent']) for k in keys], .95),
            choice_on_failed=ratio_bootstrap([(z[k]['rescued'], z[k]['failed']) for k in keys], .95))
    return out


def crossed_cell(data, units):
    """Crossed-view cohort: accuracy by view, slot-rule tabulation, and choice against generation on target rows."""
    z = defaultdict(Counter)
    strata = defaultdict(Counter)
    margin = defaultdict(float)
    ok_unit = defaultdict(dict)
    choice_unit = defaultdict(dict)
    view_of = {}
    scored = False
    for r in data:
        u = units[r['unit_id']]
        i, j = r['context_index'], r['question_index']
        assert u['answers'][i][j] == r['gold']
        view = 'identity' if all(u[k] == 0 for k in VIEW) else 'transformed'
        view_of[u['id']] = view
        info = classify(u, i, j, r['prediction'])
        ok = info['category'] == 'correct'
        c = z[u['source_id']]
        c['rows', view] += 1
        c['correct', view] += ok
        c['category', info['category']] += 1
        if info['category'] == 'same_row':
            c['same_row', 'all'] += 1
            c['follows', 'all'] += info['follows_rule']
            c['same_row', view] += 1
            c['follows', view] += info['follows_rule']
        ok_unit[u['id']][i, j] = ok
        if j < 2 and 'candidate_scores' in r:
            scored = True
            decision, means = decide(r)
            good = decision == j
            choice_unit[u['id']][i, j] = good
            c['target', view] += 1
            c['choice', view] += good
            c['generation', view] += ok
            c['tie', view] += decision is None
            if view == 'transformed':
                for name in (('slot', info['value_slot']), ('label', info['label']),
                             ('value_in_rule_slot', info['value_slot'] == info['rule_slot']),
                             ('label_rotation', u['semantic_rotation']), ('order_rotation', u['layout_rotation']),
                             ('team_exchange', u['target_team_position']), ('context', i),
                             ('generation', info['category'])):
                    s = strata[name]
                    s['n'] += 1
                    s['choice'] += good
                    s['generation'] += ok
                    margin[name] += means[j] - means[1 - j]
    keys = sorted(z)
    total = Counter()
    for k in keys:
        total.update(z[k])
    units_by_view = Counter(view_of.values())
    out = dict(units=len(ok_unit), scenes=len(keys), units_by_view=dict(units_by_view),
               rows={v: total['rows', v] for v in ('identity', 'transformed')},
               correct={v: total['correct', v] for v in ('identity', 'transformed')},
               categories={k[1]: n for k, n in sorted(total.items()) if k[0] == 'category'},
               same_row=total['same_row', 'all'], follows_rule=total['follows', 'all'],
               same_row_by_view={v: [total['follows', v], total['same_row', v]] for v in ('identity', 'transformed')},
               rule_share=ratio_bootstrap([(z[k]['follows', 'all'], z[k]['same_row', 'all']) for k in keys]),
               accuracy_transformed=ratio_bootstrap([(z[k]['correct', 'transformed'], z[k]['rows', 'transformed']) for k in keys]),
               generation_target_joint={v: sum(all(ok[i, j] for i in (0, 1) for j in (0, 1))
                                               for uid, ok in ok_unit.items() if view_of[uid] == v)
                                        for v in ('identity', 'transformed')})
    if scored:
        t = 'transformed'
        out.update(
            target_rows={v: total['target', v] for v in ('identity', t)},
            choice_correct={v: total['choice', v] for v in ('identity', t)},
            generation_correct={v: total['generation', v] for v in ('identity', t)},
            ties={v: total['tie', v] for v in ('identity', t)},
            choice_accuracy=ratio_bootstrap([(z[k]['choice', t], z[k]['target', t]) for k in keys]),
            generation_accuracy=ratio_bootstrap([(z[k]['generation', t], z[k]['target', t]) for k in keys]),
            gap=ratio_bootstrap([(z[k]['choice', t] - z[k]['generation', t], z[k]['target', t]) for k in keys]),
            choice_target_joint={v: sum(all(ok[i, j] for i in (0, 1) for j in (0, 1))
                                        for uid, ok in choice_unit.items() if view_of[uid] == v)
                                 for v in ('identity', t)},
            strata=[dict(factor=str(name[0]), level=str(name[1]), n=s['n'], choice_correct=s['choice'],
                         generation_correct=s['generation'], mean_margin=margin[name] / s['n'])
                    for name, s in sorted(strata.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1])))])
    return out


def stats(values):
    n = len(values)
    mean = math.fsum(values) / n
    if n < 2:
        return dict(n=n, mean=mean, sd=None, interval_95=[None, None])
    sd = math.sqrt(math.fsum((v - mean) ** 2 for v in values) / (n - 1))
    half = T975[n - 1] * sd / math.sqrt(n)
    return dict(n=n, mean=mean, sd=sd, se=sd / math.sqrt(n), interval_95=[mean - half, mean + half],
                minimum=min(values), maximum=max(values))


def normal_cdf(x):
    return .5 * (1 + math.erf(x / math.sqrt(2)))


def seed_summary(cells):
    """cells[(method, seed)] = the 512-update development result stored in summary.json['evaluations']['512']."""
    seeds = sorted({s for _, s in cells}, key=lambda s: (s not in OLD_SEEDS, s))
    table = []
    for seed in seeds:
        row = dict(seed=seed, new=seed in NEW_SEEDS)
        for method in ('sft', 'bottleneck'):
            e = cells[method, seed]
            row[method] = dict(target_joint=e['probe']['counts']['target_joint'][0],
                               unaffected_joint=e['probe']['counts']['locality_joint'][0],
                               controlled_rows=e['probe']['counts']['known_exact'][0],
                               natural_f1=e['natural']['known_f1'],
                               natural_exact=e['natural']['counts']['known_exact'][0])
            assert e['probe']['counts']['target_joint'][1] == 96 and e['probe']['counts']['known_exact'][1] == 1536
        a, b = cells['bottleneck', seed], cells['sft', seed]
        dt = 100 * (a['probe']['rates']['target_joint'] - b['probe']['rates']['target_joint'])
        du = 100 * (a['probe']['rates']['locality_joint'] - b['probe']['rates']['locality_joint'])
        df = 100 * (a['natural']['known_f1'] - b['natural']['known_f1'])
        row.update(target_difference_pp=dt, unaffected_difference_pp=du, natural_f1_difference_pp=df,
                   gain_at_least_5=dt >= 5 - 1e-9, passes=dt >= 5 - 1e-9 and du >= -2 - 1e-9 and df >= -2 - 1e-9)
        table.append(row)
    out = dict(seeds=table)
    groups = dict(all=table, new=[r for r in table if r['new']], original=[r for r in table if not r['new']])
    for name, group in groups.items():
        if not group:
            continue
        d = stats([r['target_difference_pp'] for r in group])
        d.update(seeds=[r['seed'] for r in group], gains_at_least_5=sum(r['gain_at_least_5'] for r in group),
                 passes=sum(r['passes'] for r in group),
                 upper_bound_below_5=d['interval_95'][1] is not None and d['interval_95'][1] < 5,
                 interval_excludes_zero=d['interval_95'][0] is not None and (d['interval_95'][0] > 0 or d['interval_95'][1] < 0))
        out['difference_' + name] = d
        out['sft_target_pp_' + name] = stats([100 * r['sft']['target_joint'] / 96 for r in group])
        out['bottleneck_target_pp_' + name] = stats([100 * r['bottleneck']['target_joint'] / 96 for r in group])
        out['unaffected_difference_' + name] = stats([r['unaffected_difference_pp'] for r in group])
        out['natural_f1_difference_' + name] = stats([r['natural_f1_difference_pp'] for r in group])
    new = groups['new']
    out['criterion_met_at_all_new_seeds'] = bool(new) and all(r['passes'] for r in new)
    sd = out.get('difference_all', {}).get('sd')
    if sd:
        # What a both-seeds rule with a five-point threshold passes when the gain at each seed is normal with
        # the between-seed standard deviation estimated here (side conditions ignored, so an upper bound).
        out['two_seed_rule'] = {str(mu): (1 - normal_cdf((5 - mu) / sd)) ** 2 for mu in (0, 5, 10)}
    return out


def metric_counts(data):
    """Recompute the joint counts of binding.v13_protocol.metrics for one family from raw rows."""
    groups = defaultdict(dict)
    for r in data:
        groups[r['unit_id']][r['context_index'], r['question_index']] = exact(r['prediction'], r['gold'])
    probe = data[0]['family'] == 'probe'
    targets, local = ((0, 1), (2, 3)) if probe else ((0,), (1,))
    return dict(known_exact=sum(sum(g.values()) for g in groups.values()), rows=len(data), units=len(groups),
                target_joint=sum(all(g[i, j] for i in (0, 1) for j in targets) for g in groups.values()),
                locality_joint=sum(all(g[i, j] for i in range(4) for j in local if (i, j) in g) for g in groups.values()))


def run(root):
    root = Path(root)
    res = root / V20_RESULTS
    runs = root / V20_RUNS
    inputs = {}

    def pin(p):
        inputs[Path(p).relative_to(root).as_posix()] = sha(p)
        return p

    units = {}
    for name, path in COHORTS.items():
        if (root / path).exists():
            units[name] = {u['id']: u for u in rows(pin(root / path)) if u['family'] == 'probe'}
    report = dict(resamples=RESAMPLES, rng=RNG)

    def cells(folder):
        out = {}
        for label_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
            for cohort_dir in sorted(p for p in label_dir.iterdir() if p.is_dir()):
                name = cohort_dir.name
                if name not in units or not (cohort_dir / 'receipt.json').exists():
                    continue
                receipt = read(pin(cohort_dir / 'receipt.json'))
                data = rows(pin(cohort_dir / 'rows.jsonl'))
                assert sha(cohort_dir / 'rows.jsonl') == receipt['rows_sha256']
                assert len(data) == receipt['rows'] == 16 * len(units[name])
                counts = metric_counts(data)
                assert counts['known_exact'] == receipt['metrics']['counts']['known_exact'][0]
                assert counts['target_joint'] == receipt['metrics']['counts']['target_joint'][0]
                assert counts['locality_joint'] == receipt['metrics']['counts']['locality_joint'][0]
                if name.endswith('_crossed'):
                    result = crossed_cell(data, units[name])
                else:
                    level = 1 - .05 / BONFERRONI_TESTS if name == 'new_scene_original' else .95
                    result = original_cell(data, units[name], 'article' if name == 'development_original' else 'source_id', level)
                result.update(reader_identity=receipt['reader_identity'], generated=receipt['generated'],
                              replay_rows=receipt['replay_rows'])
                out.setdefault(label_dir.name, {})[name] = result
        return out

    if (res / 'crossed_choice').exists():
        report['part_a'] = cells(res / 'crossed_choice')
    evaluations = {}
    for method, pattern in ORIGINAL_RUNS.items():
        for seed in OLD_SEEDS:
            p = root / pattern.format(seed=seed) / 'summary.json'
            if p.exists():
                evaluations[method, seed] = read(pin(p))['evaluations']['512']
        for seed in NEW_SEEDS:
            folder = runs / f'main-compressed-{method}-seed{seed}'
            if not (folder / 'summary.json').exists():
                continue
            summary = read(pin(folder / 'summary.json'))
            if summary['status'] != 'success' or summary['optimizer_steps'] != 512:
                continue
            e = summary['evaluations']['512']
            raw = pin(folder / 'evaluation-512/predictions.jsonl')
            assert sha(raw) == e['predictions_sha256']
            data = rows(raw)
            for family in ('probe', 'natural'):
                counts = metric_counts([r for r in data if r['family'] == family])
                for k in ('known_exact', 'target_joint', 'locality_joint'):
                    assert counts[k] == e[family]['counts'][k][0], (method, seed, family, k)
            evaluations[method, seed] = e
    complete = sorted(s for s in OLD_SEEDS + NEW_SEEDS if all((m, s) in evaluations for m in ORIGINAL_RUNS))
    if complete:
        report['part_b'] = seed_summary({(m, s): evaluations[m, s] for m in ORIGINAL_RUNS for s in complete})
        report['part_b']['checkpoints'] = {}
        for method in ORIGINAL_RUNS:
            for seed in NEW_SEEDS:
                p = runs / f'main-compressed-{method}-seed{seed}' / 'summary.json'
                if p.exists():
                    s = read(p)
                    report['part_b']['checkpoints'][f'{method}-{seed}'] = dict(
                        target_joint={k: v['probe']['counts']['target_joint'][0] for k, v in s['evaluations'].items()},
                        best_development_step=s['best_development_step'],
                        initial_reader_identity=s['initial_reader_identity'], final_reader_identity=s['final_reader_identity'],
                        cache_sha256=s['cache_sha256'], training_seconds=s['cumulative_training_seconds'],
                        wall_seconds=s['wall_seconds'], peak_allocated_mib=s['peak_allocated_mib'])
    for method in ORIGINAL_RUNS:
        p = runs / 'replay32' / f'replay_check-{method}.json'
        if p.exists():
            report.setdefault('replay', {})[method] = read(pin(p))
    if (res / 'new_seeds').exists():
        report['part_c'] = cells(res / 'new_seeds')
        tests = {}
        for label, by in report['part_c'].items():
            c = by.get('new_scene_original')
            if c and 'gap' in c:
                tests[label] = dict(gap_lower=c['gap']['interval'][0], digit_lower=c['digit_advantage']['interval'][0],
                                    level=c['gap']['level'],
                                    both_positive=bool(c['gap']['interval'][0] is not None and c['gap']['interval'][0] > 0
                                                       and c['digit_advantage']['interval'][0] is not None
                                                       and c['digit_advantage']['interval'][0] > 0))
        report['new_scene_tests'] = dict(readers=tests, all_pass=len(tests) == len(NEW_SEEDS) and all(t['both_positive'] for t in tests.values()))
    report['inputs'] = inputs
    return report


if __name__ == '__main__':
    result = run(sys.argv[1])
    text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(text, encoding='utf8')
    print(json.dumps({k: (sorted(v) if isinstance(v, dict) else v) for k, v in result.items() if k != 'inputs'})[:2000])
