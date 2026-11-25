"""Recompute the V19 follow-up results from the raw outputs in completed_evidence.zip. Standard library only.

python -B reproduction/v19_evidence_analysis.py <package_root> <evidence_zip_or_dir> <out_numbers.json> <out_extra.json>

compute() covers the auxiliary arms at 512 and 1024 updates with all seven checkpoints, the two-seed rule
calculation, the loss components of both exposures, balanced-layout SFT, the slot rule on development and on the
new scenes, the prospective replication and the check of Table "by query and context" against the target-joint
counts. compute_extra() gives the error composition on crossed views. The reserved test split is never read.
"""
import collections
import csv
import importlib.util
import io
import json
import math
import pathlib
import random
import re
import string
import sys
import zipfile

SEEDS = (73, 97)
ARMS = (("sft", "SFT"), ("assignment", "Assign."), ("relation", "Assign. + loc."), ("bottleneck", "Weakest + loc."))
STEPS = ("128", "256", "512", "640", "768", "896", "1024")


def rows(path):
    return [json.loads(x) for x in path.read_text(encoding="utf8").splitlines() if x.strip()]


def read_json(path):
    return json.loads(path.read_text(encoding="utf8"))


def read_csv(path):
    return list(csv.DictReader(io.StringIO(path.read_text(encoding="utf8"))))


def normalize(value):
    value = value.casefold().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", value).split())


def em(prediction, gold):
    return normalize(prediction) == normalize(gold)


def f1(prediction, gold):
    a, b = normalize(prediction).split(), normalize(gold).split()
    common = sum((collections.Counter(a) & collections.Counter(b)).values())
    return 2.0 * common / (len(a) + len(b)) if common else 0.0


def probe_metrics(raw):
    groups = collections.defaultdict(dict)
    for r in raw:
        assert r["family"] == "probe"
        groups[r["unit_id"]][r["context_index"], r["question_index"]] = em(r["prediction"], r["gold"])
    out = collections.Counter(units=len(groups), rows=len(raw))
    for c in groups.values():
        assert len(c) == 16
        out["correct"] += sum(c.values())
        out["T"] += all(c[i, j] for i in (0, 1) for j in (0, 1))
        out["U"] += all(c[i, j] for i in range(4) for j in (2, 3))
        out["irrelevant"] += all(c[i, j] for i in (0, 2) for j in (0, 1))
        out["R"] += all(c[i, j] for i in (0, 3) for j in (0, 1))
    return dict(out)


def natural_f1(raw):
    vals = [f1(r["prediction"], r["gold"]) for r in raw if r["family"] == "natural"]
    return math.fsum(vals) / len(vals), sum(em(r["prediction"], r["gold"]) for r in raw if r["family"] == "natural"), len(vals)


def hamming(a, b):
    return sum(x != y for x, y in zip(a, b))


def quantile(values, p):
    a = sorted(values)
    x = (len(a) - 1) * p
    i = int(x)
    return a[i] * (1 - (x - i)) + a[min(i + 1, len(a) - 1)] * (x - i)


def cluster_boot(clusters, stat, resamples=10000, seed=20261113, level=0.95):
    """Percentile interval for stat(list of cluster records); clusters are resampled with replacement."""
    rng = random.Random(seed)
    keys = sorted(clusters)
    draws = []
    for _ in range(resamples):
        v = stat([clusters[keys[rng.randrange(len(keys))]] for _ in keys])
        if v is not None:
            draws.append(v)
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    return dict(estimate=stat([clusters[k] for k in keys]), interval=[quantile(draws, lo), quantile(draws, hi)], level=level, valid=len(draws), resamples=resamples, clusters=len(keys))


def row_line(unit, i, j):
    team, mode, field = unit["query_keys"][j]
    line = next(s for s in unit["contexts"][i].splitlines() if s.startswith("Team %s, %s mode: " % (team, mode)))
    fields = re.findall(r"\b(badge|locker|route) (\d{4})\b", line)
    assert len(fields) == 3
    return field, fields


def identity(u):
    return u["semantic_rotation"] == 0 and u["layout_rotation"] == 0 and u["target_team_position"] == 0


def slot_rule_rows(raw, units):
    """One record per answer on crossed views: requested slot, correctness, same-row substitution and rule match."""
    out = []
    for r in raw:
        u = units[r["unit_id"]]
        i, j = r["context_index"], r["question_index"]
        field, fields = row_line(u, i, j)
        v = r["prediction"].strip()
        gold = u["answers"][i][j]
        assert gold == r["gold"]
        requested = [f for f, c in fields].index(field) + 1
        rule = 2 if field == "locker" else 1
        matches = [k + 1 for k, (f, c) in enumerate(fields) if c == v]
        out.append(dict(source=u["source_id"], context=i, query=j, field=field, requested=requested, rule=rule, correct=v == gold, substitution=bool(matches) and v != gold,
                        follows=bool(matches) and v != gold and matches[0] == rule, identity=identity(u)))
    return out


def slot_detail(raw, units):
    """Transformed views only: accuracy by the slot of the queried value, and what wrong outputs resemble."""
    by_slot = collections.defaultdict(lambda: [0, 0])
    by_field_slot = collections.defaultdict(lambda: [0, 0])
    c = collections.Counter()
    gold_all, gold_off, rule_off, near = [], [], [], 0
    for r in raw:
        u = units[r["unit_id"]]
        if identity(u):
            continue
        i, j = r["context_index"], r["question_index"]
        field, fields = row_line(u, i, j)
        v, gold = r["prediction"].strip(), u["answers"][i][j]
        values = [f["value"] for f in u["facts"][i]]
        requested = [f for f, _ in fields].index(field) + 1
        rule = 2 if field == "locker" else 1
        by_slot[str(requested)][0] += v == gold
        by_slot[str(requested)][1] += 1
        by_field_slot["%s-%d" % (field, requested)][0] += v == gold
        by_field_slot["%s-%d" % (field, requested)][1] += 1
        c["n"] += 1
        c["errors"] += v != gold
        absent = v != gold and re.fullmatch("[0-9]{4}", v) is not None and v not in values
        if absent:
            c["absent"] += 1
            gold_all.append(4 - hamming(v, gold))
            near += hamming(v, gold) <= 2
        if requested != rule:
            c["off_rule_n"] += 1
            c["off_rule_errors"] += v != gold
            c["off_rule_exact_rule_slot"] += v == fields[rule - 1][1]
            if absent:
                c["off_rule_absent"] += 1
                gold_off.append(4 - hamming(v, gold))
                rule_off.append(4 - hamming(v, fields[rule - 1][1]))
    out = dict(c)
    out["by_slot"] = dict(sorted(by_slot.items()))
    out["by_field_slot"] = dict(sorted(by_field_slot.items()))
    out["absent_within_two"] = near
    if gold_all:
        out["absent_gold_matches"] = sum(gold_all) / len(gold_all)
    if gold_off:
        out["off_rule_absent_gold_matches"] = sum(gold_off) / len(gold_off)
        out["off_rule_absent_rule_slot_matches"] = sum(rule_off) / len(rule_off)
    return out


def margins(PKG):
    """Held-out paired margins that the candidate objectives maximize, from the stored choice scores at 512 updates."""
    softplus = lambda x: math.log1p(math.exp(-abs(x))) + max(x, 0.0)
    out = {}
    for split in ("development", "confirmation"):
        for method, _ in ARMS:
            for seed in SEEDS:
                groups = collections.defaultdict(dict)
                for r in rows(PKG / ("followup/choice/%s/main-compressed-%s-seed%d/scores.jsonl" % (split, method, seed))):
                    assert r["candidate_values"][r["question_index"]] == r["gold"]
                    groups[r["unit_id"], r["context_index"]][r["question_index"]] = [s["mean"] for s in r["candidate_scores"]]
                g_avg, l_assign, l_weak = [], [], []
                for sc in groups.values():
                    s00, s01, s10, s11 = sc[0][0], sc[0][1], sc[1][0], sc[1][1]
                    g = (s00 + s11 - s01 - s10) / 2
                    four = [s00 - s01, s11 - s10, s00 - s10, s11 - s01]
                    soft = -0.5 * math.log(math.fsum(math.exp(-x / 0.5) for x in four))
                    g_avg.append(g)
                    l_assign.append(softplus(0.5 - g))
                    l_weak.append(softplus(0.5 - soft))
                out["%s-%s-%d" % (split, method, seed)] = dict(contexts=len(g_avg), median_margin=quantile(g_avg, 0.5), mean_assignment_loss=math.fsum(l_assign) / len(l_assign),
                                                               mean_weakest_loss=math.fsum(l_weak) / len(l_weak), margin_above_half=sum(x > 0.5 for x in g_avg))
    return out


def phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def rule_rate(cs):
    n = sum(x["substitution"] for c in cs for x in c)
    return sum(x["follows"] for c in cs for x in c) / n if n else None


def compute(PKG, EV):
    V19 = EV / "results/v19-budget-layout-20261108"
    res = {}

    def run_dir(method, seed):
        if method == "sft":
            return EV / ("runs/v16-repeated-exposure-20261106/main-compressed-sft-seed%d" % seed)
        return EV / ("runs/v19-budget-layout-20261108/extended-compressed-%s-seed%d" % (method, seed))

    # A. auxiliary arms and SFT at 512 and 1024 updates, and the seven checkpoints
    endpoint, curves = {}, {}
    for method, _ in ARMS:
        for seed in SEEDS:
            d = run_dir(method, seed)
            summ = read_json(d / "summary.json")
            assert summ["status"] == "success" and summ["optimizer_steps"] == 1024 and not summ["reserved_test_accessed"]
            for step in (512, 1024):
                raw = rows(d / ("evaluation-%d/predictions.jsonl" % step))
                pm = probe_metrics([r for r in raw if r["family"] == "probe"])
                nf1, nem, nn = natural_f1(raw)
                ev = summ["evaluations"][str(step)]
                assert pm["T"] == ev["probe"]["counts"]["target_joint"][0] and pm["U"] == ev["probe"]["counts"]["locality_joint"][0]
                assert pm["correct"] == ev["probe"]["counts"]["known_exact"][0] and abs(nf1 - ev["natural"]["known_f1"]) < 1e-12
                endpoint["%s-%d-%d" % (method, seed, step)] = dict(T=pm["T"], U=pm["U"], R=pm["R"], rows=pm["correct"], f1=nf1, em=nem, natural_rows=nn)
            curves["%s-%d" % (method, seed)] = {s: dict(T=e["probe"]["counts"]["target_joint"][0], U=e["probe"]["counts"]["locality_joint"][0], rows=e["probe"]["counts"]["known_exact"][0],
                                                         f1=e["natural"]["known_f1"]) for s, e in summ["evaluations"].items()}
            assert sorted(curves["%s-%d" % (method, seed)], key=int) == list(STEPS)
    res["endpoint"], res["curves"] = endpoint, curves
    for r in read_csv(V19 / "final_analysis/matched_endpoint_comparison.csv"):
        e = endpoint["%s-%s-%s" % (r["method"], r["seed"], r["step"])]
        assert e["T"] == int(r["target_count"]) and abs(e["rows"] / 1536 - float(r["controlled_accuracy"])) < 1e-12 and abs(e["f1"] - float(r["natural_f1"])) < 1e-12
        assert abs(e["U"] / 96 - float(r["unaffected_joint"])) < 1e-12
    delta = {"%s-%d" % (m, step): [100 * (endpoint["%s-%d-%d" % (m, s, step)]["T"] - endpoint["sft-%d-%d" % (s, step)]["T"]) / 96 for s in SEEDS] for m, _ in ARMS[1:] for step in (512, 1024)}
    res["delta_T_pp"] = delta
    gains = {"%s-%s" % (m, step): [curves["%s-%d" % (m, s)][step]["T"] - curves["sft-%d" % s][step]["T"] for s in SEEDS] for m, _ in ARMS[1:] for step in STEPS}
    res["delta_T_groups_all_steps"] = gains
    res["largest_single_seed_gain_groups"] = max(max(v) for v in gains.values())
    res["steps_with_five_points_at_both_seeds"] = sorted(k for k, v in gains.items() if min(v) * 100 / 96 >= 5 - 1e-9)

    # G. what the two-seed rule can detect (post hoc, normal model)
    sd = lambda pairs: math.sqrt(sum((a - b) ** 2 for a, b in pairs) / (2 * len(pairs)))
    p512, p1024 = [delta["%s-512" % m] for m, _ in ARMS[1:]], [delta["%s-1024" % m] for m, _ in ARMS[1:]]
    oc = dict(seed_sd_512=sd(p512), seed_sd_1024=sd(p1024), sft_seed_gap_512=100 * (endpoint["sft-97-512"]["T"] - endpoint["sft-73-512"]["T"]) / 96,
              sft_seed_gap_1024=100 * (endpoint["sft-97-1024"]["T"] - endpoint["sft-73-1024"]["T"]) / 96)
    for name in ("512", "1024"):
        oc["pass_probability_" + name] = {str(g): (1 - phi((5 - g) / oc["seed_sd_" + name])) ** 2 for g in (0, 5, 10, 15)}
    oc["pass_probability_ten_points_double_sd"] = [(1 - phi((5 - 10) / (2 * oc["seed_sd_" + name]))) ** 2 for name in ("512", "1024")]
    oc["mean_delta_pp"] = {k: sum(v) / 2 for k, v in delta.items()}
    res["two_seed_rule"] = oc

    # F. loss components over both exposures
    loss = collections.defaultdict(list)
    for r in read_csv(V19 / "final_analysis/extended_loss_components.csv"):
        loss[r["method"], int(r["seed"])].append({k: float(v) for k, v in r.items() if k not in ("method", "seed", "source_id")})
    mean = lambda xs: math.fsum(xs) / len(xs)
    lo = {}
    for (method, seed), rs in sorted(loss.items()):
        rs.sort(key=lambda r: r["step"])
        assert [int(r["step"]) for r in rs] == list(range(1, 1025))
        for r in rs:
            assert abs(r["weighted_ce"] + r["weighted_aux"] + r["weighted_locality"] - r["total_loss"]) < 1e-6
            assert abs(r["weighted_aux"] - 0.1 * r["raw_aux"]) < 1e-9 and abs(r["weighted_ce"] - 0.5 * (r["natural_ce"] + r["probe_ce"])) < 1e-9
            assert abs(r["weighted_locality"] - 0.025 * (r["natural_locality"] + r["probe_locality"])) < 1e-9
        o = {}
        for name, seg in (("first", rs[:128]), ("last1", rs[384:512]), ("last2", rs[896:1024])):
            ce = mean([r["weighted_ce"] for r in seg])
            o[name] = dict(ce=ce, weighted_aux=mean([r["weighted_aux"] for r in seg]), raw_aux=mean([r["raw_aux"] for r in seg]), weighted_locality=mean([r["weighted_locality"] for r in seg]))
            o[name]["weighted_aux_share"] = o[name]["weighted_aux"] / ce
            o[name]["raw_aux_share"] = o[name]["raw_aux"] / ce
            o[name]["weighted_locality_share"] = o[name]["weighted_locality"] / ce
        o["aux_drop_first_to_last1"] = o["first"]["weighted_aux"] / o["last1"]["weighted_aux"]
        o["ce_drop_first_to_last1"] = o["first"]["ce"] / o["last1"]["ce"]
        if o["first"]["weighted_locality"]:
            o["locality_drop_first_to_last1"] = o["first"]["weighted_locality"] / o["last1"]["weighted_locality"]
        lo["%s-%d" % (method, seed)] = o
    res["loss"] = lo
    span = lambda f: [min(f(v) for v in lo.values()), max(f(v) for v in lo.values())]
    loc = [v for v in lo.values() if "locality_drop_first_to_last1" in v]
    res["loss_ranges"] = dict(weighted_aux_share_last1=span(lambda v: v["last1"]["weighted_aux_share"]), raw_aux_share_last1=span(lambda v: v["last1"]["raw_aux_share"]),
                              raw_aux_share_last2=span(lambda v: v["last2"]["raw_aux_share"]), aux_drop_1=span(lambda v: v["aux_drop_first_to_last1"]), ce_drop_1=span(lambda v: v["ce_drop_first_to_last1"]),
                              locality_drop_1=[min(v["locality_drop_first_to_last1"] for v in loc), max(v["locality_drop_first_to_last1"] for v in loc)],
                              weighted_locality_share_last1=[min(v["last1"]["weighted_locality_share"] for v in loc), max(v["last1"]["weighted_locality_share"] for v in loc)], locality_runs=len(loc))

    # B. balanced-layout SFT on the crossed development views
    dev_cross = {u["id"]: u for u in rows(PKG / "data/v16-role-balance-20261106/development.jsonl")}
    bal = {}
    for seed in SEEDS:
        bsum = read_json(EV / ("runs/v19-budget-layout-20261108/balanced-compressed-sft-seed%d/summary.json" % seed))
        assert bsum["status"] == "success" and bsum["optimizer_steps"] == 1024 and not bsum["reserved_test_accessed"]
        for step in (512, 1024):
            cur = rows(V19 / ("balanced_layout/balanced-compressed-sft-seed%d-step%d/crossed/predictions.jsonl" % (seed, step)))
            base = rows(EV / ("results/v16-code-fidelity-20261106/roles/main-compressed-sft-seed%d-step%d/predictions.jsonl" % (seed, step)))
            o = {}
            for name, raw in (("balanced", cur), ("original_sft", base)):
                assert len(raw) == 2304
                sr = slot_rule_rows(raw, dev_cross)
                assert sum(x["correct"] for x in sr) == sum(em(r["prediction"], r["gold"]) for r in raw)
                o[name] = dict(correct=sum(x["correct"] for x in sr), identity=sum(x["correct"] for x in sr if x["identity"]), identity_n=sum(x["identity"] for x in sr),
                               transformed=sum(x["correct"] for x in sr if not x["identity"]), transformed_n=sum(not x["identity"] for x in sr), substitutions=sum(x["substitution"] for x in sr),
                               follows=sum(x["follows"] for x in sr), slots=slot_detail(raw, dev_cross))
            ev, sft = bsum["evaluations"][str(step)], endpoint["sft-%d-%d" % (seed, step)]
            o["original_dev"] = dict(rows=ev["probe"]["counts"]["known_exact"][0], T=ev["probe"]["counts"]["target_joint"][0], U=ev["probe"]["counts"]["locality_joint"][0], f1=ev["natural"]["known_f1"],
                                     sft_rows=sft["rows"], sft_T=sft["T"], sft_U=sft["U"], sft_f1=sft["f1"])
            o["crossed_gain_pp"] = 100 * (o["balanced"]["correct"] - o["original_sft"]["correct"]) / 2304
            o["transformed_gain_pp"] = 100 * (o["balanced"]["transformed"] - o["original_sft"]["transformed"]) / o["balanced"]["transformed_n"]
            o["original_change_pp"] = 100 * (o["original_dev"]["rows"] - sft["rows"]) / 1536
            o["f1_change_pp"] = 100 * (o["original_dev"]["f1"] - sft["f1"])
            bal["%d-%d" % (seed, step)] = o
    res["balanced"] = bal
    for r in read_csv(V19 / "final_analysis/balanced_layout_comparison.csv"):
        o = bal["%s-%s" % (r["seed"], r["step"])]
        assert o["balanced"]["correct"] == int(r["crossed_correct"]) and abs(o["crossed_gain_pp"] - float(r["crossed_gain_pp"])) < 1e-9
        assert abs(o["original_change_pp"] - float(r["original_controlled_change_pp"])) < 1e-9 and abs(o["f1_change_pp"] - float(r["natural_f1_change_pp"])) < 1e-9

    # E. slot rule on the crossed development views, by source (post hoc cohort)
    dev_rule = {}
    for seed in SEEDS:
        for step in (512, 1024):
            sr = slot_rule_rows(rows(PKG / ("followup/followups/roles/main-compressed-sft-seed%d-step%d/predictions.jsonl" % (seed, step))), dev_cross)
            clusters = collections.defaultdict(list)
            for x in sr:
                clusters[x["source"]].append(x)
            by = [[sum(x["follows"] for x in v), sum(x["substitution"] for x in v)] for v in clusters.values()]
            dev_rule["%d-%d" % (seed, step)] = dict(
                all_contexts=[sum(x["follows"] for x in sr), sum(x["substitution"] for x in sr)], original_context=[sum(x["follows"] for x in sr if x["context"] == 0), sum(x["substitution"] for x in sr if x["context"] == 0)],
                per_source=sorted(by), per_source_rate_range=[min(a / b for a, b in by), max(a / b for a, b in by)], correct=sum(x["correct"] for x in sr),
                rule_cell=[sum(x["correct"] for x in sr if x["requested"] == x["rule"]), sum(x["requested"] == x["rule"] for x in sr)],
                other_cell=[sum(x["correct"] for x in sr if x["requested"] != x["rule"]), sum(x["requested"] != x["rule"] for x in sr)])
    res["dev_slot_rule"] = dev_rule

    # C. prospective replication, original-schema units
    spec = importlib.util.spec_from_file_location("v19_analysis", str(PKG / "followup_v19/src/binding/v19_analysis.py"))
    v19 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v19)
    new_units = {u["id"]: u for u in rows(EV / "data/v19-prospective-20261108/original.jsonl")}
    new_cross = {u["id"]: u for u in rows(EV / "data/v19-prospective-20261108/crossed.jsonl")}
    assert len(new_units) == 96 and len(new_cross) == 144 and len({u["source_id"] for u in new_cross.values()}) == 8
    pro = {}
    for arm in ("compressed", "full"):
        for seed in SEEDS:
            label = "main-%s-sft-seed%d-step512" % (arm, seed)
            raw = rows(V19 / ("prospective/%s/original/predictions.jsonl" % label))
            receipt = read_json(V19 / ("prospective/%s/original/receipt.json" % label))
            assert len(raw) == 1536
            pm = probe_metrics(raw)
            scene = collections.defaultdict(lambda: dict(target_n=0, choice_correct=0, greedy_correct=0, absent_n=0, digit_advantage_sum=0.0, fail_n=0, fail_choice=0))
            o = collections.Counter()
            gold_m, other_m, near, nested = [], [], 0, True
            by_level = collections.Counter()
            by = {}
            for r in raw:
                u = new_units[r["unit_id"]]
                i, j = r["context_index"], r["question_index"]
                v, gold = r["prediction"].strip(), u["answers"][i][j]
                assert gold == r["gold"]
                by[u["id"], i, j] = v == gold
                z = scene[u["source_id"]]
                values = [f["value"] for f in u["facts"][i]]
                if j < 2:
                    sc = [sum(s["token_logp"]) / len(s["token_logp"]) for s in r["candidate_scores"]]
                    decision = None if sc[0] == sc[1] else int(sc[1] > sc[0])
                    assert decision == r["choice"] and r["candidate_values"][j] == gold and (decision == j) == r["choice_correct"]
                    z["target_n"] += 1
                    z["choice_correct"] += decision == j
                    z["greedy_correct"] += v == gold
                    o["ties"] += decision is None
                    nested = nested and not (v == gold and decision != j)
                    if v != gold:
                        z["fail_n"] += 1
                        z["fail_choice"] += decision == j
                        o["target_foil"] += v == r["candidate_values"][1 - j]
                if v != gold:
                    o["errors"] += 1
                    o["absent" if re.fullmatch("[0-9]{4}", v) and v not in values else "other_record_value" if v in values else "malformed"] += 1
                if re.fullmatch("[0-9]{4}", v) and v not in values:
                    d = hamming(v, gold)
                    others = [hamming(v, x) for x in values if x != gold]
                    z["absent_n"] += 1
                    z["digit_advantage_sum"] += sum(others) / len(others) - d
                    gold_m.append(4 - d)
                    other_m.append(4 - sum(others) / len(others))
                    near += d <= 2
            for uid in new_units:
                by_level[new_units[uid]["level"]] += all(by[uid, i, j] for i in (0, 1) for j in (0, 1))
            agg = [dict(source_id=k, **{x: v[x] for x in ("target_n", "choice_correct", "greedy_correct", "absent_n", "digit_advantage_sum")}) for k, v in sorted(scene.items())]
            primary = v19.primary_contrasts(agg)
            stored = receipt["primary_contrasts"]
            for key in ("choice_generation_gap", "absent_digit_advantage"):
                a, b = primary[key], stored[key]
                assert a["valid_resamples"] == b["valid_resamples"] and a["positive_lower_bound"] == b["positive_lower_bound"]
                for x, y in zip([a["estimate"]] + a["interval_98_75"], [b["estimate"]] + b["interval_98_75"]):
                    assert (x is None and y is None) or abs(x - y) < 1e-9, (label, key, x, y)
            tot = lambda k: sum(v[k] for v in scene.values())
            d = dict(rows_correct=pm["correct"], T=pm["T"], U=pm["U"], R=pm["R"], irrelevant=pm["irrelevant"], target_n=tot("target_n"), choice_correct=tot("choice_correct"), greedy_correct=tot("greedy_correct"),
                     absent_n=tot("absent_n"), errors=o["errors"], error_absent=o["absent"], error_other_record_value=o["other_record_value"], error_malformed=o["malformed"], target_foil=o["target_foil"],
                     ties=o["ties"], nested=nested, primary=primary, T_by_level={str(k): by_level[k] for k in sorted(by_level)})
            if tot("absent_n"):
                d.update(gold_matches_mean=sum(gold_m) / len(gold_m), other_matches_mean=sum(other_m) / len(other_m), within_two_wrong=near)
            if tot("fail_n"):
                d["choice_on_generation_failures"] = [tot("fail_choice"), tot("fail_n")]
                d["choice_on_failures_interval"] = cluster_boot(dict(scene), lambda cs: (sum(c["fail_choice"] for c in cs) / sum(c["fail_n"] for c in cs)) if sum(c["fail_n"] for c in cs) else None)
            tr = {}
            for j in (2, 3):
                for i in (1, 2, 3):
                    lost = sum(by[u, 0, j] and not by[u, i, j] for u in new_units)
                    rec = sum(by[u, i, j] and not by[u, 0, j] for u in new_units)
                    tr["q%d-c%d" % (j, i)] = [lost, rec]
            d["transitions"] = tr
            d["rows_by_query"] = {"q%d" % j: sum(by[u, i, j] for u in new_units for i in range(4)) for j in range(4)}
            pro["%s-%d" % (arm, seed)] = d
    res["prospective"] = pro

    # D. prospective replication, crossed views of the first eight new scenes
    cross = {}
    for arm in ("compressed", "full"):
        for seed in SEEDS:
            label = "main-%s-sft-seed%d-step512" % (arm, seed)
            raw = rows(V19 / ("prospective/%s/crossed/predictions.jsonl" % label))
            assert len(raw) == 2304
            sr = slot_rule_rows(raw, new_cross)
            clusters = collections.defaultdict(list)
            for x in sr:
                clusters[x["source"]].append(x)
            by = [[sum(x["follows"] for x in v), sum(x["substitution"] for x in v)] for v in clusters.values()]
            o = dict(correct=sum(x["correct"] for x in sr), identity=[sum(x["correct"] for x in sr if x["identity"]), sum(x["identity"] for x in sr)],
                     transformed=[sum(x["correct"] for x in sr if not x["identity"]), sum(not x["identity"] for x in sr)], substitutions_all=[sum(x["follows"] for x in sr), sum(x["substitution"] for x in sr)],
                     substitutions_original_context=[sum(x["follows"] for x in sr if x["context"] == 0), sum(x["substitution"] for x in sr if x["context"] == 0)], per_scene=sorted(by),
                     rule_cell=[sum(x["correct"] for x in sr if x["requested"] == x["rule"]), sum(x["requested"] == x["rule"] for x in sr)],
                     other_cell=[sum(x["correct"] for x in sr if x["requested"] != x["rule"]), sum(x["requested"] != x["rule"] for x in sr)])
            if o["substitutions_all"][1]:
                o["per_scene_rate_range"] = [min(a / b for a, b in by if b), max(a / b for a, b in by if b)]
                o["rate_interval"] = cluster_boot(clusters, rule_rate)
                o["slots"] = slot_detail(raw, new_cross)
            cross["%s-%d" % (arm, seed)] = o
    res["prospective_crossed"] = cross
    stored = collections.Counter()
    for r in read_csv(V19 / "final_analysis/prospective_slot_rule_intrusions.csv"):
        stored[r["job"], "n"] += 1
        stored[r["job"], "follows"] += r["follows_rule"] == "True"
    for arm in ("compressed", "full"):
        for seed in SEEDS:
            label = "main-%s-sft-seed%d-step512" % (arm, seed)
            assert cross["%s-%d" % (arm, seed)]["substitutions_all"] == [stored[label, "follows"], stored[label, "n"]], label

    # I. the all-four count of the first target badge on confirmation versus the target-joint count
    t7 = {}
    for seed in SEEDS:
        groups = collections.defaultdict(dict)
        for r in rows(PKG / ("confirmation/main-compressed-sft-seed%d/predictions.jsonl" % seed)):
            if r["family"] == "probe":
                groups[r["unit_id"]][r["context_index"], r["question_index"]] = em(r["prediction"], r["gold"])
        T = {u for u, c in groups.items() if all(c[i, j] for i in (0, 1) for j in (0, 1))}
        q0 = {u for u, c in groups.items() if all(c[i, 0] for i in range(4))}
        t7[str(seed)] = dict(T=len(T), q0_all_four=len(q0), shared=len(T & q0), locker_rows=sum(c[i, 2] for c in groups.values() for i in range(4)), rows_per_query=4 * len(groups),
                             T_four_teams=sum(1 for u in T if u.endswith("-probe-4")), units_four_teams=sum(1 for u in groups if u.endswith("-probe-4")))
    res["table7_vs_T"] = t7
    res["margins"] = margins(PKG)
    return res


def compute_extra(PKG, EV):
    """Error composition on crossed views for original-layout SFT, balanced-layout SFT and the new scenes."""
    V19 = EV / "results/v19-budget-layout-20261108"
    dev = {u["id"]: u for u in rows(PKG / "data/v16-role-balance-20261106/development.jsonl")}
    new = {u["id"]: u for u in rows(EV / "data/v19-prospective-20261108/crossed.jsonl")}

    def composition(raw, units):
        out = {}
        for name, keep in (("identity", identity), ("transformed", lambda u: not identity(u))):
            c = collections.Counter()
            gm, om = [], []
            for r in raw:
                u = units[r["unit_id"]]
                if not keep(u):
                    continue
                i, j = r["context_index"], r["question_index"]
                v, gold = r["prediction"].strip(), u["answers"][i][j]
                values = [f["value"] for f in u["facts"][i]]
                _, fields = row_line(u, i, j)
                c["n"] += 1
                if v == gold:
                    c["correct"] += 1
                elif v in [code for _, code in fields]:
                    c["same_row_substitution"] += 1
                elif v in values:
                    c["other_record_value"] += 1
                elif re.fullmatch("[0-9]{4}", v):
                    c["absent"] += 1
                    others = [hamming(v, x) for x in values if x != gold]
                    gm.append(4 - hamming(v, gold))
                    om.append(4 - sum(others) / len(others))
                else:
                    c["malformed"] += 1
            d = dict(c)
            if gm:
                d["absent_gold_matches"] = sum(gm) / len(gm)
                d["absent_other_matches"] = sum(om) / len(om)
            out[name] = d
        return out

    res = {}
    for seed in SEEDS:
        for step in (512, 1024):
            res["sft-%d-%d" % (seed, step)] = composition(rows(EV / ("results/v16-code-fidelity-20261106/roles/main-compressed-sft-seed%d-step%d/predictions.jsonl" % (seed, step))), dev)
            res["balanced-%d-%d" % (seed, step)] = composition(rows(V19 / ("balanced_layout/balanced-compressed-sft-seed%d-step%d/crossed/predictions.jsonl" % (seed, step))), dev)
        res["new-compressed-%d" % seed] = composition(rows(V19 / ("prospective/main-compressed-sft-seed%d-step512/crossed/predictions.jsonl" % seed)), new)
    return res


def open_evidence(path):
    path = pathlib.Path(path)
    return zipfile.Path(zipfile.ZipFile(path)) if path.is_file() else path


if __name__ == "__main__":
    PKG, EV = pathlib.Path(sys.argv[1]), open_evidence(sys.argv[2])
    a, b = compute(PKG, EV), compute_extra(PKG, EV)
    pathlib.Path(sys.argv[3]).write_text(json.dumps(a, indent=1) + "\n", encoding="utf8")
    pathlib.Path(sys.argv[4]).write_text(json.dumps(b, indent=1) + "\n", encoding="utf8")
    print(json.dumps(dict(status="written", prospective_seed73=[a["prospective"]["compressed-73"][k] for k in ("choice_correct", "greedy_correct", "absent_n", "T")],
                          new_scene_rule=[a["prospective_crossed"]["compressed-%d" % s]["substitutions_all"] for s in SEEDS], balanced_1024=[a["balanced"]["%d-1024" % s]["balanced"]["transformed"] for s in SEEDS])))
