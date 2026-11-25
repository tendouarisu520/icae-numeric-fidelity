"""Recompute the v8 review analyses from the archived predictions (standard library).
Run from any directory: python analysis/verify_v8.py
Use --write to regenerate the two v8 exports. The historical evidence is read-only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

BASE = Path(__file__).resolve().parent.parent
EVIDENCE = BASE / "experiment_data"
OLD_SEEDS = (73, 97)
NEW_SEEDS = (101, 131, 151)
SEEDS = OLD_SEEDS + NEW_SEEDS
STEPS = (128, 256, 512, 640, 768, 896, 1024)
HASHES = {"v19": "3ce637ac01246020ad6c0c7a986c2c47fc1a256665be8fe5d422dc2b7d43e956",
          "v20": "89da11b97d4229748423be1a98c02d72d1bdbd5593fe73c5e7d696cf7c28efb6"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def rows(data):
    return [json.loads(line) for line in data.splitlines() if line.strip()]


def member(archive, suffix):
    matches = [n for n in archive.namelist() if n.endswith(suffix)]
    require(len(matches) == 1, f"Expected one archive member ending in {suffix!r}: {matches}")
    return archive.read(matches[0])


def units(data):
    return {u["id"]: u for u in rows(data) if u["family"] == "probe"}


def scores_and_choice(row):
    candidates = row["candidate_values"]
    entries = row["candidate_scores"]
    require(len(entries) == len(candidates) == 2 and candidates[0] != candidates[1], "Candidate pair")
    require(all(len(e["token_ids"]) == len(e["token_logp"]) == 6 for e in entries), "Token count")
    raw = [statistics.fmean(e["token_logp"]) for e in entries]
    saved = [e["mean"] for e in entries]
    require(all(abs(a - b) < 1e-5 for a, b in zip(raw, saved)), "Score mismatch")
    decision = None if raw[0] == raw[1] else int(raw[1] > raw[0])
    require(decision == row["choice"], "Choice mismatch")
    require(row["choice_correct"] == (decision is not None and candidates[decision] == row["gold"]), "Correctness mismatch")
    return raw, decision


def shown_teams(unit, context):
    found = re.findall(r"^Team ([^,]+),", unit["contexts"][context], flags=re.M)
    teams = list(dict.fromkeys(found))
    require(len(teams) == len(unit["entities"]), "Displayed team order")
    return teams


def team_analysis(archive):
    result = {}
    for cohort, datafile in (("development_crossed", "data/v16-role-balance-20261106/development.jsonl"),
                             ("new_scene_crossed", "data/v19-prospective-20261108/crossed.jsonl")):
        data = units(archive.read(datafile))
        result[cohort] = {}
        for seed in SEEDS:
            lines = rows(member(archive, f"main-compressed-sft-seed{seed}-step512/{cohort}/rows.jsonl"))
            groups = defaultdict(lambda: [0, 0])
            for row in lines:
                j, c = row["question_index"], row["context_index"]
                u = data[row["unit_id"]]
                exchange = u["target_team_position"]
                if j >= 2 or not any((exchange, u["layout_rotation"], u["semantic_rotation"])):
                    continue
                _, choice = scores_and_choice(row)
                target, _, _ = u["query_keys"][j]
                pos = shown_teams(u, c).index(target) + 1
                expected = ((3 if exchange else 4) if c == 3 else (2 if exchange else 1))
                require(pos == expected and row["candidate_values"] == u["answers"][c][:2], "Position or label mismatch")
                for group in ("all", f"assignment_{exchange}", f"position_{pos}", f"ctx_{c}"):
                    groups[group][0] += choice == j
                    groups[group][1] += 1
            require(groups["all"][1] == 1088 and groups["ctx_3"][1] == 272, "Transformed row count")
            result[cohort][str(seed)] = dict(groups)
    train = units((EVIDENCE / "data/v13-answerable-relations-20261026/train.jsonl").read_bytes())
    counter = Counter()
    for u in train.values():
        q_team, q_mode, q_field = u["query_keys"][1]
        require(q_team == u["entities"][0] and q_mode == u["modes"][1] and q_field == "badge", "Training query role")
        for c in range(4):
            if shown_teams(u, c).index(q_team) == 1:
                counter[(u["level"], c)] += 1
    require(list(counter) == [(2, 3)] and sum(counter.values()) == 144, "Two-team reversal exposure")
    result["training_second_mode_badge_on_physically_second_team"] = sum(counter.values())
    return result


def locker_analysis(archive):
    data = units((EVIDENCE / "data/v16-role-balance-20261106/development.jsonl").read_bytes())
    out = {}
    for seed in OLD_SEEDS:
        for training in ("original", "balanced"):
            tail = (f"/roles/main-compressed-sft-seed{seed}-step1024/predictions.jsonl" if training == "original"
                    else f"/balanced_layout/balanced-compressed-sft-seed{seed}-step1024/crossed/predictions.jsonl")
            lines = rows(member(archive, tail))
            require(len(lines) == 2304, "Crossed answer count")
            counts, scenes = Counter(), set()
            for r in lines:
                u = data[r["unit_id"]]
                c, j = r["context_index"], r["question_index"]
                team, mode, field = u["query_keys"][j]
                prefix = f"Team {team}, {mode} mode:"
                text = next(line for line in u["contexts"][c].splitlines() if line.startswith(prefix))
                fields = re.findall(r"\b(badge|locker|route) ([0-9]{4})\b", text)
                gold = u["answers"][c][j]
                require(len(fields) == 3 and dict(fields)[field] == r["gold"] == gold, "Field mapping")
                prediction = r["prediction"].strip()
                values = [value for _, value in fields]
                if prediction == gold or prediction not in values:
                    continue
                gold_pos, predicted_pos = values.index(gold) + 1, values.index(prediction) + 1
                rule_pos = 2 if field == "locker" else 1
                counts["same_row_substitutions"] += 1
                counts["old_rule_matches"] += predicted_pos == rule_pos
                if field == "locker" and gold_pos != 2:
                    counts["eligible_locker_substitutions"] += 1
                    counts["eligible_locker_from_position_2"] += predicted_pos == 2
                    scenes.add(u["source_id"])
            out[f"{training}_{seed}"] = dict(counts, eligible_locker_scenes=len(scenes))
    return out


def distance(a, b):
    require(re.fullmatch(r"[0-9]{4}", a) and re.fullmatch(r"[0-9]{4}", b), "Expected four digits")
    return sum(x != y for x, y in zip(a, b))


def distance_analysis(archive):
    data = units(archive.read("data/v19-prospective-20261108/original.jsonl"))
    results, failure_rows = {}, []
    for seed in OLD_SEEDS:
        lines = rows(member(archive, f"/prospective/main-compressed-sft-seed{seed}-step512/original/predictions.jsonl"))
        counts, closer_scenes = Counter(), set()
        example = None
        for r in lines:
            c, j = r["context_index"], r["question_index"]
            if j >= 2:
                continue
            u = data[r["unit_id"]]
            v = r["prediction"].strip()
            pair = r["candidate_values"]
            require(pair == u["answers"][c][:2] and pair[j] == r["gold"], "Truth and foil identities")
            gold, foil = pair[j], pair[1-j]
            scores, decision = scores_and_choice(r)
            correct_choice = decision == j
            counts["target_rows"] += 1
            counts["choices_correct"] += correct_choice
            counts["generation_correct"] += v == gold
            if v == gold:
                continue
            counts["generation_failed"] += 1
            counts["failed_choice_correct"] += correct_choice
            dg, df = distance(v, gold), distance(v, foil)
            group = "closer_to_foil" if df < dg else "equidistant" if df == dg else "closer_to_gold"
            counts[group] += 1
            counts[f"{group}_choice_correct"] += correct_choice
            if group == "closer_to_foil":
                closer_scenes.add(r["source_id"])
            rec = dict(seed=seed, scene_id=r["source_id"], unit_id=r["unit_id"], teams=u["level"],
                       context=c, query=j, emitted=v, gold=gold, foil=foil,
                       distance_to_gold=dg, distance_to_foil=df, distance_group=group,
                       likelihood_selects_gold=int(correct_choice), gold_mean_logp=scores[j], foil_mean_logp=scores[1-j])
            failure_rows.append(rec)
            if r["unit_id"] == "v19-prospective-20261108-004-probe-2" and c == j == 0:
                example = rec
        require(example and (example["gold"], example["foil"], example["emitted"]) == ("6505", "6955", "6950"), "Illustrative row")
        require(example["likelihood_selects_gold"] and example["distance_to_gold"] == 3 and example["distance_to_foil"] == 1, "Illustrative decision")
        results[str(seed)] = dict(counts, distinct_scenes_closer_to_foil=len(closer_scenes), example=example)
    return results, failure_rows


def margin_analysis():
    result = {}
    for cohort in ("development", "confirmation"):
        for seed in OLD_SEEDS:
            for arm in ("sft", "assignment", "relation", "bottleneck"):
                f = EVIDENCE / "followup/choice" / cohort / f"main-compressed-{arm}-seed{seed}" / "scores.jsonl"
                matrices = defaultdict(dict)
                for row in rows(f.read_bytes()):
                    j = row["question_index"]
                    means, _ = scores_and_choice(row)
                    m = matrices[row["unit_id"], row["context_index"]]
                    require(j in (0,1) and j not in m and row["candidate_values"][j] == row["gold"], "Two distinct mode queries")
                    m[j] = means, row["candidate_values"]
                averaged, smooth = [], []
                for matrix in matrices.values():
                    (a,b), pair0 = matrix[0]
                    (c,d), pair1 = matrix[1]
                    require(pair0 == pair1, "Same candidate columns across questions")
                    gaps = (a-b, d-c, a-c, d-b)
                    avg = ((a-b)+(d-c))/2
                    lower = min(gaps)
                    soft = lower - 0.5*math.log(math.fsum(math.exp(-(g-lower)/0.5) for g in gaps))
                    averaged.append(avg)
                    smooth.append(soft)
                require(len(averaged) == (384 if cohort == "development" else 768), "Number of score matrices")
                result[f"{cohort}_{arm}_{seed}"] = dict(contexts=len(averaged), median_average_margin=statistics.median(averaged),
                      average_above_offset=sum(x > 0.5 for x in averaged), median_smooth_minimum=statistics.median(smooth),
                      smooth_above_offset=sum(x > 0.5 for x in smooth))
    return result


def target_groups(data):
    groups = defaultdict(dict)
    for r in rows(data):
        i, j = r["context_index"], r["question_index"]
        if r["family"] == "probe" and i in (0,1) and j in (0,1):
            g = groups[r["unit_id"]]
            require((i,j) not in g, "Duplicate target row")
            g[i,j] = r["gold"] == r["prediction"].strip()
    require(len(groups) == 96 and all(len(x)==4 for x in groups.values()), "96 four-answer groups")
    return sum(all(x.values()) for x in groups.values())


def seed_analysis(v19, v20):
    runs = {}
    coverage = {}
    endpoint1024 = defaultdict(dict)
    for seed in OLD_SEEDS:
        for arm in ("sft", "assignment", "relation", "bottleneck"):
            prefix = f"/main-compressed-sft-seed{seed}/" if arm == "sft" else f"/extended-compressed-{arm}-seed{seed}/"
            if arm == "sft":
                suffix = f"runs/v16-repeated-exposure-20261106{prefix}"
            else:
                suffix = f"runs/v19-budget-layout-20261108{prefix}"
            summary = json.loads(v19.read(suffix+"summary.json"))
            raw_steps = [s for s in STEPS if suffix+f"evaluation-{s}/predictions.jsonl" in v19.namelist()]
            require(raw_steps == [512,1024] and sorted(map(int,summary["evaluations"])) == list(STEPS), "Archive-level checkpoint coverage")
            if arm != "sft":
                coverage[f"{arm}_{seed}"] = dict(v19_prediction_steps=raw_steps,
                    summary_steps=list(STEPS), new_intermediate_summary_steps=[640,768,896])
            for step in raw_steps:
                t = target_groups(v19.read(suffix+f"evaluation-{step}/predictions.jsonl"))
                require(summary["evaluations"][str(step)]["probe"]["counts"]["target_joint"] == [t,96], "Independent endpoint")
                if step == 1024:
                    endpoint1024[arm][str(seed)]=t
                elif arm in ("sft","bottleneck"):
                    runs[seed,arm]=t
    for seed in NEW_SEEDS:
        for arm in ("sft","bottleneck"):
            runs[seed,arm]=target_groups(member(v20,f"/main-compressed-{arm}-seed{seed}/evaluation-512/predictions.jsonl"))
    counts={str(s): {"sft":runs[s,"sft"],"weakest":runs[s,"bottleneck"]} for s in SEEDS}
    def interval(seeds, critical):
        ds=[100*(runs[s,"bottleneck"]-runs[s,"sft"])/96 for s in seeds]
        mean, sd=statistics.mean(ds),statistics.stdev(ds)
        half=critical*sd/math.sqrt(len(ds))
        return dict(mean_pp=mean,sd_pp=sd,ci95=[mean-half,mean+half],degrees_of_freedom=len(ds)-1)
    return dict(endpoint_512_groups=counts, endpoint_1024_groups=dict(endpoint1024),
                five_seeds=interval(SEEDS,2.7764451051977987), three_new_seeds=interval(NEW_SEEDS,4.302652729696142),
                minimum_gain_groups=math.ceil(0.05*96), minimum_attainable_gain_pp=100*math.ceil(0.05*96)/96),coverage


def compute():
    paths = {k:EVIDENCE/f"followup_{k}/evidence/completed_evidence.zip" for k in HASHES}
    for tag,p in paths.items():
        require(sha(p.read_bytes()) == HASHES[tag], f"{tag} archive identity")
    with ZipFile(paths["v19"]) as v19, ZipFile(paths["v20"]) as v20:
        require(v19.testzip() is None and v20.testzip() is None, "Evidence checksum")
        closer, detailed = distance_analysis(v19)
        seed_result,coverage = seed_analysis(v19,v20)
        report = dict(protocol="Post hoc v8 manuscript audit; the frozen v19 and v20 protocols are in experiment_data.",
                      source_archives_sha256=HASHES, new_scene_failed_generation=closer,
                      team_assignment_and_displayed_position=team_analysis(v20), locker_residual=locker_analysis(v19),
                      held_out_margins=margin_analysis(), fixed_development_seed_contrasts=seed_result, auxiliary_archive_coverage=coverage)
    return report,detailed


def exports():
    report, detail = compute()
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=list(detail[0]), lineterminator="\n")
    writer.writeheader(); writer.writerows(detail)
    return {"v8_verified_numbers.json":(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n").encode(),
            "v8_failed_target_rows.csv":buf.getvalue().encode()}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write",action="store_true",help="Regenerate the two v8 exports")
    opts=parser.parse_args()
    for name,data in exports().items():
        path=Path(__file__).parent/name
        if opts.write:
            path.write_bytes(data)
        else:
            require(path.read_bytes()==data,"Mismatch in "+name)
        print(f"{sha(data)}  {name}")
