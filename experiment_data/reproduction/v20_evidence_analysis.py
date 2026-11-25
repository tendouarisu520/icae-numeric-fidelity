# -*- coding: utf-8 -*-
"""Recompute the V20 results (seed extension, crossed-view choice scoring, new-seed readouts) from the evidence archive.

python -I -X utf8 -B v20_evidence_analysis.py <package_dir> [out.json]

The archive followup_v20/evidence/completed_evidence.zip keeps the host's relative paths. It is unpacked into a
temporary folder, the analysis module shipped inside it (temp/v20/analysis.py, pinned by temp/v20/lock.json) is run on
the unpacked rows, and the result is compared with the report that the host wrote after the queue
(results/v20-seeds-crossed-choice-20261113/analysis/report.json). Nothing is read from the network or the GPU host.
"""
import hashlib
import importlib.util
import json
import pathlib
import sys
import tempfile
import zipfile

V20 = "results/v20-seeds-crossed-choice-20261113"


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def same(a, b, path="", out=None):
    """Structural comparison; floats must agree to 1e-9 (identical code and RNG on both machines)."""
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(path + "/" + str(k) + " missing")
            else:
                same(a[k], b[k], path + "/" + str(k), out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(path + " length")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                same(x, y, path + "/%d" % i, out)
    elif isinstance(a, float) or isinstance(b, float):
        if a is None or b is None or abs(a - b) > 1e-9:
            out.append(path + " %r != %r" % (a, b))
    elif a != b:
        out.append(path + " %r != %r" % (a, b))
    return out


def posthoc(root):
    """Descriptive figures outside the frozen analysis: choice by mode question and by view on the crossed cohorts, absent outputs
    within two digits of the correct code at the three further seeds, and the 128- and 256-update results of the new runs."""
    import re
    import string
    from collections import Counter, defaultdict
    root = pathlib.Path(root)
    res = root / V20
    rows = lambda p: [json.loads(s) for s in pathlib.Path(p).read_text(encoding="utf8").splitlines() if s.strip()]
    paths = {"development_crossed": "data/v16-role-balance-20261106/development.jsonl", "new_scene_crossed": "data/v19-prospective-20261108/crossed.jsonl",
             "new_scene_original": "data/v19-prospective-20261108/original.jsonl", "development_original": "data/v13-answerable-relations-20261026/development.jsonl"}
    units = {k: {u["id"]: u for u in rows(root / p) if u["family"] == "probe"} for k, p in paths.items()}
    every, new = (73, 97, 101, 131, 151), (101, 131, 151)
    lab = lambda s: "main-compressed-sft-seed%d-step512" % s

    def norm(v):
        v = v.casefold().translate(str.maketrans("", "", string.punctuation))
        return " ".join(re.sub(r"\b(a|an|the)\b", " ", v).split())

    out = {}
    for cohort in ("development_crossed", "new_scene_crossed"):
        for seed in every:
            data = rows(res / ("crossed_choice" if seed in (73, 97) else "new_seeds") / lab(seed) / cohort / "rows.jsonl")
            ctx, chosen, c, cells = defaultdict(dict), defaultdict(dict), Counter(), defaultdict(Counter)
            for r in data:
                j = r["question_index"]
                if j > 1:
                    continue
                u = units[cohort][r["unit_id"]]
                view = (u["semantic_rotation"], u["layout_rotation"], u["target_team_position"])
                kind = "identity" if view == (0, 0, 0) else "transformed"
                key = (kind, u["id"], r["context_index"])
                means = [s["mean"] for s in r["candidate_scores"]]
                pick = None if means[0] == means[1] else int(means[1] > means[0])
                assert pick == r["choice"] and (pick == j) == r["choice_correct"]
                ctx[key][j] = pick == j
                chosen[key][j] = r["candidate_values"][pick]
                ctx[key]["slot"] = u["rendered_field_order"].index(u["query_keys"][j][2]) + 1
                ctx[key]["exchange"] = u["target_team_position"]
                if kind == "transformed":
                    c["q%d_n" % j] += 1
                    c["q%d_ok" % j] += pick == j
                cells[view]["n"] += 1
                cells[view]["ok"] += pick == j
                if view == (0, 0, 1):
                    cells["teams", j]["n"] += 1
                    cells["teams", j]["ok"] += pick == j
                    cells["teams", j]["gen_ok"] += norm(r["prediction"]) == norm(r["gold"])
                    cells["teams", j]["gen_other"] += r["prediction"].strip() == r["candidate_values"][1 - j]
            tr = {k: v for k, v in ctx.items() if k[0] == "transformed"}
            idv = [v for k, v in ctx.items() if k[0] == "identity"]
            same_code = lambda keys: sum(chosen[k][0] == chosen[k][1] for k in keys)
            s3 = [k for k, v in tr.items() if v["slot"] == 3]
            ex1 = [v for v in tr.values() if v["exchange"] == 1]
            first_second = [v for v in tr.values() if v["slot"] == 1 and v["exchange"] == 1]
            byu = defaultdict(dict)
            for (kind, uid, i), v in tr.items():
                if i in (0, 1):
                    byu[uid][i] = v[0] and v[1]
            first = [cells[k] for k in ((1, 1, 0), (2, 2, 0))]
            out["%s-%d" % (cohort, seed)] = dict(
                contexts=len(tr), q0=c["q0_ok"] / c["q0_n"], q1=c["q1_ok"] / c["q1_n"], both=sum(v[0] and v[1] for v in tr.values()) / len(tr),
                same_code=same_code(tr) / len(tr), identity_both=[sum(v[0] and v[1] for v in idv), len(idv)],
                choice_T=sum(all(x.values()) for x in byu.values()), units=len(byu),
                slot3_both=sum(tr[k][0] and tr[k][1] for k in s3) / len(s3), slot3_same=same_code(s3) / len(s3),
                exchange1_q0=sum(v[0] for v in ex1) / len(ex1), exchange1_q1=sum(v[1] for v in ex1) / len(ex1),
                exchange1_both=sum(v[0] and v[1] for v in ex1) / len(ex1),
                first_slot_first_team=[sum(x["ok"] for x in first), sum(x["n"] for x in first)], teams_only=[cells[0, 0, 1]["ok"], cells[0, 0, 1]["n"]],
                teams_only_q0=[cells["teams", 0]["ok"], cells["teams", 0]["n"]],
                teams_only_q1=[cells["teams", 1]["ok"], cells["teams", 1]["n"], cells["teams", 1]["gen_ok"], cells["teams", 1]["gen_other"]],
                first_slot_second_team=sum(v[0] + v[1] for v in first_second) / (2 * len(first_second)))
    for cohort in ("development_original", "new_scene_original"):
        for seed in new:
            absent = within = 0
            for r in rows(res / "new_seeds" / lab(seed) / cohort / "rows.jsonl"):
                v = r["prediction"].strip()
                values = [f["value"] for f in units[cohort][r["unit_id"]]["facts"][r["context_index"]]]
                if re.fullmatch("[0-9]{4}", v) and v not in values:
                    absent += 1
                    within += sum(a != b for a, b in zip(v, r["gold"])) <= 2
            out["within2-%s-%d" % (cohort, seed)] = [within, absent, within / absent]
    for seed in new:
        s = {m: json.loads((root / ("runs/v20-seed-extension-20261113/main-compressed-%s-seed%d/summary.json" % (m, seed))).read_text(encoding="utf8"))
             for m in ("sft", "bottleneck")}
        for step in ("128", "256", "512"):
            a, b = s["bottleneck"]["evaluations"][step], s["sft"]["evaluations"][step]
            out["ckpt-%d-%s" % (seed, step)] = dict(
                sft=b["probe"]["counts"]["target_joint"][0], wm=a["probe"]["counts"]["target_joint"][0],
                dT=100 * (a["probe"]["rates"]["target_joint"] - b["probe"]["rates"]["target_joint"]),
                dU=100 * (a["probe"]["rates"]["locality_joint"] - b["probe"]["rates"]["locality_joint"]),
                dF1=100 * (a["natural"]["known_f1"] - b["natural"]["known_f1"]))
    return out


def compute(package):
    package = pathlib.Path(package)
    archive = package / "followup_v20/evidence/completed_evidence.zip"
    receipt = json.loads((package / "followup_v20/evidence/evidence_receipt.json").read_text(encoding="utf8"))
    assert sha(archive) == receipt["sha256"] and archive.stat().st_size == receipt["bytes"], "archive differs from the host receipt"
    with tempfile.TemporaryDirectory(prefix="v20_") as tmp:
        root = pathlib.Path(tmp)
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            for name in z.namelist():
                assert not name.startswith(("/", "\\")) and ".." not in name.split("/"), name
            z.extractall(root)
        lock = json.loads((root / "temp/v20/lock.json").read_text(encoding="utf8"))
        assert sha(root / "temp/v20/lock.json") == receipt["lock_sha256"]
        checked = 0
        for name, digest in lock["files"].items():
            if (root / name).exists():
                assert sha(root / name) == digest, name
                checked += 1
        for key, rel in lock["data_paths"].items():
            assert sha(root / rel) == lock["data_sha256"][key], rel
        spec = importlib.util.spec_from_file_location("v20_analysis", root / "temp/v20/analysis.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        report = module.run(root)
        host = json.loads((root / V20 / "analysis/report.json").read_text(encoding="utf8"))
        differences = same(json.loads(json.dumps(report)), host)
        extra = dict(
            lock_sha256=receipt["lock_sha256"], archive_sha256=receipt["sha256"], archive_entries=receipt["entries"],
            locked_files_checked=checked, differences_from_host_report=differences, posthoc=posthoc(root),
            preflight=json.loads((root / V20 / "preflight.json").read_text(encoding="utf8")),
            adapters=json.loads((root / V20 / "analysis/adapters.json").read_text(encoding="utf8")),
            protocols={p.parent.name: json.loads(p.read_text(encoding="utf8")) for p in sorted((root / V20).glob("*/protocol.json"))})
    assert not differences, differences[:10]
    return report, extra


if __name__ == "__main__":
    result, info = compute(sys.argv[1])
    summary = dict(parts=sorted(k for k in result if k.startswith("part_") or k in ("replay", "new_scene_tests")),
                   inputs=len(result["inputs"]), locked_files_checked=info["locked_files_checked"],
                   host_report_reproduced=not info["differences_from_host_report"])
    if len(sys.argv) > 2:
        out = pathlib.Path(sys.argv[2])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(dict(report=result, record=info), indent=1, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf8")
    print(json.dumps(summary))
