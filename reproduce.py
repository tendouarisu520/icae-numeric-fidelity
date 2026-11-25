"""Verify the supplementary data and optionally rerun the full CPU analysis."""
import argparse
import csv
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "experiment_data"
STEPS = (128, 256, 512, 640, 768, 896, 1024)
ANALYSES = (
    "verify_results", "replay_predictions", "score_confirmation", "audit_learning_and_queries",
    "analyze_codes", "analyze_choice", "analyze_positions", "analyze_choice_objectives",
    "analyze_teacher_prefix", "analyze_exposure", "analyze_roles", "analyze_near",
    "analyze_followup_contrasts", "analyze_memory_control", "analyze_field_positions",
    "check_slot_rule", "verify_followup_v19", "verify_followup_v19_raw", "verify_seed_extension",
)

def check(ok, message):
    if not ok:
        raise ValueError(message)

def integrity():
    seen = set()
    for line in (ROOT / "SHA256SUMS.txt").read_text(encoding="utf8").splitlines():
        digest, name = line.split("  ", 1)
        path = ROOT / name
        check(path.is_file() and ROOT in path.resolve().parents and name not in seen, name)
        check(hashlib.sha256(path.read_bytes()).hexdigest() == digest, "Digest mismatch in " + name)
        seen.add(name)
    print(f"{len(seen)} file digests verified.", flush=True)

def count_rows(paths, files, expected, description):
    paths = sorted(paths)
    check(len(paths) == files, description + " file count")
    count = 0
    for path in paths:
        with path.open("rb") as fh:
            count += sum(1 for line in fh if line.strip())
    check(count == expected, f"{description}: {count} != {expected}")
    print(f"{description}: {count:,} rows in {len(paths)} files.", flush=True)

def inventory():
    f = DATA / "followup"
    count_rows((DATA/"evidence/recorded/runs").glob("*/*/evaluation-*/predictions.jsonl"),30,52800,"Original development")
    count_rows((DATA/"confirmation").glob("main-*/predictions.jsonl"),10,35200,"Original confirmation")
    count_rows((f/"choice").glob("*/*/scores.jsonl"),20,23040,"Original candidate scoring")
    count_rows((f/"followups/roles").glob("*/predictions.jsonl"),6,13824,"Factorial field views")
    count_rows((f/"followups/near").glob("*/*/scores.jsonl"),10,10752,"Digit competitors")
    count_rows((f/"followups/memory_control").glob("*/scores.jsonl"),2,256,"Memory controls")
    for seed in (73,97):
        d = f/f"followups/exposure{seed}"
        count_rows((d/f"evaluation-{k}/predictions.jsonl" for k in STEPS),7,7*1760,f"Repeated exposure, seed {seed}")
    for tag in ("v19","v20"):
        with ZipFile(DATA/f"followup_{tag}/evidence/completed_evidence.zip") as z:
            check(z.testzip() is None, tag+" archive checksum")
            names=set(z.namelist())
            if tag=="v20":
                logs=[n for n in names if n.endswith("/steps.jsonl") and "/replay32/" not in n]
                check(len(logs)==6, "Seed-extension training logs")
                for n in logs:
                    check(len(z.read(n).splitlines())==512,n)
            else:
                for seed in (73,97):
                    for step in (512,1024):
                        name=f"results/v19-budget-layout-20261108/balanced_layout/balanced-compressed-sft-seed{seed}-step{step}/crossed/predictions.jsonl"
                        check(len(z.read(name).splitlines())==2304, name)
    romans = "I II III IV V VI VII VIII IX X XI XII XIII XIV XV XVI XVII XVIII XIX XX XXI XXII XXIII".split()
    check({p.name for p in (ROOT/"tables/current_manuscript").glob("*.csv")} ==
          {f"Table_{n}.csv" for n in romans}, "Manuscript table inventory")
    with (ROOT/"analysis/v8_failed_target_rows.csv").open(encoding="utf8",newline="") as f:
        rows=list(csv.DictReader(f))
    check(len(rows)==313, "New-scene failed generations")
    for seed,pair in (("73",(17,21)),("97",(16,20))):
        matched=[r for r in rows if r["seed"]==seed and r["distance_group"]=="closer_to_foil"]
        check((sum(int(r["likelihood_selects_gold"]) for r in matched),len(matched))==pair,"Section VI-C, seed "+seed)

def run(script, env):
    result=subprocess.run([sys.executable,"-B","-X","utf8",str(script)],cwd=script.parent,
                          env=env,capture_output=True,encoding="utf8")
    if result.returncode:
        raise RuntimeError(f"{script.name} failed\n{result.stdout[-1200:]}\n{result.stderr[-2400:]}")
    print(f"Passed {script.name}",flush=True)

def full(work_parent):
    import collections
    if work_parent is not None:
        work_parent = work_parent.resolve()
        work_parent.mkdir(parents=True,exist_ok=True)
        check(work_parent != ROOT and ROOT not in work_parent.parents, "Choose a temporary directory outside the extracted supplement.")
    with tempfile.TemporaryDirectory(prefix="nf_",dir=work_parent) as scratch_name:
        scratch=Path(scratch_name)
        working=scratch/"experiment_data"
        shutil.copytree(DATA,working)
        env={**os.environ,"PYTHONHASHSEED":"0","TMP":str(scratch),"TEMP":str(scratch),"TMPDIR":str(scratch)}
        for i,name in enumerate(ANALYSES,1):
            print(f"[{i}/{len(ANALYSES)}] {name}",flush=True)
            run(working/"reproduction"/(name+".py"),env)
        # Compare exported cells as rows. Ignore row order and platform-specific line endings.
        for name in ("field_analysis/field_by_slot_accuracy.csv","field_analysis/unaffected_bidirectional_transitions.csv",
                     "field_analysis/digit_competition_512.csv","field_analysis/slot_rule_scope_summary.csv",
                     "tables/learning_curve_counts.csv","tables/query_context_accuracy.csv"):
            def cells(path):
                with path.open(encoding="utf-8-sig",newline="") as f:
                    return collections.Counter(tuple(row) for row in csv.reader(f))
            check(cells(working/name)==cells(DATA/name), "Recomputed export differs: "+name)
    print("All 19 extended analyses passed; six regenerated CSV tables match the supplied cells.",flush=True)

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full",action="store_true",help="Run all additional CPU analyses in a temporary copy")
    parser.add_argument("--work-dir",type=Path,help="Parent directory for the temporary working copy")
    args=parser.parse_args()
    integrity()
    inventory()
    run(ROOT/"analysis/verify_v8.py",os.environ.copy())
    run(ROOT/"analysis/verify_figure_data.py",os.environ.copy())
    if args.full:
        full(args.work_dir)

