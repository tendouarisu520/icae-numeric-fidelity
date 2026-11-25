"""Recover the inputs to Figure 1 and the plotted percentages in Figure 2."""
import json
from pathlib import Path
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "experiment_data"
COHORT = DATA / "followup_v19/data/v19-prospective-20261108"

def records(lines):
    return [json.loads(line) for line in lines if line.strip()]

def read_rows(path):
    with path.open("rb") as handle:
        return records(handle)

def figure_one():
    unit = next(r for r in read_rows(COHORT/"original.jsonl") if r["id"]=="v19-prospective-20261108-000-probe-2")
    view = next(r for r in read_rows(COHORT/"crossed.jsonl") if r["source_id"]==unit["source_id"]
                and (r["semantic_rotation"],r["layout_rotation"],r["target_team_position"])==(1,2,1))
    assert unit["answers"][0] == ["6111","8603","1019","5908"]
    teams = [[l for l in ctx.splitlines() if l.startswith("Team ")] for ctx in unit["contexts"]]
    assert teams[3] == list(reversed(teams[0])) and teams[0][2:] == teams[1][2:] and teams[0][:3] == teams[2][:3]
    shown = next(line for line in view["contexts"][0].splitlines() if line.startswith("Team Lyra, fast mode"))
    assert "locker 6111" in shown and view["answers"][0][0] == "6111"
    assert view["queries"][0] == "For team Lyra in fast mode, what is the locker code?"
    return unit["id"],view["id"]

def accuracy(label, rows, expected_n):
    assert len(rows)==expected_n
    exact = sum(r["prediction"].strip()==r["gold"] for r in rows)
    choice=0
    for r in rows:
        means=[sum(c["token_logp"])/len(c["token_logp"]) for c in r["candidate_scores"]]
        assert len(means)==2
        index = None if means[0]==means[1] else int(means[1]>means[0])
        assert index == r["choice"]
        correct = index is not None and r["candidate_values"][index] == r["gold"]
        assert correct == r["choice_correct"]
        choice += correct
    return [label,round(exact*100/len(rows),4),round(choice*100/len(rows),4)]

def figure_two():
    values=[]
    for cohort,label,count in [("development","Dev.",768),("confirmation","Conf.",1536)]:
        for seed in (73,97):
            rows=read_rows(DATA/f"followup/choice/{cohort}/main-compressed-sft-seed{seed}/scores.jsonl")
            values.append(accuracy(f"{label} {seed}",rows,count))
    with ZipFile(DATA/"followup_v19/evidence/completed_evidence.zip") as z:
        for seed in (73,97):
            k=f"results/v19-budget-layout-20261108/prospective/main-compressed-sft-seed{seed}-step512/original/predictions.jsonl"
            with z.open(k) as data:
                rows=[r for r in records(data) if r["question_index"]<2]
            values.append(accuracy(f"New {seed}",rows,768))
    expected=json.loads((HERE/"figure2_values.json").read_text(encoding="utf8"))
    assert values==expected,(values,expected)
    return values

if __name__=="__main__":
    print("Figure 1 unit and crossed view",*figure_one())
    print("Figure 2",json.dumps(figure_two()))

