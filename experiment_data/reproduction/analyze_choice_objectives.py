"""Paired auxiliary-versus-SFT effects on measured readouts, not cross-budget effects."""
from pathlib import Path
from collections import defaultdict
import csv,hashlib,json,random
R=Path(__file__).resolve().parents[1];O=R/'followup'
def load(split,method,seed):
    p=O/f'choice/{split}/main-compressed-{method}-seed{seed}/scores.jsonl'
    receipt=json.loads((p.parent/'receipt.json').read_text())
    assert hashlib.sha256(p.read_bytes()).hexdigest()==receipt['scores_sha256']
    return {(r['unit_id'],r['context_index'],r['question_index']):r for r in map(json.loads,p.read_text().splitlines())}
def ci(items):
    groups=defaultdict(list)
    for article,value in items:groups[article].append(value)
    ns={a:len(v) for a,v in groups.items()};sums={a:sum(v) for a,v in groups.items()};keys=sorted(groups);rng=random.Random(20261106);sample=[]
    for _ in range(2000):
        draw=rng.choices(keys,k=len(keys));sample.append(sum(sums[a] for a in draw)/sum(ns[a] for a in draw))
    sample.sort();return 100*sum(v for _,v in items)/len(items),100*sample[49],100*sample[1949]
out=[]
for split in ('development','confirmation'):
    for seed in (73,97):
        baseline=load(split,'sft',seed)
        for method in ('assignment','relation','bottleneck'):
            other=load(split,method,seed);assert set(other)==set(baseline)
            for outcome in ('choice_correct','greedy_correct'):
                for joint in (False,True):
                    if not joint:items=[(b['article'],int(other[k][outcome])-int(b[outcome])) for k,b in baseline.items()]
                    else:
                        items=[]
                        for uid in sorted({k[0] for k in baseline}):
                            keys=[(uid,i,j) for i in (0,1) for j in (0,1)]
                            items.append((baseline[keys[0]]['article'],int(all(other[k][outcome] for k in keys))-int(all(baseline[k][outcome] for k in keys))))
                    effect,lo,hi=ci(items)
                    out.append(dict(split=split,seed=seed,method=method,outcome=outcome,unit='target_group' if joint else 'target_row',n=len(items),difference_pp=effect,cluster95_low_pp=lo,cluster95_high_pp=hi))
with (O/'choice_objective_contrasts.csv').open('w',newline='',encoding='utf8') as f:
    w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
print(json.dumps(dict(contrasts=len(out),post_hoc=True,fixed_step=512)))
