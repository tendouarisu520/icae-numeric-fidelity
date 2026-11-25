"""Reproduce digit-position controls and overlap with measured candidate decisions."""
from pathlib import Path
import csv, hashlib, json
R=Path(__file__).resolve().parents[1]
P=R
O=R/'followup'
def lines(p):return [json.loads(x) for x in p.read_text(encoding='utf8').splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,rs):
    with (O/name).open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
data=list(csv.DictReader((O/'absent_code_rows.csv').open(encoding='utf8',newline='')))
units={'development':{u['id']:u for u in lines(P/'data/v13-answerable-relations-20261026/development.jsonl')},
       'confirmation':{u['id']:u for u in lines(P/'confirmation/confirmation_units.jsonl')}}
positions=[];overlaps=[]
for split in ('development','confirmation'):
    for seed in (73,97):
        selected=[r for r in data if r['split']==split and r['arm']=='compressed' and r['method']=='sft' and int(r['seed'])==seed]
        for q in ('all','0','1','2','3'):
            part=[r for r in selected if q=='all' or r['question_index']==q];n=len(part)
            result=dict(split=split,seed=seed,query=q,absent=n)
            for j in range(4):
                result[f'gold_pos{j+1}']=sum(r['prediction'][j]==r['gold'][j] for r in part)/n
                control=[]
                for r in part:
                    facts=units[split][r['unit_id']]['facts'][int(r['context_index'])]
                    other=[f['value'] for f in facts if f['value']!=r['gold']]
                    control.append(sum(r['prediction'][j]==v[j] for v in other)/len(other))
                result[f'other_pos{j+1}']=sum(control)/n
            positions.append(result)
        path=O/f'choice/{split}/main-compressed-sft-seed{seed}/scores.jsonl'
        receipt=json.loads((path.parent/'receipt.json').read_text());assert sha(path)==receipt['scores_sha256']
        keys={(r['unit_id'],int(r['context_index']),int(r['question_index'])) for r in selected}
        relevant=[r for r in lines(path) if (r['unit_id'],r['context_index'],r['question_index']) in keys]
        overlaps.append(dict(split=split,seed=seed,absent_target=len(relevant),choice_correct=sum(r['choice_correct'] for r in relevant)))
save('digit_position_controls.csv',positions)
(O/'absent_choice_overlap.json').write_text(json.dumps(overlaps,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(position_rows=len(positions),overlaps=overlaps)))
