"""Independently recount field intrusions from displayed development records."""
from pathlib import Path
from collections import Counter
import argparse,csv,hashlib,json,re
a=argparse.ArgumentParser();a.add_argument('--package',type=Path);a.add_argument('--output',type=Path);args=a.parse_args()
P=args.package or Path(__file__).resolve().parents[1]
O=args.output or P/'field_analysis';O.mkdir(parents=True,exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def readrows(p):return [json.loads(s) for s in p.read_text(encoding='utf8').splitlines()]
data=P/'data/v16-role-balance-20261106/development.jsonl'
units={u['id']:u for u in readrows(data)};assert len(units)==144
pins=[dict(path=data.relative_to(P).as_posix(),sha256=sha(data))]
summary=[];exceptions=[]
expected={(512,73):(166,166,646,651),(512,97):(163,163,617,622),(1024,73):(163,166,638,650),(1024,97):(157,159,634,643)}
for step in (512,1024):
 for seed in (73,97):
  f=P/f'followup/followups/roles/main-compressed-sft-seed{seed}-step{step}/predictions.jsonl'
  pins.append(dict(path=f.relative_to(P).as_posix(),sha256=sha(f)))
  receipt=json.loads((f.parent/'receipt.json').read_text());assert receipt['predictions_sha256']==sha(f)
  counts=Counter();rows=readrows(f);assert len(rows)==2304
  for r in rows:
   u=units[r['unit_id']];i=r['context_index'];j=r['question_index'];team,mode,field=u['query_keys'][j]
   lines=[s for s in u['contexts'][i].splitlines() if s.startswith(f'Team {team}, {mode} mode: ')]
   assert len(lines)==1
   fields=re.findall(r'\b(badge|locker|route) (\d{4})\b',lines[0]);assert len(fields)==3
   assert dict(fields)[field]==r['gold']==u['answers'][i][j]
   pred=r['prediction'].strip()
   if pred==r['gold']:continue
   slots=[k+1 for k,(name,value) in enumerate(fields) if pred==value]
   if not slots:continue
   assert len(slots)==1
   rule=2 if field=='locker' else 1;ok=slots[0]==rule
   counts[i,'n']+=1;counts[i,'ok']+=ok
   if not ok:exceptions.append(dict(seed=seed,step=step,unit_id=u['id'],context=i,query=j,field=field,predicted_slot=slots[0],rule_slot=rule,prediction=pred,gold=r['gold']))
  original=(counts[0,'ok'],counts[0,'n']);total=(sum(counts[i,'ok'] for i in range(4)),sum(counts[i,'n'] for i in range(4)))
  assert original+total==expected[step,seed]
  for scope,ok,n in [('original',*original),('all_four_contexts',*total)]:
   summary.append(dict(cohort='development',seed=seed,step=step,context_scope=scope,follows_rule=ok,same_row_field_intrusions=n,exceptions=n-ok))
  old=list(csv.DictReader((P/'field_analysis/slot_rule_summary.csv').open(encoding='utf8')))
  for i in range(4):
   prior=next(x for x in old if x['arm']=='compressed' and int(x['seed'])==seed and int(x['step'])==step and int(x['context'])==i)
   assert int(prior['follows_rule'])==counts[i,'ok'] and int(prior['same_row_field_intrusions'])==counts[i,'n']
for name,rows in [('slot_rule_scope_summary.csv',summary),('slot_rule_exceptions.csv',exceptions)]:
 with (O/name).open('w',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
result=dict(status='pass',prediction_rows=9216,cohort='development',post_hoc=True,scope_rows=summary,input_files=pins)
(O/'scope_analysis_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
print(json.dumps(result))
