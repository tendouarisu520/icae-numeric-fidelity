"""Tabulate field positions, digit competitors, intervention transitions and the chronology of the weakest-margin objective."""
from pathlib import Path
from collections import Counter,defaultdict
import csv,hashlib,json,re
R=Path(__file__).resolve().parents[1]
P=R
O=R/'field_analysis';O.mkdir(parents=True,exist_ok=True)
def read(p):return json.loads(p.read_text(encoding='utf8'))
def rows(p):return [json.loads(s) for s in p.read_text(encoding='utf8').splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,data):
 with (O/name).open('w',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
pins=[]
def pin(p):pins.append(dict(path=p.relative_to(P).as_posix(),sha256=sha(p)))
path=P/'data/v16-role-balance-20261106/development.jsonl';pin(path);units={u['id']:u for u in rows(path)}
protocol=P/'followup/followups/roles/protocol.json';pin(protocol);cfg=read(protocol)
assert sha(path)==cfg['data_sha256'] and len(units)==144
matrix=[];intrusions=[];slot_summary=[]
# This descriptive rule was motivated by the inspected error records, not preregistered.
slot_rule={'badge':0,'route':0,'locker':1}
for job in cfg['jobs']:
 folder=P/'followup/followups/roles'/job['label'];receipt=read(folder/'receipt.json');path=folder/'predictions.jsonl';pin(path)
 assert sha(path)==receipt['predictions_sha256'] and sha(protocol)==receipt['protocol_sha256']
 raw=rows(path);assert len(raw)==2304 and len({(r['unit_id'],r['context_index'],r['question_index']) for r in raw})==2304
 counts=defaultdict(lambda:[0,0]);checks=defaultdict(lambda:[0,0]);meta={k:job[k] for k in ('arm','seed','step')}
 for r in raw:
  u=units[r['unit_id']];i=r['context_index'];j=r['question_index'];q=u['query_keys'][j];p=r['prediction'].strip()
  assert r['gold']==u['answers'][i][j]
  # Parse the displayed team/mode line, independently of fact-list order.
  lines=[line for line in u['contexts'][i].splitlines() if line.startswith(f'Team {q[0]}, {q[1]} mode: ')]
  assert len(lines)==1
  displayed=re.findall(r'\b(badge|locker|route) (\d{4})\b',lines[0]);assert len(displayed)==3
  fields=[f for f,v in displayed];assert fields==u['rendered_field_order']
  facts={f['key'][2]:f['value'] for f in u['facts'][i] if f['key'][:2]==q[:2]}
  assert dict(displayed)==facts and facts[q[2]]==r['gold']
  if i==0:
   z=counts[j,q[2],fields.index(q[2])];z[0]+=int(p==r['gold']);z[1]+=1
  matched=[(position,field) for position,(field,value) in enumerate(displayed) if value==p]
  if matched and p!=r['gold']:
   assert len(matched)==1
   position,field=matched[0];follows=position==slot_rule[q[2]]
   checks[i][0]+=1;checks[i][1]+=int(follows)
   intrusions.append(dict(**meta,unit_id=u['id'],source_id=u['source_id'],article=u['article'],context=i,query=j,query_field=q[2],requested_slot=fields.index(q[2])+1,predicted_field=field,predicted_slot=position+1,rule_slot=slot_rule[q[2]]+1,prediction=p,gold=r['gold'],follows_rule=follows))
 for (j,f,s),(correct,n) in sorted(counts.items()):
  assert n==16
  matrix.append(dict(**meta,query=j,field=f,slot=s+1,correct=correct,n=n))
 for i,(n,following) in sorted(checks.items()):slot_summary.append(dict(**meta,context=i,same_row_field_intrusions=n,follows_rule=following))
for seed,expected in [(73,{'badge':[13,0,0],'locker':[1,2,0],'route':[12,0,0]}),(97,{'badge':[14,0,0],'locker':[0,2,0],'route':[9,0,0]})]:
 for field,values in expected.items():
  actual=[next(x['correct'] for x in matrix if x['arm']=='compressed' and x['seed']==seed and x['step']==512 and x['query']==2 and x['field']==field and x['slot']==slot) for slot in (1,2,3)]
  assert actual==values
 z=next(x for x in slot_summary if x['arm']=='compressed' and x['seed']==seed and x['step']==512 and x['context']==0)
 assert z['same_row_field_intrusions']==z['follows_rule']==(166 if seed==73 else 163)
save('field_by_slot_accuracy.csv',matrix);save('same_row_field_intrusions.csv',intrusions);save('slot_rule_summary.csv',slot_summary)

# Recompute BOTH transition directions on the original confirmation from raw rows.
transitions=[]
for seed in (73,97):
 path=P/f'confirmation/main-compressed-sft-seed{seed}/predictions.jsonl';pin(path)
 raw=[r for r in rows(path) if r['family']=='probe'];lookup={(r['unit_id'],r['context_index'],r['question_index']):r for r in raw}
 for q,label in [(2,'Locker'),(3,'Other-team badge')]:
  for i,name in [(1,'Swap'),(2,'Irrelevant edit'),(3,'Reverse order')]:
   c=Counter()
   for uid in {r['unit_id'] for r in raw}:
    a,b=lookup[uid,0,q],lookup[uid,i,q];assert a['gold']==b['gold']
    c[a['prediction'].strip()==a['gold'],b['prediction'].strip()==b['gold']]+=1
   n=sum(c.values());assert n==192
   transitions.append(dict(seed=seed,query=q,label=label,intervention=name,n=n,original_correct=c[True,True]+c[True,False],original_wrong=c[False,True]+c[False,False],correct_to_wrong=c[True,False],wrong_to_correct=c[False,True],net_correct_change=c[False,True]-c[True,False]))
for seed,expect in [(73,[-12,0,-52]),(97,[-3,1,-37])]:assert [r['net_correct_change'] for r in transitions if r['seed']==seed and r['query']==2]==expect
save('unaffected_bidirectional_transitions.csv',transitions)

# Score digit competitors at the 512-update endpoint.
choices=[]
for split in ('confirmation','development'):
 for seed in (73,97):
  path=P/f'followup/followups/near/{split}/main-compressed-sft-seed{seed}-step512/scores.jsonl';pin(path)
  receipt=read(path.parent/'receipt.json');assert sha(path)==receipt['scores_sha256']
  errors=[r for r in rows(path) if not r['greedy_correct'] and r['gold_score']['tokens']==r['generated_score']['tokens']]
  n=len(errors);wins=[sum(r['gold_score']['mean']>other(r)['mean'] for r in errors) for other in (lambda r:r['condition_foil_score'],lambda r:r['near']['0']['score'],lambda r:r['near']['3']['score'],lambda r:r['generated_score'])]
  choices.append(dict(split=split,seed=seed,step=512,n=n,condition=wins[0],first_digit=wins[1],last_digit=wins[2],generated=wins[3],condition_pct=100*wins[0]/n,first_digit_pct=100*wins[1]/n,last_digit_pct=100*wins[2]/n,generated_pct=100*wins[3]/n))
for seed,expect in [(73,[161,'88.82','97.52','89.44','14.29']),(97,[153,'87.58','96.73','88.24','3.92'])]:
 r=next(x for x in choices if x['split']=='development' and x['seed']==seed)
 assert [r['n']]+[f'{r[k]:.2f}' for k in ('condition_pct','first_digit_pct','last_digit_pct','generated_pct')]==expect
save('digit_competition_512.csv',choices)
history=O/'protocol';plan=read(history/'v14-run-plan-20261026.json');proposal=history/'2026-10-26-bottleneck-conditional-followup.md'
assert sha(proposal)==plan['source_files']['docs/superpowers/plans/2026-10-26-bottleneck-conditional-followup.md']
assert '128步 target-joint 为33/96' in proposal.read_text(encoding='utf8') and '这些中途结果用于提出修改' in proposal.read_text(encoding='utf8')
launch=read(history/'v14-launch-audit-20261026.json')['files']
queue=launch['runs/v14-bottleneck-reading-20261026/queue-status.json']['data']
assert queue['phase']=='waiting_for_original_eight' and queue['scientific_starts']==0 and queue['original_step']==368
chronology=dict(designed_after_interim_development=True,original_eight_incomplete=True,original_job=queue['original_job'],original_step=queue['original_step'],execution_waited_for_eight=True,source_files=[dict(path=p.name,sha256=sha(p)) for p in sorted(history.iterdir()) if p.is_file()])
(O/'chronology_audit.json').write_text(json.dumps(chronology,indent=2)+'\n',encoding='utf8')
audit=dict(status='verified',role_models=6,field_slot_cells=len(matrix),intrusion_rows=len(intrusions),transition_rows=len(transitions),table_v_rows=len(choices),posthoc_slot_rule=slot_rule,input_files=pins,chronology_verified=True,model_inference_performed=False)
(O/'field_position_audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(status='verified',table_v=choices,original_context_slot_rule=[r for r in slot_summary if r['arm']=='compressed' and r['step']==512 and r['context']==0],locker_transitions=[r for r in transitions if r['query']==2])))
