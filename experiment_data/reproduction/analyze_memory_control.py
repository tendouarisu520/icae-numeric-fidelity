"""CPU audit of matched versus independently assigned donor memories."""
from pathlib import Path
import csv,hashlib,json,math
R=Path(__file__).resolve().parents[1]
P=R
O=R/'followup';F=O/'followups/memory_control'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def rows(p):return [json.loads(x) for x in p.read_text(encoding='utf8').splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
protocol=read(F/'protocol.json');plan=protocol['plan'];data=P/'data/v13-answerable-relations-20261026/development.jsonl'
assert not protocol['reserved_test_accessed'] and sha(data)==plan['source_sha256']
units={u['id']:u for u in rows(data)};pairs={p['unit_id']:p for p in plan['pairs']}
assert len(pairs)==len({units[k]['article'] for k in pairs})==8
assert set(pairs)=={p['donor_unit_id'] for p in pairs.values()}
for uid,p in pairs.items():
 u=units[uid];d=units[p['donor_unit_id']]
 assert u['article']!=d['article'] and u['level']==d['level']==4
 assert p['source_id']==u['source_id'] and p['donor_source_id']==d['source_id']
 assert not {a for row in u['answers'] for a in row[:2]}&{f['value'] for facts in d['facts'] for f in facts}
summaries=[];by_source=[]
for job in protocol['jobs']:
 folder=F/job['name'];receipt=read(folder/'receipt.json');raw=rows(folder/'scores.jsonl')
 assert receipt['protocol_sha256']==sha(F/'protocol.json') and receipt['scores_sha256']==sha(folder/'scores.jsonl')
 assert receipt['exact_greedy_replay_rows']==2 and receipt['rows']==len(raw)==128
 expected={(uid,i,j,m) for uid in pairs for i in range(4) for j in (0,1) for m in ('matched','shuffled')}
 assert {(r['unit_id'],r['context_index'],r['question_index'],r['mode']) for r in raw}==expected
 prior={(r['unit_id'],r['context_index'],r['question_index']):r for r in rows(O/'choice/development'/job['name']/'scores.jsonl')}
 delta=0.0
 for r in raw:
  uid,i,j,mode=r['unit_id'],r['context_index'],r['question_index'],r['mode'];u=units[uid];old=prior[uid,i,j]
  assert r['gold']==u['answers'][i][j] and r['candidate_values']==u['answers'][i][:2]
  assert r['article']==u['article'] and r['source_id']==u['source_id']
  assert r['memory_unit_id']==(uid if mode=='matched' else pairs[uid]['donor_unit_id'])
  for s,t in zip(r['candidate_scores'],old['candidate_scores']):
   assert s['token_ids']==t['token_ids'] and len(s['token_ids'])==len(s['token_logp'])==6
   assert all(math.isfinite(v) and v<=1e-6 for v in s['token_logp'])
   assert abs(sum(s['token_logp'])-s['sum'])<1e-4 and abs(sum(s['token_logp'])/6-s['mean'])<2e-5
   if mode=='matched':delta=max(delta,abs(s['mean']-t['mean']))
  a,b=[s['mean'] for s in r['candidate_scores']];choice=None if a==b else int(b>a)
  assert r['choice']==choice and r['choice_correct']==(choice==j)
  if mode=='matched':assert r['choice']==old['choice'] and r['choice_correct']==old['choice_correct']
 assert delta<=1e-5 and abs(delta-receipt['max_matched_score_difference'])<1e-12
 for mode in ('matched','shuffled'):
  subset=[r for r in raw if r['mode']==mode];correct=sum(r['choice_correct'] for r in subset)
  assert correct==receipt[mode+'_correct'] and len(subset)==64
  summaries.append(dict(reader=job['name'],seed=job['seed'],mode=mode,rows=64,correct=correct,accuracy=correct/64,ties=sum(r['choice'] is None for r in subset)))
  for uid in pairs:
   ss=[r for r in subset if r['unit_id']==uid]
   by_source.append(dict(reader=job['name'],seed=job['seed'],unit_id=uid,article=units[uid]['article'],mode=mode,rows=len(ss),correct=sum(r['choice_correct'] for r in ss)))
assert len(summaries)==4
for name,data in [('memory_control_summary.csv',summaries),('memory_control_by_source.csv',by_source)]:
 with (O/name).open('w',newline='',encoding='utf8') as f:
  writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
audit=dict(status='verified',readers=2,rows=256,source_articles=8,protocol_sha256=sha(F/'protocol.json'),reserved_test_accessed=False)
(O/'memory_control_cpu_audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(**audit,summary=summaries)))
