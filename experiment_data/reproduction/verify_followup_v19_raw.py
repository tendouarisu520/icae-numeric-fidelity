"""Recompute the V19 follow-up results from the raw outputs and compare them with the stored numbers. Standard library only."""
from pathlib import Path
import hashlib,json,sys,zipfile
P=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(P/'reproduction'))
import v19_evidence_analysis as A
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
Z=P/'followup_v19/evidence/completed_evidence.zip'
receipt=json.loads((P/'followup_v19/evidence/archive_identity.json').read_text(encoding='utf8'))
assert sha(Z)==receipt['sha256']=='3ce637ac01246020ad6c0c7a986c2c47fc1a256665be8fe5d422dc2b7d43e956' and Z.stat().st_size==receipt['bytes']==2952444
checks=[]
with zipfile.ZipFile(Z) as z:
 assert z.testzip() is None;names=z.namelist();assert len(names)==receipt['entries']==84
 assert not any('heldout_test' in n or 'paired-v2-20261022/test' in n for n in names)
 lock=json.loads(z.read('temp/v19/lock.json'))
 assert hashlib.sha256(z.read('temp/v19/lock.json')).hexdigest()==sha(P/'followup_v19/temp/v19/lock.json')
 for name in ('original','crossed'):
  data=z.read(f'data/v19-prospective-20261108/{name}.jsonl')
  assert hashlib.sha256(data).hexdigest()==lock['generated_data_sha256'][name]==sha(P/f'followup_v19/data/v19-prospective-20261108/{name}.jsonl')
 report=json.loads(z.read('results/v19-budget-layout-20261108/final_analysis/final_report.json'))
 assert report==json.loads((P/'followup_v19/results/v19-budget-layout-20261108/final_status_20261112/final_report.json').read_text(encoding='utf8'))
 assert report['status']=='verified' and report['test_accessed'] is False
 predictions=[n for n in names if n.endswith('predictions.jsonl')];lines=sum(len(z.read(n).splitlines()) for n in predictions)
checks.append(f'Evidence archive matches its recorded SHA-256, the frozen lock and the earlier verified report; {len(predictions)} prediction files with {lines} rows; no reserved test file')
EV=A.open_evidence(Z)
new,extra=A.compute(P,EV),A.compute_extra(P,EV)
count=[0]
def same(x,y,path=''):
 if isinstance(x,dict):
  assert isinstance(y,dict) and set(x)==set(y),path
  for k in x:same(x[k],y[k],path+'/'+k)
 elif isinstance(x,list):
  assert isinstance(y,list) and len(x)==len(y),path
  for i,(a,b) in enumerate(zip(x,y)):same(a,b,f'{path}/{i}')
 elif isinstance(x,float) or isinstance(y,float):
  assert abs(x-y)<1e-9,(path,x,y);count[0]+=1
 else:
  assert x==y,(path,x,y);count[0]+=1
same(json.loads(json.dumps(new)),json.loads((P/'followup_v19/analysis_v5/v19_numbers.json').read_text(encoding='utf8')))
same(json.loads(json.dumps(extra)),json.loads((P/'followup_v19/analysis_v5/v19_numbers_extra.json').read_text(encoding='utf8')))
checks.append(f'{count[0]} stored values recomputed from the raw outputs: endpoints and seven checkpoints of the extended arms, loss components, balanced-layout SFT, slot rule on development and new scenes, prospective replication, error composition')
p=new['prospective'];c=new['prospective_crossed']
assert all(p[f'compressed-{s}']['primary']['both_positive'] and p[f'compressed-{s}']['nested'] for s in (73,97)) and all(p[f'full-{s}']['rows_correct']==1536 and c[f'full-{s}']['correct']==2304 for s in (73,97))
assert not new['steps_with_five_points_at_both_seeds'] and new['largest_single_seed_gain_groups']==3
checks.append('Primary contrasts positive at both seeds; full-context readers exact on original and crossed new-scene views; no auxiliary arm reaches five points at both seeds at any checkpoint')
result=dict(status='pass',checks=checks,values_compared=count[0],evidence_sha256=sha(Z))
(P/'reproduction/followup_v19_raw_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))
