"""Rebuild confirmation tables from the saved row-level generations."""
import csv
import hashlib
import json
from pathlib import Path
import sys

package=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(package/'code/src'))
from binding.v11_pipeline import em_f1
from binding.v13_protocol import metrics
from binding.v13_error_profile import profile,paired_intervals

def read(path):
    return json.loads(path.read_text(encoding='utf8'))
def lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf8').splitlines()]
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def save(name,rows):
    with (package/'tables'/name).open('w',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

root=package/'confirmation'
cfg=read(root/'protocol.json')
units=lines(root/'confirmation_units.jsonl')
source=lines(root/'confirmation.jsonl')
assert sha(root/'confirmation.jsonl')==cfg['pins']['data/v11-natural-frozen-20261025/confirmation.jsonl']
assert sha(root/'evaluate_confirmation.py')==cfg['runner']
assert len(source)==64 and len({u['article'] for u in source})==22
from binding.v13_training import build_units
derived=build_units(source,'development')
for u in derived:u['split']='confirmation'
assert units==json.loads(json.dumps(derived))
for split in ('train','development'):
    reference=lines(package/f'data/v11-natural-frozen-20261025/{split}.jsonl')
    for field in ('article','id'):
        assert {u[field] for u in source}.isdisjoint(u[field] for u in reference)
answers={(u['id'],i,j):(a,u['article'],u['source_id'],u['family']) for u in units for i,line in enumerate(u['answers']) for j,a in enumerate(line) if a is not None}
assert len(answers)==3520
profiles={}
endpoints=[]
contrasts=[]
errors=[]
for job in cfg['jobs']:
    folder=root/job['name']
    receipt=read(folder/'receipt.json')
    expected=read(folder/'metrics.json')
    path=folder/'predictions.jsonl'
    assert receipt['predictions']==expected['predictions_sha256']==sha(path)
    assert receipt['protocol']==sha(root/'protocol.json') and receipt['units']==sha(root/'confirmation_units.jsonl')
    assert receipt['checkpoint']==cfg['pins'][job['checkpoint']]
    rows=lines(path)
    assert len(rows)==3520 and {(r['unit_id'],r['context_index'],r['question_index']):(r['gold'],r['article'],r['source_id'],r['family']) for r in rows}==answers
    for fam in ('probe','natural'):
        actual=metrics([r for r in rows if r['family']==fam])
        assert actual==expected[fam]
    audit=profile(rows,units)
    profiles[job['arm'],job['method'],job['seed']]=audit['source_scores']
    e=audit['errors']; p=expected['probe']['counts']; natural=expected['natural']
    assert sum(e.values())==3072 and e['correct']==p['known_exact'][0]
    groups={}
    for r in rows:
        if r['family']=='probe':
            groups.setdefault(r['unit_id'],{})[r['context_index'],r['question_index']]=em_f1(r['prediction'],r['gold'])[0]
    matched=sum(all(v[i,j] for i in (0,1) for j in (2,3)) for v in groups.values())
    names={k:job[k] for k in ('arm','method','seed')}
    endpoints.append(dict(**names,natural_em=natural['counts']['known_exact'][0],natural_f1=natural['known_f1'],controlled_exact=e['correct'],target_joint=p['target_joint'][0],unaffected_joint=p['locality_joint'][0],unaffected_four=matched,reorder=p['reorder_target_joint'][0],wrong=3072-e['correct'],absent=e.get('unobserved_code',0),foil=p['candidate_misbinding'][0]))
    errors.append(dict(**names,**{k:e.get(k,0) for k in ('correct','unobserved_code','condition','entity','field','mixed_binding','format_with_code','non_code')}))
for seed in (73,97):
    baseline=profiles['compressed','sft',seed]
    for arm,method in (('full','sft'),('compressed','assignment'),('compressed','relation'),('compressed','bottleneck')):
        res=paired_intervals(baseline,profiles[arm,method,seed])
        for field,stats in res['fields'].items():
            contrasts.append(dict(seed=seed,arm=arm,method=method,metric=field,delta_pp=100*stats['delta'],lo_pp=100*stats['interval95'][0],hi_pp=100*stats['interval95'][1]))
save('confirmation_endpoints.csv',endpoints)
save('confirmation_errors.csv',errors)
save('confirmation_paired.csv',contrasts)
print(json.dumps(dict(verified_models=len(endpoints),verified_predictions=len(endpoints)*3520,sources=64,articles=22)))

