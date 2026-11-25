"""Offline CPU verification of included evidence and published tables; no model loading."""
from pathlib import Path
import hashlib,json,sys,csv,math
P=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(P/'code/src'))
from binding.v13_error_profile import paired_intervals
from binding.v13_protocol import metrics
A=json.loads((P/'evidence/v13-v14-completed-audit-20261027.json').read_text(encoding='utf8'))
V=json.loads((P/'evidence/v13-v14-results-reviewed-20261027.json').read_text(encoding='utf8'))
plan=A['files']['docs/v14-run-plan-20261026.json']['data'];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert all(A['verified_source_pins'].values()) and all(A['verified_data_pins'].values())
for name,h in plan['source_files'].items():assert sha(P/'code'/name)==h,name
verified=[]
for name,h in plan['data_files'].items():
    path=P/name
    if path.exists():assert sha(path)==h,name;verified.append(name)
assert len(verified)==4
train=[json.loads(x) for x in (P/'data/v13-answerable-relations-20261026/train.jsonl').read_text(encoding='utf8').splitlines()]
dev=[json.loads(x) for x in (P/'data/v13-answerable-relations-20261026/development.jsonl').read_text(encoding='utf8').splitlines()]
assert len(train)==1024 and len(dev)==128
assert not {x['article'] for x in train}&{x['article'] for x in dev}
assert len({x['source_id'] for x in train})==512 and len({x['source_id'] for x in dev})==32
assert all(a!='UNKNOWN' for u in train+dev for row in u['answers'] for a in row if a is not None)
assert not any('confirmation' in p.name.lower() or 'reserved_test' in p.name.lower() for p in (P/'data').rglob('*.jsonl'))
profiles={}
for seed in (73,97):
    for method in ('sft','assignment','relation','bottleneck'):
        job=f'main-compressed-{method}-seed{seed}'
        name=(f'results/v14-bottleneck-reading-20261026/{job}-profile.json' if method=='bottleneck' else f'results/v13-binding-error-analysis-20261026/{job}--512.json')
        profiles[seed,method]=A['files'][name]['data']['source_scores']
    for method in ('assignment','relation','bottleneck'):
        actual=paired_intervals(profiles[seed,'sft'],profiles[seed,method])
        expected=V['comparisons'][f'{method}-seed{seed}']['paired_intervals']
        for metric in ('target_joint','locality_joint','natural_f1'):
            x,y=actual['fields'][metric],expected['fields'][metric]
            assert math.isclose(x['delta'],y['delta'],abs_tol=1e-12)
            assert all(abs(a-b)<1e-12 for a,b in zip(x['interval95'],y['interval95']))
with (P/'tables/table2_endpoint_results.csv').open(encoding='utf8',newline='') as f:table=list(csv.DictReader(f))
assert len(table)==10
for actual,expected in zip(table,V['rows']):
    assert int(actual['seed'])==expected['seed'] and actual['method']==expected['method'] and actual['arm']==expected['arm']
    for key in ('natural_exact','probe_exact','target_joint','locality_joint','reorder_joint','candidate_misbinding'):
        assert [int(actual[key+'_correct']),int(actual[key+'_total'])]==expected[key]
    assert abs(float(actual['natural_f1_percent'])-100*expected['natural_f1'])<1e-12
assert not V['confirmation_evaluated'] and not V['reserved_test_accessed']
print(json.dumps({'status':'pass','pinned_source_files':len(plan['source_files']),'byte_identical_datasets':len(verified),'experiment_rows':10,'paired_bootstrap_comparisons_reproduced':6,'raw_prediction_replay_performed':False,'gpu_used':False},indent=2))
