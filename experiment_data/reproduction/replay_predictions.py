"""Re-score recorded generations; no model inference or access to held-out partitions."""
import csv,json,hashlib,sys,math
from collections import defaultdict
from pathlib import Path
P=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(P/'code/src'))
from binding.v13_protocol import metrics
from binding.v11_pipeline import em_f1
from binding.v13_error_profile import profile
def read(p): return json.loads(p.read_text(encoding='utf8'))
units=[json.loads(x) for x in (P/'data/v13-answerable-relations-20261026/development.jsonl').read_text(encoding='utf8').splitlines()]
unit={u['id']:u for u in units}
raw=P/'evidence/recorded'
cells=[]; profiles={}; n=0
for path in sorted(raw.glob('runs/*/main-*/evaluation-*/predictions.jsonl')):
    folder=path.parent.parent; run=folder.name; version=folder.parent.name; step=path.parent.name.split('-')[1]
    summary=read(P/'evidence/retrieved/runs'/version/run/'summary.json')
    e=summary['evaluations'][step]; assert hashlib.sha256(path.read_bytes()).hexdigest()==e['predictions_sha256']
    rows=[json.loads(x) for x in path.read_text(encoding='utf8').splitlines()]; assert len(rows)==1760
    assert {(r['unit_id'],r['context_index'],r['question_index']) for r in rows}=={(u['id'],i,j) for u in units for i,answers in enumerate(u['answers']) for j,a in enumerate(answers) if a is not None}
    for r in rows:
        u=unit[r['unit_id']]; assert r['gold']==u['answers'][r['context_index']][r['question_index']]
        assert (r['source_id'],r['article'],r['family'])==(u['source_id'],u['article'],u['family'])
    for fam in ('natural','probe'):
        m=metrics([r for r in rows if r['family']==fam]); assert m['counts']==e[fam]['counts'] and math.isclose(m['known_f1'],e[fam]['known_f1'],abs_tol=1e-12)
    n+=len(rows)
    if step!='512': continue
    pr=profile(rows,units); seed=summary['seed']; method=summary['method']; arm=summary['arm']
    expect=(P/'evidence/retrieved/results'/('v14-bottleneck-reading-20261026/'+run+'-profile.json' if method=='bottleneck' else 'v13-binding-error-analysis-20261026/'+run+'--512.json'))
    ref=read(expect); assert all(pr[k]==ref[k] for k in ('errors','by_context','by_level','by_query','source_scores'))
    profiles[seed,arm,method]=pr['source_scores']; groups=defaultdict(dict)
    for r in rows:
        if r['family']=='probe': groups[r['unit_id']][r['context_index'],r['question_index']]=int(em_f1(r['prediction'],r['gold'])[0])
    def count(inds,qs): return sum(all(v[i,j] for i in inds for j in qs) for v in groups.values())
    t=count((0,1),(0,1)); u=count((0,1),(2,3)); u8=count(range(4),(2,3))
    assert [t,96]==e['probe']['counts']['target_joint'] and [u8,96]==e['probe']['counts']['locality_joint']
    cells.append(dict(seed=seed,arm=arm,method=method,target_four=t,unaffected_four=u,unaffected_eight=u8,units=len(groups)))
assert n==52800 and len(cells)==10
sensitivity=[]
for seed in (73,97):
    b=profiles[seed,'compressed','sft']
    for method in ('assignment','relation','bottleneck'):
        c=profiles[seed,'compressed',method]; assert set(b)==set(c) and len(b)==32
        delta={s:round((c[s]['target_joint']-b[s]['target_joint'])*3) for s in b}
        arts={v['article'] for v in b.values()}; assert len(arts)==24
        loo=[]
        for a in arts:
            ds=[v for s,v in delta.items() if b[s]['article']!=a]; loo.append(100*sum(ds)/(3*len(ds)))
        sensitivity.append(dict(seed=seed,method=method,better=sum(x>0 for x in delta.values()),unchanged=sum(x==0 for x in delta.values()),worse=sum(x<0 for x in delta.values()),delta_pp=100*sum(delta.values())/96,omit_one_article_min_pp=min(loo),omit_one_article_max_pp=max(loo)))
def output_csv(name,rows):
    with (P/'tables'/name).open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
output_csv('matched_four_answer_groups.csv',cells)
output_csv('article_deletion_sensitivity.csv',sensitivity)
print(json.dumps({'files':30,'development_predictions':n,'endpoint_profiles_verified':10,'article_deletions_per_comparison':24,'model_inference_performed':False}))

