"""CPU verification and within-query factorial summaries for fixed role controls."""
from pathlib import Path
from collections import defaultdict,Counter
import csv,hashlib,json,math,sys
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'code/src'))
from binding.v13_protocol import metrics
from binding.v13_error_profile import classify
from binding.v11_pipeline import normalize,em_f1
from binding.v16_code_fidelity import compare_codes
O=R/'followup';F=O/'followups/roles';P=R
def read(p):return json.loads(p.read_text(encoding='utf8'))
def rows(p):return [json.loads(x) for x in p.read_text(encoding='utf8').splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,records):
    with (O/name).open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
cfg=read(F/'protocol.json');data=R/'data/v16-role-balance-20261106/development.jsonl';assert sha(data)==cfg['data_sha256']
training=rows(P/'data/v13-answerable-relations-20261026/train.jsonl')
query_fields=Counter((j,k[2]) for u in training if u['family']=='probe' for j,k in enumerate(u['query_keys']))
assert query_fields=={(0,'badge'):512,(1,'badge'):512,(2,'locker'):512,(3,'badge'):512}
(O/'role_training_distribution.json').write_text(json.dumps(dict(query_keys_per_training_source=4,counts=[dict(query=q,field=f,count=n) for (q,f),n in sorted(query_fields.items())],
    interpretation='The route field occurs in records but is never queried in controlled training. Semantic rotations therefore include an unqueried-field transfer condition; report fields separately. Only first-two-team roles are crossed.'),indent=2)+'\n',encoding='utf8')
units={u['id']:u for u in rows(data)};assert len(units)==144 and len({u['source_id'] for u in units.values()})==8
assert len({u['article'] for u in units.values()})==8
expected={(u,i,j) for u in units for i in range(4) for j in range(4)}
summaries=[];strata=[];transitions=[];checks=[];source_marginals=[];seen_field_strata=[];absent_details=[];error_categories=[];error_by_view=[]
for job in cfg['jobs']:
    folder=F/job['label'];receipt=read(folder/'receipt.json');p=folder/'predictions.jsonl';assert sha(p)==receipt['predictions_sha256'] and sha(F/'protocol.json')==receipt['protocol_sha256']
    raw=rows(p);assert len(raw)==2304 and {(r['unit_id'],r['context_index'],r['question_index']) for r in raw}==expected
    m=metrics(raw);assert m['counts']==receipt['metrics']['counts'] and math.isclose(m['known_f1'],receipt['metrics']['known_f1'],abs_tol=1e-12)
    if job['step']==1024:oldpath=O/f"followups/exposure{job['seed']}/evaluation-1024/predictions.jsonl"
    else:oldpath=next((P/'evidence/recorded/runs').glob(f"*/{job['name']}/evaluation-512/predictions.jsonl"))
    original={(r['unit_id'],r['context_index'],r['question_index']):r for r in rows(oldpath)}
    identity=0;identity_correct=0;identity_q2_original_correct=0;absent=[];categories=Counter();view_categories=defaultdict(Counter);groups=defaultdict(dict)
    for r in raw:
        u=units[r['unit_id']];i,j=r['context_index'],r['question_index'];key=u['query_keys'][j]
        assert r['gold']==u['answers'][i][j] and r['base_unit_id']==u['base_unit_id']
        assert r['query_field']==key[2] and r['field_position']==u['rendered_field_order'].index(key[2])
        teams=list(dict.fromkeys(f['key'][0] for f in u['facts'][i]));assert r['team_position']==teams.index(key[0])
        assert r['target_team_role']==u['target_team_position'] and r['semantic_rotation']==u['semantic_rotation'] and r['layout_rotation']==u['layout_rotation']
        assert r['article']==u['article'] and r['source_id']==u['source_id']
        if u['semantic_rotation']==u['layout_rotation']==u['target_team_position']==0:
            old=original[u['base_unit_id'],i,j];assert r['prediction']==old['prediction'] and r['generation']==old['generation'];identity+=1
        r['correct']=bool(em_f1(r['prediction'],r['gold'])[0]);groups[r['unit_id']][i,j]=r
        if u['semantic_rotation']==u['layout_rotation']==u['target_team_position']==0:
            identity_correct+=int(r['correct'])
            if i==0 and j==2:identity_q2_original_correct+=int(r['correct'])
        category=classify(r,u);categories[category]+=1
        identity_view=u['semantic_rotation']==u['layout_rotation']==u['target_team_position']==0
        view_categories['identity' if identity_view else 'transformed'][category]+=1
        if category=='unobserved_code':
            pred=normalize(r['prediction']);comparison=compare_codes(pred,r['gold'],[f['value'] for f in u['facts'][i]]);absent.append(comparison)
            first_key=tuple(key[:2]+[u['rendered_field_order'][0]])
            first_value=next(f['value'] for f in u['facts'][i] if tuple(f['key'])==first_key)
            absent_details.append(dict(arm=job['arm'],seed=job['seed'],step=job['step'],source_id=r['source_id'],article=r['article'],unit_id=r['unit_id'],context=i,query=j,
                field=r['query_field'],field_position=r['field_position'],identity_view=int(identity_view),prediction=pred,gold=r['gold'],first_field_value=first_value,gold_matches=comparison['matched_positions'],
                other_matches=4-comparison['other_mean_hamming'],first_field_matches=sum(a==b for a,b in zip(pred,first_value))))
    assert identity==receipt['exact_identity_replay_rows']==128
    for source in {r['source_id'] for r in raw}:
        for q in range(4):
            first=[r for r in raw if r['source_id']==source and r['context_index']==0 and r['question_index']==q]
            cells=Counter((r['query_field'],r['field_position'],r['target_team_role']) for r in first)
            assert len(cells)==18 and set(cells.values())=={1}
            assert set(cells)=={(f,p,t) for f in ('badge','locker','route') for p in range(3) for t in range(2)}
    meta=dict(arm=job['arm'],seed=job['seed'],step=job['step'])
    error_categories.extend(dict(**meta,category=category,count=count) for category,count in sorted(categories.items()))
    error_by_view.extend(dict(**meta,view=view,category=category,count=count) for view,counts in sorted(view_categories.items()) for category,count in sorted(counts.items()))
    summaries.append(dict(**meta,rows=2304,correct=m['counts']['known_exact'][0],identity_rows=128,identity_correct=identity_correct,identity_q2_original_rows=8,identity_q2_original_correct=identity_q2_original_correct,
        nonidentity_rows=2176,nonidentity_correct=m['counts']['known_exact'][0]-identity_correct,groups=144,target=m['counts']['target_joint'][0],unaffected=m['counts']['locality_joint'][0],
        absent=len(absent),absent_hamming_le2=sum(x['hamming']<=2 for x in absent),mean_gold_matches=sum(x['matched_positions'] for x in absent)/len(absent) if absent else None,
        mean_other_matches=sum(4-x['other_mean_hamming'] for x in absent)/len(absent) if absent else None))
    for q in range(4):
        part=[r for r in raw if r['context_index']==0 and r['question_index']==q and r['query_field'] in ('badge','locker')]
        assert len(part)==96
        for factor in ('query_field','field_position','target_team_role'):
            for value in sorted({r[factor] for r in part}):
                values=[r for r in part if r[factor]==value]
                seen_field_strata.append(dict(**meta,context=0,query=q,factor=factor,value=value,n=len(values),correct=sum(r['correct'] for r in values)))
    for context in ('all',0,1,2,3):
        pool=[r for r in raw if context=='all' or r['context_index']==context]
        for q in range(4):
            part=[r for r in pool if r['question_index']==q]
            for factor in ('query_field','field_position','target_team_role','team_position'):
                for value in sorted({r[factor] for r in part}):
                    values=[r for r in part if r[factor]==value]
                    strata.append(dict(**meta,context=context,query=q,factor=factor,value=value,n=len(values),correct=sum(r['correct'] for r in values)))
                    if context in ('all',0):
                        for source in sorted({r['source_id'] for r in values}):
                            v=[r for r in values if r['source_id']==source]
                            source_marginals.append(dict(**meta,context=context,query=q,factor=factor,value=value,source_id=source,article=v[0]['article'],n=len(v),correct=sum(r['correct'] for r in v)))
    for q in (2,3):
        for context in (1,2,3):
            pairs=[(g[0,q]['correct'],g[context,q]['correct']) for g in groups.values()]
            transitions.append(dict(**meta,query=q,context=context,n=len(pairs),original_correct=sum(a for a,b in pairs),correct_to_wrong=sum(a and not b for a,b in pairs),wrong_to_correct=sum(not a and b for a,b in pairs)))
    checks.append(dict(job=job['label'],rows=2304,identity_replay=identity,predictions_sha256=sha(p)))
assert len(checks)==6
absent_by_view=[]
for meta in summaries:
    for identity_view in (1,0):
        part=[r for r in absent_details if (r['arm'],r['seed'],r['step'],r['identity_view'])==(meta['arm'],meta['seed'],meta['step'],identity_view)]
        if part:absent_by_view.append(dict(arm=meta['arm'],seed=meta['seed'],step=meta['step'],view='identity' if identity_view else 'transformed',absent=len(part),mean_gold_matches=sum(r['gold_matches'] for r in part)/len(part),mean_other_matches=sum(r['other_matches'] for r in part)/len(part)))
for name,records in [('role_summary.csv',summaries),('role_query_factors.csv',strata),('role_transitions.csv',transitions),('role_source_marginals.csv',source_marginals),('role_seen_fields_query_factors.csv',seen_field_strata),('role_absent_rows.csv',absent_details),('role_error_categories.csv',error_categories),('role_error_by_view.csv',error_by_view),('role_absent_by_view.csv',absent_by_view)]:save(name,records)
(O/'role_cpu_audit.json').write_text(json.dumps(dict(status='verified',models=6,rows=13824,sources=8,articles=8,factorial='3 semantic rotations x3 serialized-field rotations x2 first-two-team roles',test_accessed=False,files=checks),indent=2)+'\n',encoding='utf8')
print(json.dumps(summaries))
