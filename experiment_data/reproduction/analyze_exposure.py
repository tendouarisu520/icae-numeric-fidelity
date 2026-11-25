"""Verify repeated-exposure runs and rescore every development prediction."""
from pathlib import Path
import csv, hashlib, json, math, random, statistics, sys
from collections import Counter
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'code/src'))
from binding.v13_protocol import metrics
from binding.v13_error_profile import classify
from binding.v11_pipeline import normalize
from binding.v16_exposure import exposure_rate,plateau_diagnostic
from binding.v16_code_fidelity import compare_codes
O=R/'followup';P=R
def read(p):return json.loads(p.read_text(encoding='utf8'))
def rows(p):return [json.loads(x) for x in p.read_text(encoding='utf8').splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
units={u['id']:u for u in rows(P/'data/v13-answerable-relations-20261026/development.jsonl')}
expected={(u['id'],i,j) for u in units.values() for i,answers in enumerate(u['answers']) for j,a in enumerate(answers) if a is not None}
pins={x['path']:x['sha256'] for x in read(P/'reproduction/interpretation_audit.json')['prediction_files']}
out=[];checks=[];errors=[];paired_training=[]
for seed in (73,97):
    root=O/f'followups/exposure{seed}'
    if not root.exists():continue
    summary=read(root/'summary.json');config=read(root/'config.json');steps=rows(root/'steps.jsonl')
    assert summary['status']=='success' and summary['optimizer_steps']==1024 and summary['step512_exact_replay']
    assert summary['cache_unchanged'] and summary['frozen_parameters_unchanged'] and summary['reload_exact']
    assert len(steps)==len(config['schedule'])==1024 and set(Counter(config['schedule']).values())=={2}
    assert len(set(config['schedule']))==512 and set(config['schedule'][:512])==set(config['schedule'][512:])
    train_sources=list(dict.fromkeys(u['source_id'] for u in rows(P/'data/v13-answerable-relations-20261026/train.jsonl')))
    rng=random.Random(seed);reconstructed=[]
    for _ in range(2):
        epoch=list(train_sources);rng.shuffle(epoch);reconstructed.extend(epoch)
    assert reconstructed==config['schedule']
    for n,s in enumerate(steps,1):
        assert s['step']==n and s['source_id']==config['schedule'][n-1]
        assert math.isclose(s['learning_rate'],exposure_rate(n),rel_tol=1e-12)
        assert math.isfinite(s['loss']) and math.isfinite(s['gradient_norm'])
    first={s['source_id']:.5*sum(v['sft_loss'] for v in s['families']) for s in steps[:512]}
    second={s['source_id']:.5*sum(v['sft_loss'] for v in s['families']) for s in steps[512:]}
    for source in sorted(first):paired_training.append(dict(seed=seed,source_id=source,first_exposure_loss=first[source],second_exposure_loss=second[source],difference=second[source]-first[source]))
    for step,e in summary['evaluations'].items():
        k=int(step);p=root/f'evaluation-{step}/predictions.jsonl';assert sha(p)==e['predictions_sha256'];raw=rows(p)
        assert len(raw)==1760 and {(r['unit_id'],r['context_index'],r['question_index']) for r in raw}==expected
        for r in raw:
            u=units[r['unit_id']];assert r['gold']==u['answers'][r['context_index']][r['question_index']]
            assert (r['source_id'],r['article'],r['family'])==(u['source_id'],u['article'],u['family'])
        for family in ('natural','probe'):
            m=metrics([r for r in raw if r['family']==family]);assert m['counts']==e[family]['counts'] and math.isclose(m['known_f1'],e[family]['known_f1'],abs_tol=1e-12)
        if k<=512:
            old=next(v for path,v in pins.items() if f'main-compressed-sft-seed{seed}/evaluation-{k}/' in path)
            assert sha(p)==old
        probe=[r for r in raw if r['family']=='probe'];categories=Counter(classify(r,units[r['unit_id']]) for r in probe);absent=[]
        for r in probe:
            u=units[r['unit_id']]
            if classify(r,u)!='unobserved_code':continue
            v=compare_codes(normalize(r['prediction']),r['gold'],[f['value'] for f in u['facts'][r['context_index']]])
            absent.append(v);errors.append(dict(seed=seed,step=k,unit_id=r['unit_id'],source_id=r['source_id'],article=r['article'],context_index=r['context_index'],question_index=r['question_index'],prediction=normalize(r['prediction']),gold=r['gold'],**v))
        out.append(dict(seed=seed,step=k,sources_seen=min(k,512),source_exposures=k/512,controlled_correct=e['probe']['counts']['known_exact'][0],controlled_rows=1536,
            target_rows_correct=sum(r['prediction'].strip()==r['gold'] for r in probe if r['question_index'] in (0,1)),target_rows=768,
            target=e['probe']['counts']['target_joint'][0],groups=96,unaffected=e['probe']['counts']['locality_joint'][0],reorder=e['probe']['counts']['reorder_target_joint'][0],
            natural_exact=e['natural']['counts']['known_exact'][0],natural_rows=224,natural_f1=e['natural']['known_f1'],absent=categories['unobserved_code'],
            absent_hamming_le2=sum(v['hamming']<=2 for v in absent),absent_mean_gold_matches=sum(v['matched_positions'] for v in absent)/len(absent),
            absent_mean_other_matches=sum(4-v['other_mean_hamming'] for v in absent)/len(absent),predictions_sha256=sha(p)))
    losses=[.5*sum(v['sft_loss'] for v in s['families']) for s in steps[-256:]]
    plateau=plateau_diagnostic([summary['evaluations'][str(k)]['probe']['counts']['target_joint'][0] for k in (768,896,1024)],losses)
    differences=[]
    for key,value in summary['plateau_diagnostic'].items():
        if type(value) is float:
            differences.append(abs(plateau[key]-value));assert math.isclose(plateau[key],value,rel_tol=0,abs_tol=1e-12)
        else:assert plateau[key]==value
    checks.append(dict(seed=seed,complete=True,exact_replay_checkpoints=[128,256,512],source_exposures=2,plateau=plateau,maximum_cpu_float_reduction_difference=max(differences),
        paired_training=dict(sources=512,first_exposure_mean=statistics.mean(first.values()),second_exposure_mean=statistics.mean(second.values()),sources_with_lower_loss=sum(second[k]<v for k,v in first.items()),interpretation='Same source at its first and second online exposures; models differ across updates, not a fixed-checkpoint training-set evaluation'),
        wall_seconds=summary['wall_seconds'],training_seconds=summary['cumulative_training_seconds'],peak_allocated_mib=summary['peak_allocated_mib'],summary_sha256=sha(root/'summary.json')))
assert checks,'No completed downloaded runs'
for name,records in [('exposure_summary.csv',out),('exposure_absent_rows.csv',errors),('exposure_paired_training_loss.csv',paired_training)]:
    with (O/name).open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
(O/'exposure_cpu_audit.json').write_text(json.dumps(dict(status='verified',completed_runs=len(checks),test_accessed=False,runs=checks),indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(runs=checks,curves=[{k:r[k] for k in ('seed','step','target','controlled_correct','absent')} for r in out])))
