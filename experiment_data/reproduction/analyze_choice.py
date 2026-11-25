"""Independent CPU audit of matched candidate scoring and recorded generation."""
from pathlib import Path
import csv, hashlib, json, math, random
from collections import defaultdict

R=Path(__file__).resolve().parents[1]
P=R
O=R/'followup'
C=O/'choice'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def rows(p):return [json.loads(x) for x in p.read_text(encoding='utf8').splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def interval(records,field):
    grouped=defaultdict(list)
    for r in records:grouped[r['article']].append(float(r[field]))
    keys=sorted(grouped);sums={k:sum(v) for k,v in grouped.items()};ns={k:len(v) for k,v in grouped.items()}
    rng=random.Random(20261106);values=[]
    for _ in range(2000):
        sample=rng.choices(keys,k=len(keys));values.append(sum(sums[k] for k in sample)/sum(ns[k] for k in sample))
    values.sort();return [values[49],values[1949]]
def save(name,records):
    with (O/name).open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)

protocol=read(C/'protocol.json')
assert read(C/'status.json')['phase']=='complete'
units={'development':{u['id']:u for u in rows(P/'data/v13-answerable-relations-20261026/development.jsonl')},
       'confirmation':{u['id']:u for u in rows(P/'confirmation/confirmation_units.jsonl')}}
pins={x['path']:x['sha256'] for x in read(P/'reproduction/interpretation_audit.json')['prediction_files']}
summary=[];strata=[];audit=[]
for recpath in sorted(C.glob('*/*/receipt.json')):
    receipt=read(recpath);split=receipt['split'];name=receipt['job'];path=recpath.parent/'scores.jsonl'
    assert sha(path)==receipt['scores_sha256'] and sha(C/'protocol.json')==receipt['protocol_sha256']
    _,arm,method,seed=name.split('-');seed=int(seed[4:]);data=rows(path)
    candidates=[p for p in pins if name in p and ((split=='confirmation' and p.startswith('confirmation/')) or (split=='development' and '/evaluation-512/' in p))]
    assert len(candidates)==1
    original=P/candidates[0];assert sha(original)==pins[candidates[0]]==receipt['raw_predictions_sha256']
    raw={(r['unit_id'],r['context_index'],r['question_index']):r for r in rows(original) if r['family']=='probe'}
    seen=set();groups=defaultdict(dict);token_lengths=set()
    for r in data:
        key=r['unit_id'],r['context_index'],r['question_index'];assert key not in seen;seen.add(key)
        u=units[split][r['unit_id']];old=raw[key];i,j=key[1:]
        assert i in range(4) and j in (0,1)
        assert r['source_id']==u['source_id'] and r['article']==u['article'] and r['level']==u['level']
        assert r['gold']==u['answers'][i][j]==old['gold'] and r['prediction']==old['prediction']
        assert r['candidate_values']==u['answers'][i][:2]
        scores=r['candidate_scores'];assert len(scores)==2
        assert len(scores[0]['token_ids'])==len(scores[1]['token_ids'])
        for s in scores:
            n=len(s['token_ids']);token_lengths.add(n)
            assert len(s['token_logp'])==len(s['teacher_forced_argmax'])==n
            assert all(math.isfinite(x) and x<=1e-6 for x in s['token_logp'])
            assert abs(sum(s['token_logp'])-s['sum'])<1e-4 and abs(sum(s['token_logp'])/n-s['mean'])<2e-5
        a,b=[s['mean'] for s in scores];choice=None if a==b else int(b>a)
        assert r['choice']==choice and r['choice_correct']==(choice==j)
        # Outputs in this controlled probe are exact four-digit codes; whitespace normalization is sufficient.
        greedy=r['prediction'].strip()==r['gold'];assert greedy==r['greedy_correct']
        assert abs(r['gold_margin']-(scores[j]['mean']-scores[1-j]['mean']))<1e-6
        r['difference']=int(r['choice_correct'])-int(greedy);groups[key[0]][i,j]=r
    expected={(u['id'],i,j) for u in units[split].values() if u['family']=='probe' for i in range(4) for j in (0,1)}
    assert seen==expected
    joint=[]
    for uid,g in groups.items():
        assert len(g)==8
        a=all(g[i,j]['choice_correct'] for i in (0,1) for j in (0,1));b=all(g[i,j]['greedy_correct'] for i in (0,1) for j in (0,1))
        joint.append(dict(article=units[split][uid]['article'],choice=int(a),greedy=int(b),difference=int(a)-int(b)))
    calc=dict(rows=len(data),choice_correct=sum(r['choice_correct'] for r in data),greedy_correct=sum(r['greedy_correct'] for r in data),ties=sum(r['choice'] is None for r in data),
        wrong_generation_correct_choice=sum(r['choice_correct'] and not r['greedy_correct'] for r in data),correct_generation_wrong_choice=sum(r['greedy_correct'] and not r['choice_correct'] for r in data),
        choice_target_joint=sum(r['choice'] for r in joint),greedy_target_joint=sum(r['greedy'] for r in joint),groups=len(joint))
    for k,v in calc.items():assert receipt[k]==v,(name,k)
    lo,hi=interval(data,'difference');jlo,jhi=interval(joint,'difference')
    summary.append(dict(split=split,arm=arm,method=method,seed=seed,**calc,choice_accuracy=calc['choice_correct']/len(data),greedy_accuracy=calc['greedy_correct']/len(data),
        paired_accuracy_difference=(calc['choice_correct']-calc['greedy_correct'])/len(data),paired_difference_cluster95_low=lo,paired_difference_cluster95_high=hi,
        paired_joint_difference=(calc['choice_target_joint']-calc['greedy_target_joint'])/len(joint),joint_difference_cluster95_low=jlo,joint_difference_cluster95_high=jhi))
    for factor in ('question_index','context_index','level'):
        for value in sorted({r[factor] for r in data}):
            part=[r for r in data if r[factor]==value]
            strata.append(dict(split=split,arm=arm,method=method,seed=seed,factor=factor,value=value,n=len(part),choice_correct=sum(r['choice_correct'] for r in part),greedy_correct=sum(r['greedy_correct'] for r in part)))
    audit.append(dict(split=split,job=name,rows=len(data),score_sha256=sha(path),raw_prediction_sha256=sha(original),candidate_lengths=sorted(token_lengths),gpu_greedy_replay_rows=receipt['exact_replay_rows']))
assert len(summary)==20
save('choice_summary.csv',summary);save('choice_strata.csv',strata)
(O/'choice_cpu_audit.json').write_text(json.dumps(dict(status='verified',cells=20,total_rows=sum(r['rows'] for r in summary),paired_bootstrap='2000 article-cluster resamples, seed20261106, percentile interval',
    joint_definition='all four target responses: original and condition-swap contexts, both target queries; other contexts included in row accuracy',post_hoc=True,test_accessed=False,files=audit),indent=2)+'\n',encoding='utf8')
for r in summary:
    if r['method']=='sft':print(json.dumps(r))
