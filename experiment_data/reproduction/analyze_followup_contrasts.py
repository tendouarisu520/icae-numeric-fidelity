"""Paired descriptive article-cluster contrasts for the completed follow-ups."""
from pathlib import Path
from collections import defaultdict
import csv,json,random
R=Path(__file__).resolve().parents[1]
O=R/'followup'

def read(p):return json.loads(p.read_text(encoding='utf8'))
def rows(p):return [json.loads(x) for x in p.read_text(encoding='utf8').splitlines()]
def interval(pairs,universe=None):
    assert pairs,'A conditional contrast requires at least one qualifying observation'
    by=defaultdict(list)
    if universe is not None:
        for article in universe:by[article]=[]
    for article,value in pairs:by[article].append(value)
    keys=sorted(by);total={k:sum(by[k]) for k in keys};n={k:len(by[k]) for k in keys}
    rng=random.Random(20261106);samples=[]
    while len(samples)<2000:
        draw=rng.choices(keys,k=len(keys));denominator=sum(n[k] for k in draw)
        if denominator:samples.append(sum(total[k] for k in draw)/denominator)
    samples.sort()
    return dict(n=len(pairs),articles=len(keys),difference=sum(v for _,v in pairs)/len(pairs),low95=samples[49],high95=samples[1949])

# Exact zero and fixed-effect checks guard the resampling/denominator convention.
assert interval([('a',0),('b',0)])['high95']==0
assert interval([('a',1),('b',1),('b',1)])['low95']==1
assert interval([('a',1)],universe=('a','b'))['low95']==1
out=[]
assert read(O/'exposure_cpu_audit.json')['completed_runs']==2
assert read(O/'near_cpu_audit.json')['cells']==10
for seed in (73,97):
    root=O/f'followups/exposure{seed}'
    old={(r['unit_id'],r['context_index'],r['question_index']):r for r in rows(root/'evaluation-512/predictions.jsonl') if r['family']=='probe'}
    final={(r['unit_id'],r['context_index'],r['question_index']):r for r in rows(root/'evaluation-1024/predictions.jsonl') if r['family']=='probe'}
    assert old.keys()==final.keys();paired=[];groups=defaultdict(list)
    for key,r in old.items():
        s=final[key];assert r['gold']==s['gold'] and r['article']==s['article']
        a=r['prediction'].strip()==r['gold'];b=s['prediction'].strip()==s['gold']
        paired.append((r['article'],int(b)-int(a)))
        if key[1] in (0,1) and key[2] in (0,1):groups[key[0]].append((r['article'],a,b))
    out.append(dict(experiment='exposure',split='development',arm='compressed',seed=seed,step=1024,contrast='controlled_accuracy_1024_minus_512',**interval(paired)))
    joint=[]
    for g in groups.values():
        assert len(g)==4 and len({x[0] for x in g})==1
        joint.append((g[0][0],int(all(x[2] for x in g))-int(all(x[1] for x in g))))
    out.append(dict(experiment='exposure',split='development',arm='compressed',seed=seed,step=1024,contrast='target_joint_1024_minus_512',**interval(joint)))

protocol=read(O/'followups/near/protocol.json')
for job in protocol['jobs']:
    for split in job['splits']:
        data=rows(O/'followups/near'/split/job['label']/'scores.jsonl')
        pairs=defaultdict(list);wrong_pairs=defaultdict(list)
        for r in data:
            gold=r['gold_score']['mean'];condition=gold>r['condition_foil_score']['mean']
            first=gold>r['near']['0']['score']['mean'];last=gold>r['near']['3']['score']['mean']
            for label,value in [('condition_minus_first',int(condition)-int(first)),('condition_minus_last',int(condition)-int(last)),('first_minus_last',int(first)-int(last))]:pairs[label].append((r['article'],value))
            if not r['greedy_correct'] and r['gold_score']['tokens']==r['generated_score']['tokens']:
                generated=r['gold_score']['sum']>r['generated_score']['sum']
                for label,value in [('condition_minus_generated',int(condition)-int(generated)),('first_minus_generated',int(first)-int(generated)),('last_minus_generated',int(last)-int(generated))]:wrong_pairs[label].append((r['article'],value))
        for label,values in pairs.items():out.append(dict(experiment='near_code',split=split,arm=job['arm'],seed=job['seed'],step=job['step'],contrast=label,**interval(values)))
        for label,values in wrong_pairs.items():out.append(dict(experiment='incorrect_equal_length_generation',split=split,arm=job['arm'],seed=job['seed'],step=job['step'],contrast=label,**interval(values,{r['article'] for r in data})))
assert len(out)==52
with (O/'followup_paired_contrasts.csv').open('w',newline='',encoding='utf8') as f:
    w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
(O/'followup_contrast_audit.json').write_text(json.dumps(dict(status='verified',contrasts=len(out),bootstrap='2000 paired article-cluster resamples; seed20261106; percentile interval; ratio of sums. Incorrect-generation contrasts retain the full cohort article universe, including articles with zero qualifying errors; empty-denominator resamples are redrawn.',scope='descriptive post hoc contrasts conditional on fitted readers',test_accessed=False),indent=2)+'\n',encoding='utf8')
print(json.dumps(out))
