"""Post hoc descriptive digit analysis of all stored, hash-verified generations."""
from pathlib import Path
import csv
import hashlib
import json
import random
import statistics
import sys
from collections import Counter,defaultdict

R=Path(__file__).resolve().parents[1]
P=R
O=R/'followup'
sys.path.insert(0,str(R/'code/src'))
from binding.v13_error_profile import classify
from binding.v11_pipeline import normalize
from binding.v16_code_fidelity import compare_codes

def lines(p):return [json.loads(x) for x in p.read_text(encoding='utf8').splitlines()]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,rows):
    with (O/name).open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def cluster_interval(rows,field):
    articles=defaultdict(list)
    for r in rows:articles[r['article']].append(float(r[field]))
    rng=random.Random(20261106);keys=sorted(articles);samples=[]
    sums={k:sum(articles[k]) for k in keys};counts={k:len(articles[k]) for k in keys}
    for _ in range(2000):
        draw=rng.choices(keys,k=len(keys))
        samples.append(sum(sums[k] for k in draw)/sum(counts[k] for k in draw))
    samples.sort()
    return [samples[49],samples[1949]]

O.mkdir(parents=True,exist_ok=True)
units={'development':{u['id']:u for u in lines(P/'data/v13-answerable-relations-20261026/development.jsonl')},
       'confirmation':{u['id']:u for u in lines(P/'confirmation/confirmation_units.jsonl')}}
pins={x['path']:x['sha256'] for x in json.loads((P/'reproduction/interpretation_audit.json').read_text())['prediction_files']}
summary=[];by_role=[];details=[];tokens=[]
for rel,pin in sorted(pins.items()):
    path=P/rel;assert digest(path)==pin
    split='confirmation' if rel.startswith('confirmation/') else 'development'
    name=path.parent.name if split=='confirmation' else path.parent.parent.name
    _,arm,method,seedstring=name.split('-');seed=int(seedstring[4:])
    step=512 if split=='confirmation' else int(path.parent.name.split('-')[1])
    meta=dict(split=split,arm=arm,method=method,seed=seed,step=step)
    raw=lines(path);probe=[r for r in raw if r['family']=='probe'];absent=[]
    counts=Counter()
    for row in probe:
        unit=units[split][row['unit_id']];category=classify(row,unit);counts[category]+=1
        if category!='unobserved_code':continue
        pred=normalize(row['prediction']);gold=row['gold'];facts=unit['facts'][row['context_index']]
        value=compare_codes(pred,gold,[f['value'] for f in facts])
        item=dict(**meta,source_id=row['source_id'],article=row['article'],unit_id=row['unit_id'],
            level=row['level'],context_index=row['context_index'],question_index=row['question_index'],
            prediction=pred,gold=gold,**value)
        item['hamming_advantage_over_other_mean']=value['other_mean_hamming']-value['hamming']
        item['at_minimum_excess']=int(value['gold_at_minimum'])-value['null_at_minimum_fraction']
        absent.append(item)
        if step==512:details.append(item)
        if split=='development' and step==512 and arm=='compressed' and method=='sft':
            for c,token in zip(row['prediction'].lstrip(),row['generation']['token_ids'][1:]):
                tokens.append((c,token))
    n=len(absent)
    record=dict(**meta,controlled_rows=len(probe),correct=counts['correct'],wrong=len(probe)-counts['correct'],
        absent=n,non_code=counts['non_code'],format_with_code=counts['format_with_code'])
    if n:
        record.update(hamming1=sum(x['hamming']==1 for x in absent),hamming2=sum(x['hamming']==2 for x in absent),
            hamming3=sum(x['hamming']==3 for x in absent),hamming4=sum(x['hamming']==4 for x in absent),
            adjacent_swap=sum(x['adjacent_swap'] for x in absent),anagram=sum(x['same_digit_multiset'] for x in absent),
            mean_matched_positions=statistics.mean(x['matched_positions'] for x in absent),
            mean_digit_overlap=statistics.mean(x['digit_overlap'] for x in absent),
            gold_unique_nearest=sum(x['gold_unique_nearest'] for x in absent),gold_at_minimum=sum(x['gold_at_minimum'] for x in absent),
            mean_other_matched_positions=statistics.mean(4-x['other_mean_hamming'] for x in absent),
            mean_hamming_advantage=statistics.mean(x['hamming_advantage_over_other_mean'] for x in absent),
            closest_null_fraction=statistics.mean(x['null_at_minimum_fraction'] for x in absent),
            position_matches=[sum(x['match_mask'][j]=='=' for x in absent) for j in range(4)])
        if step==512:
            record['hamming_advantage_article_cluster95']=cluster_interval(absent,'hamming_advantage_over_other_mean')
            record['closest_excess_article_cluster95']=cluster_interval(absent,'at_minimum_excess')
        for field in ('question_index','context_index','level'):
            for role in sorted({r[field] for r in absent}):
                subset=[r for r in absent if r[field]==role]
                by_role.append(dict(**meta,factor=field,value=role,n=len(subset),
                    hamming_le2=sum(r['hamming']<=2 for r in subset),
                    gold_at_minimum=sum(r['gold_at_minimum'] for r in subset),
                    mean_hamming_advantage=statistics.mean(r['hamming_advantage_over_other_mean'] for r in subset)))
    summary.append(record)
save('absent_code_rows.csv',details)
save('absent_code_strata.csv',by_role)
allfields=list(dict.fromkeys(k for r in summary for k in r))
save('absent_code_summary.csv',[{k:r.get(k,'') for k in allfields} for r in summary])
report=dict(status='complete',date='2026-11-06',analysis='post hoc, exploratory in both inspected cohorts',
    input_prediction_files=len(pins),input_prediction_rows=88000,input_hashes=pins,
    endpoint_absent_rows=len(details),per_record_null='Prediction compared with every non-gold value in its current record',
    bootstrap='2000 article-cluster resamples; ratio-of-sums among absent rows, conditional on observed absent errors',
    correction_performed=False,test_accessed=False,token_map_observed=sorted(set(tokens)),summaries=summary)
(O/'analysis.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
for r in summary:
    if r['step']==512 and r['method']=='sft' and r['arm']=='compressed':print(json.dumps(r))
print(json.dumps(dict(status='complete',prediction_files=40,endpoint_absent_rows=len(details))))
