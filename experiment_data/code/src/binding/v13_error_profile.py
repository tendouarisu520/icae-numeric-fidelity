"""Read-only relation error taxonomy and article-clustered paired intervals."""
from collections import Counter,defaultdict
import random
import re
from .v11_pipeline import normalize
from .v13_protocol import metrics

def classify(row,unit):
    i,j=row['context_index'],row['question_index']
    if row['gold']!=unit['answers'][i][j]:raise ValueError('Gold/scene mismatch')
    prediction=normalize(row['prediction'])
    if prediction==normalize(row['gold']):return 'correct'
    mapping={f['value']:tuple(f['key']) for f in unit['facts'][i]}
    if prediction in mapping:
        expected=tuple(unit['query_keys'][j]);actual=mapping[prediction]
        differences=[k for k in range(3) if expected[k]!=actual[k]]
        return {0:'entity',1:'condition',2:'field'}[differences[0]] if len(differences)==1 else 'mixed_binding'
    if re.fullmatch(r'\d{4}',prediction):return 'unobserved_code'
    if any(code in mapping for code in re.findall(r'\b\d{4}\b',row['prediction'])):return 'format_with_code'
    return 'non_code'

def profile(rows,units):
    by_unit={u['id']:u for u in units};groups=defaultdict(list)
    for row in rows:groups[row['unit_id']].append(row)
    if set(groups)!=set(by_unit):raise ValueError('Scene coverage differs')
    errors=Counter();by_context=defaultdict(Counter);by_level=defaultdict(Counter);by_query=defaultdict(Counter)
    scores={};probe_rows=0
    for uid,values in groups.items():
        unit=by_unit[uid];result=metrics(values);sid=unit['source_id']
        s=scores.setdefault(sid,dict(article=unit['article'],target=[],locality=[],natural_f1=None))
        if s['article']!=unit['article']:raise ValueError('Source article mismatch')
        if unit['family']=='natural':s['natural_f1']=result['known_f1'];continue
        s['target'].append(result['rates']['target_joint']);s['locality'].append(result['rates']['locality_joint'])
        for row in values:
            category=classify(row,unit);errors[category]+=1;probe_rows+=1
            by_context[str(row['context_index'])][category]+=1;by_level[str(unit['level'])][category]+=1
            by_query[str(row['question_index'])][category]+=1
    source_scores={sid:dict(article=s['article'],target_joint=sum(s['target'])/len(s['target']) if s['target'] else None,
        locality_joint=sum(s['locality'])/len(s['locality']) if s['locality'] else None,natural_f1=s['natural_f1']) for sid,s in scores.items()}
    return dict(rows=probe_rows,errors=dict(errors),by_context={k:dict(v) for k,v in by_context.items()},
        by_level={k:dict(v) for k,v in by_level.items()},by_query={k:dict(v) for k,v in by_query.items()},source_scores=source_scores,
        primary_predictions_and_metrics_unchanged=True)

def paired_intervals(baseline,candidate,repetitions=2000,seed=20261026):
    if set(baseline)!=set(candidate):raise ValueError('Paired source coverage differs')
    fields=('target_joint','locality_joint','natural_f1');articles=defaultdict(list)
    for sid,a in baseline.items():
        b=candidate[sid]
        if a['article']!=b['article']:raise ValueError('Paired article mismatch')
        articles[a['article']].append([b[f]-a[f] for f in fields])
    keys=sorted(articles);rng=random.Random(seed);samples=[[] for _ in fields]
    for _ in range(repetitions):
        values=[v for a in rng.choices(keys,k=len(keys)) for v in articles[a]]
        for j in range(len(fields)):samples[j].append(sum(v[j] for v in values)/len(values))
    result={}
    for j,field in enumerate(fields):
        values=sorted(samples[j]);point=sum(candidate[s][field]-baseline[s][field] for s in baseline)/len(baseline)
        result[field]=dict(delta=point,interval95=[values[int(.025*(repetitions-1))],values[int(.975*(repetitions-1))]])
    return dict(method='Paired source-weighted article-cluster bootstrap; development diagnostic',
        sources=len(baseline),articles=len(keys),repetitions=repetitions,fields=result)
