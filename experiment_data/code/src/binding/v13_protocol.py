"""Complete-group answerable evaluation; no prediction-based exclusions."""
from collections import defaultdict
import math
from .v11_pipeline import em_f1,normalize

def prediction_rows(unit,predictions):
    if len(predictions)!=4 or any(len(p)!=len(unit['queries']) for p in predictions):raise ValueError('Four complete context rows required')
    return [dict(unit_id=unit['id'],source_id=unit['source_id'],article=unit['article'],family=unit['family'],level=unit['level'],
        context_index=i,question_index=j,prediction=predictions[i][j],gold=gold,
        candidates=unit['answers'][i][:2] if unit['family']=='probe' else [unit['answers'][0][0],unit['answers'][1][0]])
        for i,row in enumerate(unit['answers']) for j,gold in enumerate(row) if gold is not None]

def metrics(rows):
    if not rows or len({r['family'] for r in rows})!=1:raise ValueError('Nonempty single-family groups required')
    family=rows[0]['family'];groups=defaultdict(list)
    for row in rows:
        if row['gold'] is None or row['gold']=='UNKNOWN':raise ValueError('Answerable rows only')
        groups[row['unit_id']].append(row)
    counts=dict(known_exact=[0,len(rows)],target_joint=[0,len(groups)],locality_joint=[0,len(groups)],
        irrelevant_target_joint=[0,len(groups)],reorder_target_joint=[0,len(groups)],candidate_misbinding=[0,0])
    if family=='natural':counts['reorder_target_joint']=[0,0]
    f1=[]
    for values in groups.values():
        by={(r['context_index'],r['question_index']):r for r in values}
        expected={(i,j) for i in range(4) for j in range(4)} if family=='probe' else {(0,0),(0,1),(1,0),(1,1),(2,1),(3,0),(3,1)}
        if len(by)!=len(values) or set(by)!=expected:raise ValueError('Incomplete or duplicate relation group')
        correct={k:bool(em_f1(r['prediction'],r['gold'])[0]) for k,r in by.items()}
        counts['known_exact'][0]+=sum(correct.values());f1.extend(em_f1(r['prediction'],r['gold'])[1] for r in values)
        targets=(0,1) if family=='probe' else (0,);local=(2,3) if family=='probe' else (1,)
        counts['target_joint'][0]+=int(all(correct[i,j] for i in (0,1) for j in targets))
        counts['locality_joint'][0]+=int(all(correct[i,j] for i in range(4) for j in local))
        counts['irrelevant_target_joint'][0]+=int(all(correct[i,j] for i in (0,2 if family=='probe' else 3) for j in targets))
        if family=='probe':counts['reorder_target_joint'][0]+=int(all(correct[i,j] for i in (0,3) for j in targets))
        for row in values:
            if row['question_index'] in targets:
                counts['candidate_misbinding'][1]+=1
                counts['candidate_misbinding'][0]+=int(any(normalize(row['prediction'])==normalize(c) and normalize(c)!=normalize(row['gold']) for c in row['candidates'] if c is not None))
    return dict(family=family,counts=counts,rates={k:a/b if b else None for k,(a,b) in counts.items()},
        known_f1=math.fsum(f1)/len(f1),units=len(groups),sources=len({r['source_id'] for r in rows}),
        articles=len({r['article'] for r in rows}),rows=len(rows))

def learning_rate(step,total=512,peak=5e-5,warmup=32):
    if not 1<=step<=total<=1024 or warmup>=total:raise ValueError('Bounded cosine schedule required')
    if step<=warmup:return peak*step/warmup
    return peak*(.1+.9*(1+math.cos(math.pi*(step-warmup)/(total-warmup)))/2)
