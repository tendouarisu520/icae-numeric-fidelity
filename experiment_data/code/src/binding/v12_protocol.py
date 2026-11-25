"""Pure fixed-memory intervention and development-only calibration protocol."""
from collections import Counter
import hashlib
import math
from .v11_pipeline import em_f1,normalize

SOURCES={'sft512':('sft',512),'sft1024':('sft',1024),'relation1024':('relation',1024)}


def shuffled_donors(units):
    if len({u['id'] for u in units})!=len(units) or len(units)<2:raise ValueError('Unique units required')
    counts=Counter(u['article'] for u in units);offset=max(counts.values())
    if 2*offset>len(units):raise ValueError('No article-disjoint bijection exists')
    ordered=sorted(units,key=lambda u:(hashlib.sha256(u['article'].encode()).hexdigest(),u['id']))
    result={u['id']:ordered[(i+offset)%len(ordered)]['id'] for i,u in enumerate(ordered)}
    articles={u['id']:u['article'] for u in units}
    if any(articles[k]==articles[v] for k,v in result.items()):raise ValueError('Donor article collision')
    return result


def banned_next(prefix,sequences):
    return {s[-1] for s in sequences if s and list(prefix)==list(s[:-1])}


def prediction_at(row,threshold):
    if not row['alternative_valid'] or row['margin'] is None:return 'UNKNOWN'
    return row['alternative'] if row['margin']>threshold else 'UNKNOWN'


def _fit(rows):
    margins=sorted({r['margin'] for r in rows if r['alternative_valid'] and r['margin'] is not None})
    if any(not math.isfinite(v) for v in margins):raise ValueError('Nonfinite calibration score')
    candidates=([margins[0]-1.]+[(a+b)/2 for a,b in zip(margins,margins[1:])]+[margins[-1]+1.]) if margins else [0.]
    # Outcomes are cached; fitting thresholds never changes the generated alternatives.
    outcomes=[(r,em_f1(r['alternative'],r['gold'])[1],float(normalize(r['gold'])=='unknown')) for r in rows]
    def objective(threshold):
        f1=0.;false_unknown=0
        for r,alternative_f1,unknown_f1 in outcomes:
            answered=r['alternative_valid'] and r['margin'] is not None and r['margin']>threshold
            f1+=alternative_f1 if answered else unknown_f1
            false_unknown+=int(not answered and r['gold']!='UNKNOWN')
        return (f1/len(rows),-false_unknown,-abs(threshold),threshold)
    return max(candidates,key=objective)


def calibrate(records,articles):
    if not records:raise ValueError('Nonempty development scores required')
    keys=[(r['unit_id'],r['context_index'],r['question_index']) for r in records]
    if len(keys)!=len(set(keys)):raise ValueError('Duplicate scored question')
    used={articles[r['unit_id']] for r in records}
    if len(used)<2:raise ValueError('At least two development articles required')
    ordered=sorted(used,key=lambda a:hashlib.sha256(a.encode()).hexdigest())
    folds={a:i%2 for i,a in enumerate(ordered)};predictions=[None]*len(records);details=[]
    for heldout in (0,1):
        fit=[r for r in records if folds[articles[r['unit_id']]]!=heldout]
        indices=[i for i,r in enumerate(records) if folds[articles[r['unit_id']]]==heldout]
        threshold=_fit(fit)
        for i in indices:predictions[i]=prediction_at(records[i],threshold)
        details.append(dict(fold=heldout,threshold=threshold,
            fit_articles=sorted(a for a in used if folds[a]!=heldout),
            evaluation_articles=sorted(a for a in used if folds[a]==heldout),fit_rows=len(fit),evaluation_rows=len(indices)))
    return dict(method='Twofold article-held-out development diagnostic; rawgreedy remains primary',
        article_folds=folds,folds=details,out_of_fold_predictions=predictions,
        out_of_fold_f1=sum(em_f1(p,r['gold'])[1] for p,r in zip(predictions,records))/len(records),
        all_development_threshold=_fit(records),confirmation_evaluated=False,reserved_test_accessed=False)
