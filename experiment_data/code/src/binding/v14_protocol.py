"""Pure bottleneck conditional-margin reference, independent of Torch."""
import math

def gaps(scores):
    if len(scores)!=2 or any(len(row)!=2 for row in scores):raise ValueError('Two-by-two scores required')
    a,b=scores[0];c,d=scores[1]
    values=[a-b,d-c,a-c,d-b]
    if any(not math.isfinite(v) for v in values):raise ValueError('Finite conditional gaps required')
    return values

def soft_min(values,temperature=.5):
    if not values or temperature<=0 or any(not math.isfinite(v) for v in values):raise ValueError('Finite gaps and positive temperature required')
    m=min(values)
    return m-temperature*math.log(sum(math.exp(-(v-m)/temperature) for v in values))

def penalty(scores):
    value=.5-soft_min(gaps(scores))
    return max(value,0.)+math.log1p(math.exp(-abs(value)))

def should_run(comparison):
    expected={(a,m,s) for s in (73,97) for a,m in
        (('compressed','sft'),('compressed','assignment'),('compressed','relation'),('full','sft'))}
    cells=comparison.get('cells',[])
    if comparison.get('status')!='complete' or comparison.get('endpoint')!=512 or len(cells)!=8 or \
        {(c['arm'],c['method'],c['seed']) for c in cells}!=expected:
        raise ValueError('All eight original cells at the fixed512 endpoint required')
    results=comparison['comparisons']
    flags=[results[m][str(s)]['continuation_gate'] for m in ('assignment','relation') for s in (73,97)]
    if any(type(value) is not bool for value in flags):raise ValueError('Explicit original continuation decisions required')
    return not any(all(results[m][str(s)]['continuation_gate'] for s in (73,97)) for m in ('assignment','relation'))
