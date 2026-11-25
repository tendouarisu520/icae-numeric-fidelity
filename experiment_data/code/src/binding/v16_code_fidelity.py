"""Digit fidelity diagnostics. Never repairs a prediction using its gold label."""
from collections import Counter
import math
import re

def choice_decision(scores):
    if len(scores)!=2 or not all(math.isfinite(v) for v in scores):
        raise ValueError('Two finite scores required')
    return None if scores[0]==scores[1] else int(scores[1]>scores[0])

def hamming(a,b):
    return sum(x!=y for x,y in zip(a,b))

def compare_codes(prediction,gold,values):
    if not all(re.fullmatch('[0-9]{4}',v) for v in [prediction,gold,*values]):
        raise ValueError('Exactly four ASCII digits required; no normalization or padding')
    if len(values)!=len(set(values)) or gold not in values or len(values)<2:
        raise ValueError('Unique record values including gold required')
    distance=hamming(prediction,gold)
    other=[hamming(prediction,v) for v in values if v!=gold]
    minimum=min(distance,min(other))
    swapped=[]
    for i in range(3):
        chars=list(gold);chars[i],chars[i+1]=chars[i+1],chars[i]
        if ''.join(chars)==prediction and prediction!=gold:swapped.append(i)
    return dict(hamming=distance,matched_positions=4-distance,
        match_mask=''.join('=' if a==b else 'X' for a,b in zip(prediction,gold)),
        digit_overlap=sum((Counter(prediction)&Counter(gold)).values()),
        same_digit_multiset=Counter(prediction)==Counter(gold),adjacent_swap=bool(swapped),
        prefix_length=next((i for i,(a,b) in enumerate(zip(prediction,gold)) if a!=b),4),
        nearest_other_hamming=min(other),other_mean_hamming=sum(other)/len(other),
        gold_at_minimum=distance==minimum,gold_unique_nearest=distance<min(other),
        gold_better_than_other_fraction=sum(distance<v for v in other)/len(other),
        gold_equal_other_fraction=sum(distance==v for v in other)/len(other),
        null_at_minimum_fraction=sum(v==minimum for v in other)/len(other))
