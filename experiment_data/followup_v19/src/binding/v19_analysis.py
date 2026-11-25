"""Analysis fixed before new-scene inference; bootstrap at independent scene level."""
import math,random

def loss_components(row):
 families=row['families'];assert len(families)==2
 ce=math.fsum(.5*v['sft_loss'] for v in families)
 auxiliary=math.fsum(.1*v.get('bottleneck_loss',v.get('assignment_loss',0.)) for v in families)
 local=math.fsum(.025*v.get('locality_loss',0.) for v in families)
 total=ce+auxiliary+local
 if not math.isclose(total,row['loss'],rel_tol=1e-6,abs_tol=1e-6):raise ValueError('Weighted losses do not reproduce logged total')
 if not math.isclose(math.fsum(v['loss'] for v in families),row['loss'],rel_tol=1e-6,abs_tol=1e-6):raise ValueError('Family total mismatch')
 return dict(weighted_ce=ce,weighted_aux=auxiliary,weighted_locality=local,total_loss=row['loss'],natural_ce=families[0]['sft_loss'],probe_ce=families[1]['sft_loss'],raw_aux=families[1].get('bottleneck_loss',families[1].get('assignment_loss',0.)),natural_locality=families[0].get('locality_loss',0.),probe_locality=families[1].get('locality_loss',0.))

def primary_contrasts(scene_rows,resamples=10000):
 if not scene_rows or len({r['source_id'] for r in scene_rows})!=len(scene_rows):raise ValueError('One aggregate per source scene required')
 def calc(rows):
  n=sum(r['target_n'] for r in rows);a=sum(r['absent_n'] for r in rows)
  return ((sum(r['choice_correct']-r['greedy_correct'] for r in rows)/n if n else None),sum(r['digit_advantage_sum'] for r in rows)/a if a else None)
 point=calc(scene_rows);rng=random.Random(20261108);draws=[[],[]]
 for _ in range(resamples):
  v=calc([scene_rows[rng.randrange(len(scene_rows))] for _ in scene_rows])
  for j in (0,1):
   if v[j] is not None:draws[j].append(v[j])
 def quantile(a,p):
  a=sorted(a);x=(len(a)-1)*p;i=int(x);return a[i]*(1-(x-i))+a[min(i+1,len(a)-1)]*(x-i)
 result={}
 for j,key in enumerate(('choice_generation_gap','absent_digit_advantage')):
  # Undefined bootstrap draws must not be silently discarded.
  valid=len(draws[j])==resamples and point[j] is not None
  bounds=[quantile(draws[j],.00625),quantile(draws[j],.99375)] if valid else [None,None]
  result[key]=dict(estimate=point[j],interval_98_75=bounds,valid_resamples=len(draws[j]),positive_lower_bound=valid and bounds[0]>0)
 result.update(both_positive=all(result[k]['positive_lower_bound'] for k in ('choice_generation_gap','absent_digit_advantage')),scenes=len(scene_rows),resamples=resamples,seed=20261108)
 return result
