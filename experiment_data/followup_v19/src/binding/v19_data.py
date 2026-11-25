"""Fixed exposure-balanced training and new synthetic scenes; no test data."""
from copy import deepcopy
from .v13_data import FIELDS,make_probe,digest
from .v16_roles import render

def transform(unit,semantic,layout,exchange):
 if unit['family']!='probe' or unit['level'] not in (2,3,4):raise ValueError('Controlled unit required')
 if semantic not in range(3) or layout not in range(3) or exchange not in range(2):raise ValueError('Unknown factor cell')
 background=[s[12:] for s in unit['contexts'][0].splitlines() if s.startswith('Background: ')]
 background=background[0] if background else ''
 teams=list(unit['entities']);modes=list(unit['modes'])
 fm=dict(zip(FIELDS,FIELDS[semantic:]+FIELDS[:semantic]));order=FIELDS[layout:]+FIELDS[:layout]
 tm={t:t for t in teams}
 if exchange:tm[teams[0]],tm[teams[1]]=teams[1],teams[0]
 u=deepcopy(unit);u['base_unit_id']=unit['id'];u['id']=unit['id']+f'-v19-s{semantic}-p{layout}-e{exchange}'
 u.update(semantic_rotation=semantic,layout_rotation=layout,target_team_position=exchange,rendered_field_order=list(order))
 u['query_keys']=[[tm[t],m,fm[f]] for t,m,f in unit['query_keys']]
 u['queries']=[f'For team {t} in {m} mode, what is the {f} code?' for t,m,f in u['query_keys']]
 values=[]
 for i,facts in enumerate(unit['facts']):
  row=[dict(key=[tm[t],m,fm[f]],value=v['value']) for v in facts for t,m,f in [v['key']]]
  row.sort(key=lambda v:(teams.index(v['key'][0]),modes.index(v['key'][1]),order.index(v['key'][2])),reverse=i==3)
  assert [dict((tuple(f['key']),f['value']) for f in row)[tuple(k)] for k in u['query_keys']]==u['answers'][i]
  values.append(row)
 u['facts']=values;u['contexts']=[render(v,background,order) for v in values]
 return u

def balanced_training(probes):
 if len(probes)!=512 or len({u['source_id'] for u in probes})!=512:raise ValueError('Exactly512 source probes required')
 ordered=sorted(probes,key=lambda u:digest('v19-balance:'+u['source_id']));out=[]
 for epoch in range(2):
  for k,u in enumerate(ordered):
   cell=(epoch*512+k)%18;s,p,e=cell//6,(cell//2)%3,cell%2
   v=transform(u,s,p,e);v['id']+=f'-exposure{epoch}';v['exposure']=epoch;out.append(v)
 return out

def prospective_units():
 ordinary=[];crossed=[]
 for i in range(32):
  uid=f'v19-prospective-20261108-{i:03d}'
  source=dict(id=uid,article=uid,contexts=['This background is not evidence for the codes in the records.'])
  for level in (2,3,4):
   u=make_probe(source,'development',level);u['cohort']='prospective_synthetic';ordinary.append(u)
   if i<8 and level==4:
    crossed.extend(transform(u,s,p,e) for s in range(3) for p in range(3) for e in range(2))
 return ordinary,crossed
