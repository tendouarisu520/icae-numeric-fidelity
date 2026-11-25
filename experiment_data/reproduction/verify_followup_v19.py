"""Check the V19 follow-up files shipped in followup_v19/. Standard library only.

This check covers the exported loss components, the new data files and the arithmetic of the verified report.
verify_followup_v19_raw.py recomputes the results from the per-row predictions in followup_v19/evidence/.
"""
from pathlib import Path
import collections,csv,hashlib,json
P=Path(__file__).resolve().parents[1]
F=P/'followup_v19'
V=F/'results/v19-budget-layout-20261108'
FS=V/'final_status_20261112'
def rows(p):return [json.loads(x) for x in p.read_text(encoding='utf8').splitlines() if x.strip()]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
checks=[]
# 1. loss export: components add up, six runs of 512 updates, window summary reproduces
table=list(csv.DictReader((V/'losses/auxiliary_losses.csv').open(encoding='utf8')))
assert sha(V/'losses/auxiliary_losses.csv')==json.loads((V/'losses/manifest.json').read_text(encoding='utf8'))['csv_sha256']
by=collections.defaultdict(list)
for r in table:
 by[r['job']].append(r)
 assert abs(float(r['weighted_ce'])+float(r['weighted_aux'])+float(r['weighted_locality'])-float(r['total_loss']))<1e-9
assert len(by)==6 and all(len(v)==512 for v in by.values())
mean=lambda xs:sum(xs)/len(xs)
drops=[];shares=[];ldrops=[];lshares=[]
for job,rs in by.items():
 rs.sort(key=lambda r:int(r['step']));assert [int(r['step']) for r in rs]==list(range(1,513))
 aux=[float(r['weighted_aux']) for r in rs];ce=[float(r['weighted_ce']) for r in rs]
 drops.append(mean(aux[:128])/mean(aux[384:]));shares.append(mean(aux[384:])/mean(ce[384:]))
 loc=[float(r['weighted_locality']) for r in rs]
 if any(loc):ldrops.append(mean(loc[:128])/mean(loc[384:]));lshares.append(mean(loc[384:])/mean(ce[384:]))
for s in csv.DictReader((V/'losses/loss_window_summary.csv').open(encoding='utf8')):
 if s['component']!='weighted_aux':continue
 rs=by[f"main-compressed-{s['method']}-seed{s['seed']}"];aux=[float(r['weighted_aux']) for r in rs]
 assert abs(float(s['first128_mean'])-mean(aux[:128]))<1e-12 and abs(float(s['last128_mean'])-mean(aux[384:]))<1e-12
assert round(min(drops),1)==7.0 and round(max(drops),1)==8.6 and round(100*min(shares),1)==1.8 and round(100*max(shares),1)==4.1
assert len(ldrops)==4 and 1.5<=min(ldrops) and max(ldrops)<1.6 and round(100*min(lshares),1)==2.5 and round(100*max(lshares),1)==2.7
checks.append('Auxiliary loss export: components add up in 3,072 updates; the assignment or weakest-margin term falls 7.0 to 8.6 times and ends at 1.8% to 4.1% of the answer loss; the locality term falls about 1.5 times and ends at 2.5% to 2.7%')
# 2. new scenes: sizes, background, no overlap with earlier units, pools
orig=rows(F/'data/v19-prospective-20261108/original.jsonl');crossed=rows(F/'data/v19-prospective-20261108/crossed.jsonl');balanced=rows(F/'data/v19-prospective-20261108/balanced_training.jsonl')
lock=json.loads((F/'temp/v19/lock.json').read_text(encoding='utf8'))
for name,path in (('original','original.jsonl'),('crossed','crossed.jsonl'),('balanced_training','balanced_training.jsonl')):
 assert sha(F/'data/v19-prospective-20261108'/path)==lock['generated_data_sha256'][name]
earlier=[]
for p in (P/'data/v13-answerable-relations-20261026/train.jsonl',P/'data/v13-answerable-relations-20261026/development.jsonl',P/'confirmation/confirmation_units.jsonl',P/'data/v16-role-balance-20261106/development.jsonl'):
 earlier+=[u for u in rows(p) if 'probe' in u['id'] or u.get('family')=='probe']
ctx_old={c for u in earlier for c in u['contexts']};rec=lambda u:frozenset((tuple(f['key']),f['value']) for f in u['facts'][0]);rec_old={rec(u) for u in earlier if 'facts' in u}
assert len(orig)==96 and len({u['source_id'] for u in orig})==32 and sorted(collections.Counter(u['level'] for u in orig).values())==[32,32,32]
assert len(crossed)==144 and len({u['source_id'] for u in crossed})==8
assert {x for u in orig for c in u['contexts'] for x in c.splitlines() if x.startswith('Background')}=={'Background: This background is not evidence for the codes in the records.'}
assert not any(c in ctx_old for u in orig+crossed for c in u['contexts']) and not any(rec(u) in rec_old for u in orig)
train=[u for u in rows(P/'data/v13-answerable-relations-20261026/train.jsonl') if 'probe' in u['id']];dev=[u for u in rows(P/'data/v13-answerable-relations-20261026/development.jsonl') if 'probe' in u['id']]
codes=lambda us:{f['value'] for u in us for fs in u['facts'] for f in fs};names=lambda us:{f['key'][0] for u in us for fs in u['facts'] for f in fs}
assert not codes(orig)&codes(train) and names(orig)<=names(dev)
cells=collections.Counter((u['semantic_rotation'],u['layout_rotation'],u['target_team_position']) for u in balanced);uses=collections.Counter(u['source_id'] for u in balanced)
assert len(balanced)==1024 and len(uses)==512 and set(uses.values())=={2} and len(cells)==18 and min(cells.values())==56 and max(cells.values())==57
checks.append('New scenes: 96 units from 32 scenes, 144 crossed views, neutral background, no context or record shared with earlier units, development name and code pools; balanced training uses 18 views 56 or 57 times each')
# 3. verified report arithmetic and agreement with earlier tables
report=json.loads((FS/'final_report.json').read_text(encoding='utf8'));assert report['status']=='verified' and report['test_accessed'] is False and report['new_scene_analysis_fixed_before_inference']
mec=list(csv.DictReader((FS/'matched_endpoint_comparison.csv').open(encoding='utf8')))
get=lambda m,s,st:next(r for r in mec if r['method']==m and int(r['seed'])==s and int(r['step'])==st)
for m,seed,step,t in (('sft',73,512,54),('sft',97,512,63),('assignment',73,512,53),('assignment',97,512,58),('relation',73,512,50),('relation',97,512,56),('bottleneck',73,512,57),('bottleneck',97,512,61),('sft',73,1024,61),('sft',97,1024,61)):
 assert int(get(m,seed,step)['target_count'])==t
for c in report['auxiliary_1024_continuation']:
 for s in c['seeds']:
  dt=100*(int(get(c['method'],s['seed'],1024)['target_count'])-int(get('sft',s['seed'],1024)['target_count']))/96
  assert abs(dt-s['target_difference_pp'])<1e-9 and s['passes']==(dt>=5-1e-9 and s['natural_f1_difference_pp']>=-2-1e-9 and s['unaffected_difference_pp']>=-2-1e-9)
for r in csv.DictReader((FS/'balanced_layout_comparison.csv').open(encoding='utf8')):
 base={73:387,97:369}[int(r['seed'])]
 assert abs(100*(int(r['crossed_correct'])-base)/2304-float(r['crossed_gain_pp']))<1e-9
tot={t['seed']:t for t in json.loads((FS/'prospective_totals.json').read_text(encoding='utf8'))}
for p in report['prospective']:
 if p['arm']=='compressed':
  t=tot[p['seed']];assert abs((t['choice_correct']-t['greedy_correct'])/t['target_n']-p['primary']['choice_generation_gap']['estimate'])<1e-12 and p['primary']['both_positive']
 else:
  assert p['metrics']['counts']['known_exact']==[1536,1536] and p['primary']['absent_digit_advantage']['valid_resamples']==0
checks.append('Verified V19 report: 512-update values match the fixed 512-update endpoint table, 1024-update SFT matches the learning-curve table, crossed baselines match the layout audit, and every pass/fail flag follows from its numbers')
result=dict(status='pass',checks=checks,per_row_predictions_local=True)
(P/'reproduction/followup_v19_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))
