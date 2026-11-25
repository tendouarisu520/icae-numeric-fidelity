"""Bounded CPU collector after the GPU queue; no training or checkpoint selection."""
from common import *
from collections import defaultdict,Counter
import csv,math,re,sys,zipfile

def save_csv(path,data):
 with Path(path).open('x',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)

def main():
 environment();verify_lock();status=O/'analysis_status.json';started=time.time()
 assert not status.exists(),'Analysis already started; inspect before resuming'
 emit(status,phase='waiting_for_gpu_queue',deadline_seconds=24*3600)
 while True:
  q=read(O/'queue_status.json')
  if q['phase']=='failed':raise RuntimeError('GPU queue failed; preserve outputs and inspect before proceeding')
  if q['phase']=='complete':break
  if time.time()-started>24*3600:raise TimeoutError('24h bounded wait reached; inspect GPU queue')
  time.sleep(30)
 from binding.v13_protocol import metrics
 from binding.v16_code_fidelity import compare_codes,choice_decision
 from binding.v19_analysis import primary_contrasts,loss_components
 output=O/'final_analysis';output.mkdir(exist_ok=False);emit(status,phase='rescoring')
 pinpaths={R/'temp/v19/lock.json',Path(__file__),R/'temp/v19/common.py',O/'preflight.json'}
 def pinned(p):pinpaths.add(Path(p));return p
 comparisons=[];losses=[];summaries={}
 for method in ('sft','assignment','relation','bottleneck'):
  for seed in (73,97):
   folder=R/(f'runs/v16-repeated-exposure-20261106/main-compressed-sft-seed{seed}' if method=='sft' else f'runs/v19-budget-layout-20261108/extended-compressed-{method}-seed{seed}')
   summary=read(pinned(folder/'summary.json'));assert summary['status']=='success' and summary['optimizer_steps']==1024 and summary['step512_exact_replay']
   summaries[method,seed]=summary
   for step in (512,1024):
    ev=summary['evaluations'][str(step)];path=pinned(folder/f'evaluation-{step}/predictions.jsonl');assert sha(path)==ev['predictions_sha256'];raw=rows(path)
    for family in ('natural','probe'):
     actual=metrics([r for r in raw if r['family']==family]);assert actual==ev[family]
    comparisons.append(dict(method=method,seed=seed,step=step,target_joint=ev['probe']['rates']['target_joint'],target_count=ev['probe']['counts']['target_joint'][0],groups=ev['probe']['units'],unaffected_joint=ev['probe']['rates']['locality_joint'],controlled_accuracy=ev['probe']['rates']['known_exact'],natural_f1=ev['natural']['known_f1']))
   if method!='sft':
    steps=rows(pinned(folder/'steps.jsonl'));assert len(steps)==1024
    original=R/('runs/v14-bottleneck-reading-20261026' if method=='bottleneck' else 'runs/v13-answerable-reading-20261026')/f'main-compressed-{method}-seed{seed}'/'steps.jsonl'
    prior=rows(pinned(original));assert len(prior)==512
    for a,b in zip(steps[:512],prior):assert all(a[k]==b[k] for k in ('step','source_id','loss','families','gradient_norm','learning_rate'))
    losses.extend(dict(method=method,seed=seed,step=r['step'],source_id=r['source_id'],**loss_components(r)) for r in steps)
 criteria=[]
 for method in ('assignment','relation','bottleneck'):
  seeds=[]
  for seed in (73,97):
   a=next(x for x in comparisons if x['method']==method and x['seed']==seed and x['step']==1024);b=next(x for x in comparisons if x['method']=='sft' and x['seed']==seed and x['step']==1024)
   dt=a['target_joint']-b['target_joint'];df=a['natural_f1']-b['natural_f1'];du=a['unaffected_joint']-b['unaffected_joint'];seeds.append(dict(seed=seed,target_difference_pp=100*dt,natural_f1_difference_pp=100*df,unaffected_difference_pp=100*du,passes=dt>=.05-1e-12 and df>=-.02-1e-12 and du>=-.02-1e-12))
  criteria.append(dict(method=method,seeds=seeds,both_seeds_pass=all(s['passes'] for s in seeds)))
 balanced=[]
 for seed in (73,97):
  folder=R/f'runs/v19-budget-layout-20261108/balanced-compressed-sft-seed{seed}';summary=read(pinned(folder/'summary.json'));assert summary['status']=='success' and summary['optimizer_steps']==1024
  for step in (512,1024):
   current=O/f'balanced_layout/balanced-compressed-sft-seed{seed}-step{step}/crossed';receipt=read(pinned(current/'receipt.json'));prediction=pinned(current/'predictions.jsonl');assert sha(prediction)==receipt['predictions_sha256'];result=metrics(rows(prediction));assert result==receipt['metrics']
   base=R/f'results/v16-code-fidelity-20261106/roles/main-compressed-sft-seed{seed}-step{step}';br=read(pinned(base/'receipt.json'));bp=pinned(base/'predictions.jsonl');assert sha(bp)==br['predictions_sha256'];baseline=metrics(rows(bp));assert baseline==br['metrics']
   ev=summary['evaluations'][str(step)];original=next(x for x in comparisons if x['method']=='sft' and x['seed']==seed and x['step']==step)
   gain=result['rates']['known_exact']-baseline['rates']['known_exact'];dc=ev['probe']['rates']['known_exact']-original['controlled_accuracy'];df=ev['natural']['known_f1']-original['natural_f1']
   balanced.append(dict(seed=seed,step=step,crossed_correct=result['counts']['known_exact'][0],crossed_rows=result['rows'],crossed_gain_pp=100*gain,original_controlled_change_pp=100*dc,natural_f1_change_pp=100*df,passes=step==1024 and gain>=.1-1e-12 and dc>=-.02-1e-12 and df>=-.02-1e-12))
 prospect=[];slot=[];matrix=[];transitions=[]
 for job in read(pinned(O/'prospective/protocol.json'))['jobs']:
  for split in ('original','crossed'):
   units={u['id']:u for u in rows(pinned(R/f'data/v19-prospective-20261108/{split}.jsonl'))};folder=O/'prospective'/job['label']/split;r=read(pinned(folder/'receipt.json'));pred=pinned(folder/'predictions.jsonl');assert sha(pred)==r['predictions_sha256'];raw=rows(pred);assert metrics(raw)==r['metrics']
   scene=defaultdict(lambda:dict(target_n=0,choice_correct=0,greedy_correct=0,absent_n=0,digit_advantage_sum=0.));counts=Counter();by={}
   for row in raw:
    u=units[row['unit_id']];i,j=row['context_index'],row['question_index'];by[u['id'],i,j]=row;v=row['prediction'].strip();gold=u['answers'][i][j];assert gold==row['gold'];key=u['query_keys'][j];z=scene[u['source_id']]
    if split=='original':
     if j<2:
      scores=row['candidate_scores'];assert len(scores[0]['token_ids'])==len(scores[1]['token_ids']);decision=choice_decision([sum(s['token_logp'])/len(s['token_logp']) for s in scores]);assert decision==row['choice'];z['target_n']+=1;z['choice_correct']+=decision==j;z['greedy_correct']+=v==gold
     values=[f['value'] for f in u['facts'][i]]
     if re.fullmatch('[0-9]{4}',v) and v not in values:
      c=compare_codes(v,gold,values);z['absent_n']+=1;z['digit_advantage_sum']+=c['other_mean_hamming']-c['hamming']
    else:
     line=next(s for s in u['contexts'][i].splitlines() if s.startswith(f'Team {key[0]}, {key[1]} mode: '));fields=re.findall(r'\b(badge|locker|route) (\d{4})\b',line);requested=[f for f,c in fields].index(key[2])+1
     counts['n',i,j,key[2],requested]+=1;counts['correct',i,j,key[2],requested]+=v==gold
     matches=[pos+1 for pos,(f,c) in enumerate(fields) if c==v]
     if matches and v!=gold:
      rule=2 if key[2]=='locker' else 1;slot.append(dict(job=job['label'],unit_id=u['id'],source_id=u['source_id'],context=i,query=j,field=key[2],predicted_slot=matches[0],rule_slot=rule,follows_rule=matches[0]==rule))
   if split=='original':
    aggregates=[dict(source_id=k,**v) for k,v in sorted(scene.items())];actual=primary_contrasts(aggregates);assert actual==r['primary_contrasts'];prospect.append(dict(job=job['label'],arm=job['arm'],seed=job['seed'],primary=actual,metrics=r['metrics']))
    for j in (2,3):
     for i in (1,2,3):
      losses_=recoveries=0
      for uid in units:
       a,b=by[uid,0,j],by[uid,i,j];ac=a['prediction'].strip()==a['gold'];bc=b['prediction'].strip()==b['gold'];losses_+=ac and not bc;recoveries+=bc and not ac
      transitions.append(dict(job=job['label'],query=j,context=i,correct_to_wrong=losses_,wrong_to_correct=recoveries,net_change=recoveries-losses_))
   else:
    for k,n in counts.items():
     if k[0]=='n':matrix.append(dict(job=job['label'],context=k[1],query=k[2],field=k[3],slot=k[4],n=n,correct=counts[('correct',)+k[1:]]))
 save_csv(output/'matched_endpoint_comparison.csv',comparisons);save_csv(output/'extended_loss_components.csv',losses);save_csv(output/'balanced_layout_comparison.csv',balanced);save_csv(output/'prospective_field_by_slot.csv',matrix);save_csv(output/'prospective_bidirectional_transitions.csv',transitions)
 if slot:save_csv(output/'prospective_slot_rule_intrusions.csv',slot)
 report=dict(status='verified',auxiliary_1024_continuation=criteria,balanced=balanced,balanced_two_seed_pass=all(x['passes'] for x in balanced if x['step']==1024),prospective=prospect,prospective_compressed_both_seeds_pass=all(x['primary']['both_positive'] for x in prospect if x['arm']=='compressed'),inputs=[dict(path=p.relative_to(R).as_posix(),sha256=sha(p)) for p in sorted(pinpaths)],test_accessed=False,posthoc_old_cohorts=True,new_scene_analysis_fixed_before_inference=True)
 write(output/'final_report.json',report)
 # A portable evidence archive excludes model tensors but retains raw predictions and traces.
 files={p for p in pinpaths if p.suffix not in ('.safetensors','.pt')}|set(output.iterdir())
 archive=O/'completed_evidence.zip'
 with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
  for p in sorted(files):z.write(p,p.relative_to(R).as_posix())
 emit(status,phase='complete',archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),report=str(output/'final_report.json'))
if __name__=='__main__':
 try:main()
 except BaseException as e:emit(O/'analysis_status.json',phase='failed',error=str(e));raise
