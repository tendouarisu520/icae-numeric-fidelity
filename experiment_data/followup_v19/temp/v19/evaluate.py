"""Frozen reader evaluations: prospective original/crossed and balanced-layout controls."""
from common import *
import gc,math,re,sys
from collections import defaultdict
def main(mode):
 environment();verify_lock();pre=read(O/'preflight.json');assert pre['frozen_before_new_inference']
 from binding.repro_probe import configure_environment,configure_torch
 configure_environment('det_auto');configure_torch('det_auto')
 import torch
 from transformers import AutoTokenizer
 from safetensors.torch import load_file
 from binding.model import load_icae
 from binding.v11_training import load_model
 from binding.v12_reader import build_ram_cache,greedy
 from binding.v13_data import prepare
 from binding.v13_protocol import prediction_rows,metrics
 from binding.v13_objectives import prefix,logits
 from binding.v9_model import load_state,adapter_state
 from binding.v4_model import state_identity
 from binding.v16_code_fidelity import compare_codes,choice_decision
 from binding.v19_analysis import primary_contrasts
 assert torch.cuda.mem_get_info()[0]>=23*1024**3,'GPU occupied'
 dest=O/mode;dest.mkdir(exist_ok=False);status=dest/'status.json'
 if mode=='prospective':
  splits={k:rows(R/pre['datasets'][k]['path']) for k in ('original','crossed')}
  jobs=[dict(j,step=512,label=j['name']+'-step512') for j in pre['original_jobs'] if j['method']=='sft']
 else:
  assert mode=='balanced_layout'
  splits={'crossed':rows(R/'data/v16-role-balance-20261106/development.jsonl')};jobs=[]
  for seed in (73,97):
   run=R/f'runs/v19-budget-layout-20261108/balanced-compressed-sft-seed{seed}';summary=read(run/'summary.json');assert summary['status']=='success' and summary['optimizer_steps']==1024
   for step in (512,1024):
    checkpoint=run/('step512-adapter.safetensors' if step==512 else 'final-adapter.safetensors')
    jobs.append(dict(arm='compressed',seed=seed,step=step,label=f'balanced-compressed-sft-seed{seed}-step{step}',checkpoint=checkpoint.relative_to(R).as_posix()))
 protocol=dict(mode=mode,lock_sha256=sha(R/'temp/v19/lock.json'),data={k:sha(R/pre['datasets'][k]['path']) for k in splits} if mode=='prospective' else {'crossed':sha(R/'data/v16-role-balance-20261106/development.jsonl')},jobs=[dict(j,checkpoint_sha256=sha(R/j['checkpoint'])) for j in jobs],started_before_inference=time.time(),model_selection=False)
 write(dest/'protocol.json',protocol)
 tokenizer=AutoTokenizer.from_pretrained(str(R/'models/mistral'),local_files_only=True,use_fast=False)
 items={k:{u['id']:prepare(u,tokenizer) for u in us} for k,us in splits.items()}
 replay_unit=next(u for u in rows(R/'data/v13-answerable-relations-20261026/development.jsonl') if u['family']=='probe')
 replay_item=prepare(replay_unit,tokenizer)
 emit(status,phase='encoding',mode=mode)
 encoder,_=load_icae(str(R/'models/mistral'),str(R/'models/icae/mistral_7b_ft_icae.safetensors'),device='cuda',dtype=torch.bfloat16)
 encoder.requires_grad_(False);encoder.eval();ev={n:p._version for n,p in encoder.named_parameters()}
 cache=build_ram_cache(encoder,[c for sp in items.values() for it in sp.values() for c in it['contexts']]+replay_item['contexts'],lambda **v:emit(status,phase='encoding',**v))
 assert ev=={n:p._version for n,p in encoder.named_parameters()};cache_hash=cache.digest();del encoder;gc.collect();torch.cuda.empty_cache()
 model,_=load_model('full',73);model.eval();assert torch.equal(cache.ft,model.ft.detach().cpu())
 frozen={n:p._version for n,p in model.named_parameters() if not p.requires_grad}
 for job in protocol['jobs']:
  assert sha(R/job['checkpoint'])==job['checkpoint_sha256'];load_state(model,load_file(str(R/job['checkpoint']),device='cpu'))
  identity=state_identity(adapter_state(model))['sha256']
  if 'identity' in job:assert job['identity']==identity
  reference_path=(R/job['checkpoint']).parent/f'evaluation-{job["step"]}/predictions.jsonl'
  reference={(r['unit_id'],r['context_index'],r['question_index']):r for r in rows(reference_path)}
  with torch.no_grad():
   for i,j in ((0,0),(1,1),(0,2),(3,3)):
    pred,g=greedy(model,prefix(model,replay_item,i,j,cache,job['arm']),tokenizer);old=reference[replay_unit['id'],i,j]
    assert pred==old['prediction'] and g==old['generation'],'Historical greedy replay differs'
  for split,units in splits.items():
   out=dest/job['label']/split;out.mkdir(parents=True,exist_ok=False);allrows=[];scene=defaultdict(lambda:dict(target_n=0,choice_correct=0,greedy_correct=0,absent_n=0,digit_advantage_sum=0.));start=time.perf_counter()
   with torch.no_grad(),(out/'predictions.jsonl').open('x',encoding='utf8',buffering=1) as f:
    for number,u in enumerate(units,1):
     item=items[split][u['id']];predictions=[];extras={}
     for i in range(4):
      values=[]
      for j in range(4):
       p=prefix(model,item,i,j,cache,job['arm']);pred,g=greedy(model,p,tokenizer);values.append(pred);ex=dict(generation=g)
       if mode=='balanced_layout' and u['semantic_rotation']==u['layout_rotation']==u['target_team_position']==0:
        old=reference[u['base_unit_id'],i,j];assert pred==old['prediction'] and g==old['generation'],'Balanced identity replay differs'
       if j<2 and mode=='prospective' and split=='original':
        scores=[]
        for answer in item['answers'][i][:2]:
         output=logits(model,p,answer);truth=torch.tensor(answer,device=model.device);lp=output.log_softmax(-1).gather(-1,truth[:,None]).squeeze(-1)
         assert bool(torch.isfinite(lp).all());scores.append(dict(mean=float(lp.mean()),token_logp=lp.tolist(),token_ids=answer))
        assert len(scores[0]['token_ids'])==len(scores[1]['token_ids'])
        choice=choice_decision([s['mean'] for s in scores]);ex.update(candidate_values=u['answers'][i][:2],candidate_scores=scores,choice=choice,choice_correct=choice==j)
        z=scene[u['source_id']];z['target_n']+=1;z['choice_correct']+=choice==j;z['greedy_correct']+=pred.strip()==u['answers'][i][j]
       if mode=='prospective' and split=='original':
        vals=[v['value'] for v in u['facts'][i]];v=pred.strip()
        if re.fullmatch('[0-9]{4}',v) and v not in vals:
         c=compare_codes(v,u['answers'][i][j],vals);ex['absent_code']=c;z=scene[u['source_id']];z['absent_n']+=1;z['digit_advantage_sum']+=c['other_mean_hamming']-c['hamming']
       extras[i,j]=ex
      predictions.append(values)
     group=prediction_rows(u,predictions)
     for row in group:
      i,j=row['context_index'],row['question_index'];row.update(extras[i,j]);f.write(json.dumps(row,allow_nan=False)+'\n')
     allrows.extend(group)
     if number==1 or number%8==0 or number==len(units):emit(status,phase='evaluation',job=job['label'],split=split,units=number,total_units=len(units),seconds=time.perf_counter()-start)
   receipt=dict(metrics=metrics(allrows),rows=len(allrows),predictions_sha256=sha(out/'predictions.jsonl'),protocol_sha256=sha(dest/'protocol.json'),reader_identity=identity,historical_greedy_replay_rows=4,identity_replay_rows=128 if mode=='balanced_layout' else 0,wall_seconds=time.perf_counter()-start)
   if mode=='prospective' and split=='original':
    aggregate=[dict(source_id=k,**v) for k,v in sorted(scene.items())];write(out/'scene_aggregates.json',aggregate);receipt['primary_contrasts']=primary_contrasts(aggregate)
   write(out/'receipt.json',receipt)
  assert identity==state_identity(adapter_state(model))['sha256']
 assert frozen=={n:p._version for n,p in model.named_parameters() if not p.requires_grad} and cache.digest()==cache_hash
 emit(status,phase='complete',mode=mode,jobs=len(jobs))
if __name__=='__main__':
 try:main(sys.argv[1])
 except BaseException as e:
  emit(O/sys.argv[1]/'status.json',phase='failed',error=str(e));raise
