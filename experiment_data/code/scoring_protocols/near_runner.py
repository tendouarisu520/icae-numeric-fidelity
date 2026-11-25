"""Prespecified one-digit-foil and greedy-sequence-likelihood diagnostics."""
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
R=Path('E:/Research/ICAE-ConditionalMemory-20261021')
O=R/'results/v16-code-fidelity-20261106/near'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def emit(**v):
    v=dict(time=time.time(),pid=os.getpid(),**v);print(json.dumps(v),flush=True)
    (O/'status.json').write_text(json.dumps(v),encoding='utf8')
def main():
    O.mkdir(parents=True,exist_ok=True);os.chdir(R)
    for name,sub in {'TEMP':'temp','TMP':'temp','HF_HOME':'cache/huggingface','TORCH_HOME':'cache/torch','CUDA_CACHE_PATH':'cache/cuda','TRITON_CACHE_DIR':'cache/triton','TORCHINDUCTOR_CACHE_DIR':'cache/torchinductor','HF_DATASETS_CACHE':'cache/datasets','HUGGINGFACE_HUB_CACHE':'cache/huggingface/hub','XDG_CACHE_HOME':'cache','NUMBA_CACHE_DIR':'cache/numba'}.items():os.environ[name]=str(R/sub)
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
    sys.path.insert(0,str(R/'src'))
    lock=read(R/'temp/v16/near_lock.json')
    for name,pin in lock['files'].items():assert sha(R/name)==pin,name
    # This decision and foil construction are fixed before their model scores are observed.
    while True:
        state=read(O.parent/'roles/status.json')
        if state['phase']=='failed':raise RuntimeError('Role diagnostic failed; inspect before near-foil run')
        if state['phase']=='complete':break
        emit(phase='waiting_for_roles',upstream_phase=state['phase'])
        time.sleep(20)
    from binding.repro_probe import configure_environment,configure_torch
    configure_environment('det_auto');configure_torch('det_auto')
    import torch
    deadline=time.time()+120
    while torch.cuda.mem_get_info()[0]<23*1024**3:
        if time.time()>deadline:raise RuntimeError('GPU not free')
        time.sleep(5)
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from binding.model import load_icae
    from binding.v11_pipeline import read_rows,write_json,em_f1
    from binding.v11_training import load_model
    from binding.v12_reader import build_ram_cache,greedy
    from binding.v13_data import prepare
    from binding.v13_objectives import prefix,logits
    from binding.v16_near_codes import near_code
    from binding.v9_model import load_state,adapter_state
    from binding.v4_model import state_identity
    paths={'development':R/'data/v13-answerable-relations-20261026/development.jsonl',
        'confirmation':R/'results/confirmation-20261104/confirmation_units.jsonl'}
    units={s:[u for u in read_rows(p) if u['family']=='probe'] for s,p in paths.items()}
    original=read(R/'results/confirmation-20261104/protocol.json')
    jobs=[dict(j,step=512,label=j['name']+'-step512',splits=['development','confirmation']) for j in original['jobs'] if j['method']=='sft']
    for seed in (73,97):
        run=f'runs/v16-repeated-exposure-20261106/main-compressed-sft-seed{seed}';summary=read(R/run/'summary.json')
        assert summary['step512_exact_replay']
        jobs.append(dict(name=f'main-compressed-sft-seed{seed}',label=f'main-compressed-sft-seed{seed}-step1024',
            arm='compressed',seed=seed,method='sft',step=1024,checkpoint=run+'/final-adapter.safetensors',identity=summary['final_reader_identity'],splits=['development']))
    cfg=dict(date='2026-11-06',runner_sha256=sha(Path(__file__)),lock_sha256=sha(R/'temp/v16/near_lock.json'),
        data={s:sha(p) for s,p in paths.items()},jobs=[dict(j,checkpoint_sha256=sha(R/j['checkpoint'])) for j in jobs],
        scoring='Mean answer-plus-EOS log probability; equal-length codes; exact ties fail strict accuracy',
        foils='One out-of-record single-digit alternative at first or last position, fixed by row hash, identical across readers',
        generated_sequence_score='Original generated token IDs including EOS; report sum and mean separately; no output replaced',
        scope='Post hoc diagnostic; no model selection; extended1024 readers evaluated on development only',reserved_test_accessed=False)
    assert not (O/'protocol.json').exists();write_json(O/'protocol.json',cfg)
    tok=AutoTokenizer.from_pretrained(str(R/'models/mistral'),local_files_only=True,use_fast=False)
    items={s:{u['id']:prepare(u,tok) for u in v} for s,v in units.items()}
    enc,_=load_icae(str(R/'models/mistral'),str(R/'models/icae/mistral_7b_ft_icae.safetensors'),device='cuda',dtype=torch.bfloat16)
    enc.requires_grad_(False);enc.eval()
    cache=build_ram_cache(enc,[c for vs in items.values() for item in vs.values() for c in item['contexts']],lambda **v:emit(phase='encoding',**v))
    cache_sha=cache.digest();del enc;gc.collect();torch.cuda.empty_cache()
    model,_=load_model('full',73);model.eval();assert torch.equal(model.ft.detach().cpu(),cache.ft)
    def score(p,answer):
        output=logits(model,p,answer);truth=torch.tensor(answer,device=model.device)
        lp=output.log_softmax(-1).gather(-1,truth.unsqueeze(-1)).squeeze(-1)
        assert bool(torch.isfinite(lp).all())
        return dict(mean=float(lp.mean()),sum=float(lp.sum()),tokens=len(answer),token_logp=lp.tolist())
    for job in jobs:
        load_state(model,load_file(str(R/job['checkpoint']),device='cpu'));assert state_identity(adapter_state(model))['sha256']==job['identity']
        for split in job['splits']:
            dest=O/split/job['label'];dest.mkdir(parents=True,exist_ok=False)
            rawpath=(R/job['checkpoint']).parent/f'evaluation-{job["step"]}/predictions.jsonl' if split=='development' else R/'results/confirmation-20261104'/job['name']/'predictions.jsonl'
            raw={(r['unit_id'],r['context_index'],r['question_index']):r for r in read_rows(rawpath) if r['family']=='probe'}
            old_choices={}
            if job['step']==512:
                old_dir=O.parent/'choice'/split/job['name'];old_receipt=read(old_dir/'receipt.json')
                assert sha(rawpath)==old_receipt['raw_predictions_sha256']
                assert sha(old_dir/'scores.jsonl')==old_receipt['scores_sha256']
                old_choices={(r['unit_id'],r['context_index'],r['question_index']):r for r in read_rows(old_dir/'scores.jsonl')}
            rows=[];start=time.perf_counter();replayed=0;max_replay_score_difference=0.
            with torch.no_grad(),(dest/'scores.jsonl').open('x',encoding='utf8',buffering=1) as f:
                for number,u in enumerate(units[split],1):
                    item=items[split][u['id']]
                    for i in range(4):
                        values=[v['value'] for v in u['facts'][i]]
                        for j in (0,1):
                            old=raw[u['id'],i,j];p=prefix(model,item,i,j,cache,job['arm']);gold=item['answers'][i][j]
                            if number==1 and i==0:
                                pred,g=greedy(model,p,tok);assert pred==old['prediction'] and g==old['generation'];replayed+=1
                            memo={}
                            def cached(ids):
                                key=tuple(ids)
                                if key not in memo:memo[key]=score(p,ids)
                                return memo[key]
                            gscore=cached(gold);condition=cached(item['answers'][i][1-j]);near={}
                            if old_choices:
                                previous=old_choices[u['id'],i,j]
                                difference=max(abs(gscore['mean']-previous['candidate_scores'][j]['mean']),abs(condition['mean']-previous['candidate_scores'][1-j]['mean']))
                                max_replay_score_difference=max(max_replay_score_difference,difference)
                                assert difference<=1e-5 and (gscore['mean']>condition['mean'])==previous['choice_correct'],'Candidate score replay differs'
                            for pos in (0,3):
                                foil=near_code(old['gold'],values,f'{u["id"]}:{i}:{j}',pos)
                                ids=tok.encode(' '+foil,add_special_tokens=False)+[tok.eos_token_id]
                                assert len(ids)==len(gold)
                                near[str(pos)]=dict(value=foil,score=cached(ids))
                            generated=cached(old['generation']['token_ids'])
                            row=dict(unit_id=u['id'],source_id=u['source_id'],article=u['article'],level=u['level'],context_index=i,question_index=j,
                                gold=old['gold'],prediction=old['prediction'],greedy_correct=bool(em_f1(old['prediction'],old['gold'])[0]),
                                gold_score=gscore,condition_foil_value=u['answers'][i][1-j],condition_foil_score=condition,
                                near=near,generated_score=generated)
                            f.write(json.dumps(row,allow_nan=False)+'\n');rows.append(row)
                    if number==1 or number%16==0 or number==len(units[split]):emit(phase='scoring',job=job['label'],split=split,units=number,total_units=len(units[split]),seconds=time.perf_counter()-start)
            correct=lambda r,s:r['gold_score']['mean']>s['mean']
            result=dict(job=job['label'],split=split,rows=len(rows),condition_correct=sum(correct(r,r['condition_foil_score']) for r in rows),
                first_digit_correct=sum(correct(r,r['near']['0']['score']) for r in rows),last_digit_correct=sum(correct(r,r['near']['3']['score']) for r in rows),
                greedy_correct=sum(r['greedy_correct'] for r in rows),wrong_generation_gold_more_likely=sum(not r['greedy_correct'] and r['gold_score']['sum']>r['generated_score']['sum'] for r in rows),
                exact_replay_rows=replayed,max_candidate_score_replay_difference=max_replay_score_difference,
                scores_sha256=sha(dest/'scores.jsonl'),raw_predictions_sha256=sha(rawpath),protocol_sha256=sha(O/'protocol.json'))
            write_json(dest/'receipt.json',result);emit(phase='cell_complete',**result)
    assert cache.digest()==cache_sha
    emit(phase='complete',cells=10)
if __name__=='__main__':
    try:main()
    except BaseException as error:emit(phase='failed',error=str(error));raise
