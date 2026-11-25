"""Factorial field/layout/team diagnostic after the exposure queue completes."""
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
R=Path('E:/Research/ICAE-ConditionalMemory-20261021')
O=R/'results/v16-code-fidelity-20261106/roles'
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
    lock=read(R/'temp/v16/roles_lock.json')
    for name,pin in lock['files'].items():assert sha(R/name)==pin,name
    # Data and factor matrix are locked while the earlier jobs are still running.
    data=R/'data/v16-role-balance-20261106';cfg=read(data/'protocol.json')
    assert sha(data/'development.jsonl')==cfg['data_sha256']==lock['data_sha256']
    queue=O.parent/'exposure_queue_status.json'
    while True:
        state=read(queue)
        if state['phase']=='failed':raise RuntimeError('Exposure failed; inspect before role evaluation')
        if state['phase']=='complete':break
        emit(phase='waiting_for_exposure',upstream_phase=state['phase'])
        time.sleep(20)
    from binding.repro_probe import configure_environment,configure_torch
    configure_environment('det_auto');configure_torch('det_auto')
    import torch
    deadline=time.time()+120
    while torch.cuda.mem_get_info()[0]<23*1024**3:
        if time.time()>deadline:raise RuntimeError('GPU occupied; never stop another task')
        time.sleep(5)
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from binding.model import load_icae
    from binding.v11_pipeline import read_rows,write_json
    from binding.v11_training import load_model
    from binding.v12_reader import build_ram_cache,greedy
    from binding.v13_protocol import metrics,prediction_rows
    from binding.v13_data import prepare
    from binding.v13_objectives import prefix
    from binding.v9_model import load_state,adapter_state
    from binding.v4_model import state_identity
    units=read_rows(data/'development.jsonl');assert len(units)==144
    tokenizer=AutoTokenizer.from_pretrained(str(R/'models/mistral'),local_files_only=True,use_fast=False)
    items={u['id']:prepare(u,tokenizer) for u in units}
    original=read(R/'results/confirmation-20261104/protocol.json')
    jobs=[]
    for j in original['jobs']:
        if j['method']=='sft':jobs.append(dict(j,step=512,label=j['name']+'-step512'))
    for seed in (73,97):
        run=f'runs/v16-repeated-exposure-20261106/main-compressed-sft-seed{seed}'
        summary=read(R/run/'summary.json')
        assert summary['step512_exact_replay'] and summary['optimizer_steps']==1024
        jobs.append(dict(name=f'main-compressed-sft-seed{seed}',label=f'main-compressed-sft-seed{seed}-step1024',arm='compressed',seed=seed,method='sft',step=1024,checkpoint=run+'/final-adapter.safetensors',identity=summary['final_reader_identity']))
    protocol=dict(date='2026-11-06',purpose=cfg['purpose'],runner_sha256=sha(Path(__file__)),data_sha256=sha(data/'development.jsonl'),
        generator_protocol_sha256=sha(data/'protocol.json'),lock_sha256=sha(R/'temp/v16/roles_lock.json'),
        jobs=[dict(j,checkpoint_sha256=sha(R/j['checkpoint'])) for j in jobs],reserved_test_accessed=False,model_selection=False)
    assert not (O/'protocol.json').exists(),'Role study already initialized; audit before retry'
    write_json(O/'protocol.json',protocol)
    emit(phase='encoding')
    enc,_=load_icae(str(R/'models/mistral'),str(R/'models/icae/mistral_7b_ft_icae.safetensors'),device='cuda',dtype=torch.bfloat16)
    enc.requires_grad_(False);enc.eval()
    cache=build_ram_cache(enc,[c for item in items.values() for c in item['contexts']],lambda **v:emit(phase='encoding',**v))
    cache_sha=cache.digest();del enc;gc.collect();torch.cuda.empty_cache()
    model,_=load_model('full',73);model.eval();assert torch.equal(model.ft.detach().cpu(),cache.ft)
    for job in jobs:
        dest=O/job['label'];dest.mkdir(exist_ok=False)
        load_state(model,load_file(str(R/job['checkpoint']),device='cpu'))
        assert state_identity(adapter_state(model))['sha256']==job['identity']
        reference_path=(R/job['checkpoint']).parent/f'evaluation-{job["step"]}/predictions.jsonl'
        reference={(r['unit_id'],r['context_index'],r['question_index']):r for r in read_rows(reference_path)}
        rows=[];replayed=0;start=time.perf_counter()
        with torch.no_grad(),(dest/'predictions.jsonl').open('x',encoding='utf8',buffering=1) as f:
            for number,u in enumerate(units,1):
                item=items[u['id']];predictions=[];generations={}
                for i in range(4):
                    values=[]
                    for j in range(4):
                        pred,g=greedy(model,prefix(model,item,i,j,cache,job['arm']),tokenizer)
                        if u['semantic_rotation']==u['layout_rotation']==u['target_team_position']==0:
                            old=reference[u['base_unit_id'],i,j]
                            assert pred==old['prediction'] and g==old['generation'],'Identity view replay differs'
                            replayed+=1
                        values.append(pred);generations[i,j]=g
                    predictions.append(values)
                group=prediction_rows(u,predictions)
                for row in group:
                    i,j=row['context_index'],row['question_index'];key=u['query_keys'][j]
                    teams=list(dict.fromkeys(v['key'][0] for v in u['facts'][i]))
                    row.update(generation=generations[i,j],base_unit_id=u['base_unit_id'],
                        query_field=key[2],field_position=u['rendered_field_order'].index(key[2]),team_position=teams.index(key[0]),
                        semantic_rotation=u['semantic_rotation'],layout_rotation=u['layout_rotation'],target_team_role=u['target_team_position'])
                    f.write(json.dumps(row)+'\n')
                rows.extend(group)
                if number==1 or number%12==0 or number==len(units):emit(phase='evaluating',job=job['label'],units=number,total_units=144,seconds=time.perf_counter()-start)
        assert len(rows)==2304 and replayed==128
        result=dict(job=job['label'],metrics=metrics(rows),exact_identity_replay_rows=replayed,
            predictions_sha256=sha(dest/'predictions.jsonl'),protocol_sha256=sha(O/'protocol.json'),wall_seconds=time.perf_counter()-start)
        write_json(dest/'receipt.json',result);emit(phase='cell_complete',job=job['label'],correct=result['metrics']['counts']['known_exact'])
    assert cache.digest()==cache_sha
    emit(phase='complete',jobs=len(jobs))
if __name__=='__main__':
    try:main()
    except BaseException as error:emit(phase='failed',error=str(error));raise
