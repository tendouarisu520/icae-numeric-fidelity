"""A small frozen-memory replacement control after the existing near-code queue."""
import gc,hashlib,io,json,lzma,os,tarfile,time
from pathlib import Path
import sys
R=Path('E:/Research/ICAE-ConditionalMemory-20261021');O=R/'results/v16-code-fidelity-20261106/memory_control'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def emit(**v):
    v=dict(pid=os.getpid(),time=time.time(),**v);print(json.dumps(v),flush=True);(O/'status.json').write_text(json.dumps(v),encoding='utf8')
def main():
    O.mkdir(parents=True,exist_ok=False);os.chdir(R)
    for name,sub in {'TEMP':'temp','TMP':'temp','HF_HOME':'cache/huggingface','TORCH_HOME':'cache/torch','CUDA_CACHE_PATH':'cache/cuda','TRITON_CACHE_DIR':'cache/triton','TORCHINDUCTOR_CACHE_DIR':'cache/torchinductor','HF_DATASETS_CACHE':'cache/datasets','HUGGINGFACE_HUB_CACHE':'cache/huggingface/hub','XDG_CACHE_HOME':'cache','NUMBA_CACHE_DIR':'cache/numba'}.items():os.environ[name]=str(R/sub)
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
    lock=read(R/'temp/v16/memory_control_lock.json')
    for name,pin in lock['files'].items():assert sha(R/name)==pin,name
    plan=read(R/'temp/v16/memory_control_plan.json');data=R/'data/v13-answerable-relations-20261026/development.jsonl';assert sha(data)==plan['source_sha256']
    deadline=time.time()+7200
    while True:
        try:s=read(O.parent/'near/status.json')
        except json.JSONDecodeError:time.sleep(1);continue
        if s['phase']=='failed':raise RuntimeError('Near-code evaluation failed; inspect before continuation')
        if s['phase']=='complete':break
        if time.time()>deadline:raise RuntimeError('Near-code wait exceeded2h')
        emit(phase='waiting_for_near',upstream_phase=s['phase']);time.sleep(20)
    sys.path.insert(0,str(R/'src'))
    from binding.repro_probe import configure_environment,configure_torch
    configure_environment('det_auto');configure_torch('det_auto')
    import torch
    deadline=time.time()+120
    while torch.cuda.mem_get_info()[0]<23*1024**3:
        if time.time()>deadline:raise RuntimeError('GPU not free; do not stop unrelated processes')
        time.sleep(5)
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from binding.model import load_icae
    from binding.v11_training import load_model
    from binding.v11_pipeline import read_rows,write_json
    from binding.v12_reader import build_ram_cache,greedy,prefix_for
    from binding.v13_data import prepare
    from binding.v13_objectives import logits
    from binding.v9_model import load_state,adapter_state
    from binding.v4_model import state_identity
    all_units={u['id']:u for u in read_rows(data)};units=[all_units[p['unit_id']] for p in plan['pairs']];donors={p['unit_id']:all_units[p['donor_unit_id']] for p in plan['pairs']}
    assert len(units)==8 and len({u['article'] for u in units})==8
    for u in units:
        donor=donors[u['id']];assert u['article']!=donor['article'] and u['level']==donor['level']==4
        assert not {a for row in u['answers'] for a in row[:2]}&{f['value'] for facts in donor['facts'] for f in facts}
    prior=read(O.parent/'choice/protocol.json');jobs=[j for j in prior['jobs'] if j['name'] in plan['readers']];assert len(jobs)==2
    protocol=dict(plan=plan,lock_sha256=sha(R/'temp/v16/memory_control_lock.json'),jobs=[dict(j,checkpoint_sha256=sha(R/j['checkpoint'])) for j in jobs],
        scoring='Equal-length mean answer-plus-EOS log probability; ties fail; same original questions and candidates in both memory modes',reserved_test_accessed=False)
    write_json(O/'protocol.json',protocol)
    tok=AutoTokenizer.from_pretrained(str(R/'models/mistral'),local_files_only=True,use_fast=False);items={u['id']:prepare(u,tok) for u in units}
    enc,_=load_icae(str(R/'models/mistral'),str(R/'models/icae/mistral_7b_ft_icae.safetensors'),device='cuda',dtype=torch.bfloat16);enc.requires_grad_(False);enc.eval()
    cache=build_ram_cache(enc,[c for item in items.values() for c in item['contexts']],lambda **v:emit(phase='encoding',**v));cache_sha=cache.digest();del enc;gc.collect();torch.cuda.empty_cache()
    model,_=load_model('full',73);model.eval();assert torch.equal(model.ft.detach().cpu(),cache.ft)
    def score(p,answer):
        logp=logits(model,p,answer).log_softmax(-1).gather(-1,torch.tensor(answer,device=model.device).unsqueeze(-1)).squeeze(-1)
        assert bool(torch.isfinite(logp).all());return dict(mean=float(logp.mean()),sum=float(logp.sum()),token_ids=answer,token_logp=logp.tolist())
    for job in jobs:
        load_state(model,load_file(str(R/job['checkpoint']),device='cpu'));assert state_identity(adapter_state(model))['sha256']==job['identity']
        folder=O/job['name'];folder.mkdir();prior_dir=O.parent/'choice/development'/job['name'];old_receipt=read(prior_dir/'receipt.json');assert sha(prior_dir/'scores.jsonl')==old_receipt['scores_sha256']
        previous={(r['unit_id'],r['context_index'],r['question_index']):r for r in read_rows(prior_dir/'scores.jsonl')}
        raw_path=(R/job['checkpoint']).parent/'evaluation-512/predictions.jsonl';assert sha(raw_path)==old_receipt['raw_predictions_sha256'];raw={(r['unit_id'],r['context_index'],r['question_index']):r for r in read_rows(raw_path)}
        rows=[];replayed=0;max_delta=0.
        with torch.no_grad(),(folder/'scores.jsonl').open('x',encoding='utf8') as f:
            for number,u in enumerate(units,1):
                item=items[u['id']];donor=items[donors[u['id']]['id']]
                for i in range(4):
                    assert not torch.equal(cache.get(item['contexts'][i]),cache.get(donor['contexts'][i]))
                    for j in (0,1):
                        for mode in plan['modes']:
                            p=prefix_for(model,item,i,j,cache,mode,donor if mode=='shuffled' else None)
                            scores=[score(p,ids) for ids in item['answers'][i][:2]];assert all(len(s['token_ids'])==6 for s in scores)
                            a,b=[s['mean'] for s in scores];choice=None if a==b else int(b>a)
                            if mode=='matched':
                                old=previous[u['id'],i,j];delta=max(abs(s['mean']-t['mean']) for s,t in zip(scores,old['candidate_scores']));max_delta=max(max_delta,delta)
                                assert delta<=1e-5 and choice==old['choice']
                                if number==1 and i==0:
                                    prediction,generation=greedy(model,p,tok);assert prediction==raw[u['id'],i,j]['prediction'] and generation==raw[u['id'],i,j]['generation'];replayed+=1
                            row=dict(unit_id=u['id'],source_id=u['source_id'],article=u['article'],context_index=i,question_index=j,mode=mode,
                                memory_unit_id=donors[u['id']]['id'] if mode=='shuffled' else u['id'],candidate_values=u['answers'][i][:2],gold=u['answers'][i][j],candidate_scores=scores,choice=choice,choice_correct=choice==j)
                            f.write(json.dumps(row,allow_nan=False)+'\n');rows.append(row)
                emit(phase='scoring',job=job['name'],units=number,total_units=8)
        assert len(rows)==128 and replayed==2 and state_identity(adapter_state(model))['sha256']==job['identity']
        write_json(folder/'receipt.json',dict(job=job['name'],rows=128,matched_correct=sum(r['choice_correct'] for r in rows if r['mode']=='matched'),shuffled_correct=sum(r['choice_correct'] for r in rows if r['mode']=='shuffled'),
            exact_greedy_replay_rows=replayed,max_matched_score_difference=max_delta,scores_sha256=sha(folder/'scores.jsonl'),protocol_sha256=sha(O/'protocol.json')))
    assert cache.digest()==cache_sha
    emit(phase='complete',readers=2,rows=256,cache_unchanged=True)
    files=sorted(p for p in O.rglob('*') if p.is_file() and p.suffix in ('.json','.jsonl'));manifest=dict(part='memory_control',files=[dict(path=p.relative_to(O).as_posix(),size=p.stat().st_size,sha256=sha(p)) for p in files],models_included=False,reserved_test_accessed=False)
    buf=io.BytesIO()
    with tarfile.open(fileobj=buf,mode='w') as tar:
        for p in files:tar.add(p,arcname=p.relative_to(O).as_posix(),recursive=False)
        blob=(json.dumps(manifest,indent=2)+'\n').encode();info=tarfile.TarInfo('export_manifest.json');info.size=len(blob);tar.addfile(info,io.BytesIO(blob))
    packed=lzma.compress(buf.getvalue());dest=O.parent/'exports/memory_control.tar.xz'
    with dest.open('xb') as f:f.write(packed)
    write_json(O.parent/'exports/memory_control.json',dict(part='memory_control',archive=str(dest),bytes=len(packed),sha256=sha(dest),files=len(files)+1))
if __name__=='__main__':
    try:main()
    except BaseException as e:
        if O.exists():emit(phase='failed',error=str(e))
        raise
