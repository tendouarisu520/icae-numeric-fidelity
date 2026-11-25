"""Actual forced-choice scoring of locked readers; no optimizer or test partition."""
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path('E:/Research/ICAE-ConditionalMemory-20261021')
OUT=ROOT/'results/v16-code-fidelity-20261106/choice'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4194304),b''):h.update(block)
    return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf8'))
def emit(**values):
    value=dict(time=time.time(),pid=os.getpid(),**values)
    print(json.dumps(value),flush=True)
    (OUT/'status.json').write_text(json.dumps(value),encoding='utf8')

def main():
    if ROOT.drive.lower()!='e:':raise ValueError('E only')
    os.chdir(ROOT);OUT.mkdir(parents=True,exist_ok=True)
    for name,sub in {'TEMP':'temp','TMP':'temp','HF_HOME':'cache/huggingface','TORCH_HOME':'cache/torch','CUDA_CACHE_PATH':'cache/cuda','TRITON_CACHE_DIR':'cache/triton','TORCHINDUCTOR_CACHE_DIR':'cache/torchinductor','HF_DATASETS_CACHE':'cache/datasets','HUGGINGFACE_HUB_CACHE':'cache/huggingface/hub','XDG_CACHE_HOME':'cache','NUMBA_CACHE_DIR':'cache/numba'}.items():
        os.environ[name]=str(ROOT/sub)
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
    sys.path.insert(0,str(ROOT/'src'))
    from binding.repro_probe import configure_environment,configure_torch
    configure_environment('det_auto');configure_torch('det_auto')
    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from binding.model import load_icae
    from binding.v11_pipeline import read_rows,write_json,em_f1
    from binding.v11_training import load_model
    from binding.v12_reader import build_ram_cache,greedy
    from binding.v13_data import prepare
    from binding.v13_objectives import prefix,logits
    from binding.v16_code_fidelity import choice_decision
    from binding.v9_model import load_state,adapter_state
    from binding.v4_model import state_identity
    assert torch.cuda.mem_get_info()[0]>=23*1024**3,'GPU busy; do not stop another task'
    assert shutil.disk_usage(ROOT).free>=1024**3
    original=read(ROOT/'results/confirmation-20261104/protocol.json')
    for name,pin in original['pins'].items():assert sha(ROOT/name)==pin,name
    paths={'development':ROOT/'data/v13-answerable-relations-20261026/development.jsonl',
           'confirmation':ROOT/'results/confirmation-20261104/confirmation_units.jsonl'}
    units={s:[u for u in read_rows(p) if u['family']=='probe'] for s,p in paths.items()}
    cfg=dict(date='2026-11-06',runner_sha256=sha(Path(__file__)),original_protocol_sha256=sha(ROOT/'results/confirmation-20261104/protocol.json'),
        data={s:sha(p) for s,p in paths.items()},choice_module=sha(ROOT/'src/binding/v16_code_fidelity.py'),
        scoring='mean log probability of answer including leading blank and EOS; equal token lengths; exact ties fail strict choice',
        selection='two target values, not shown in the input; all four contexts and two target queries',
        model_selection=False,post_hoc_diagnostic=True,reserved_test_accessed=False,
        free_generation='unchanged recorded greedy strings; exact per-reader replay checked',jobs=original['jobs'])
    protocol=OUT/'protocol.json'
    if protocol.exists():assert read(protocol)==cfg
    else:write_json(protocol,cfg)
    tokenizer=AutoTokenizer.from_pretrained(str(ROOT/'models/mistral'),local_files_only=True,use_fast=False)
    items={s:{u['id']:prepare(u,tokenizer) for u in values} for s,values in units.items()}
    emit(phase='encoding',contexts=sum(4*len(v) for v in units.values()))
    encoder,_=load_icae(str(ROOT/'models/mistral'),str(ROOT/'models/icae/mistral_7b_ft_icae.safetensors'),device='cuda',dtype=torch.bfloat16)
    encoder.requires_grad_(False);encoder.eval()
    cache=build_ram_cache(encoder,[c for split in items.values() for item in split.values() for c in item['contexts']],lambda **v:emit(phase='encoding',**v))
    cache_sha=cache.digest();del encoder;gc.collect();torch.cuda.empty_cache()
    model,_=load_model('full',73);model.eval()
    assert torch.equal(model.ft.detach().cpu(),cache.ft)
    jobs=sorted(original['jobs'],key=lambda j:(j['method']!='sft',j['arm']=='full',j['seed'],j['method']))
    for job in jobs:
        load_state(model,load_file(str(ROOT/job['checkpoint']),device='cpu'))
        assert state_identity(adapter_state(model))['sha256']==job['identity']
        for split,values in units.items():
            dest=OUT/split/job['name']
            if (dest/'receipt.json').exists():
                receipt=read(dest/'receipt.json')
                assert receipt['scores_sha256']==sha(dest/'scores.jsonl') and receipt['protocol_sha256']==sha(protocol)
                continue
            assert not dest.exists(),'Partial job; diagnose before retry'
            dest.mkdir(parents=True)
            if split=='confirmation':rawpath=ROOT/'results/confirmation-20261104'/job['name']/'predictions.jsonl'
            else:rawpath=(ROOT/job['checkpoint']).parent/'evaluation-512/predictions.jsonl'
            raw={(r['unit_id'],r['context_index'],r['question_index']):r for r in read_rows(rawpath) if r['family']=='probe'}
            expected_sha=read(rawpath.parent/'metrics.json')['predictions_sha256'] if split=='confirmation' else read((ROOT/job['checkpoint']).parent/'evaluations.json')['512']['predictions_sha256']
            assert sha(rawpath)==expected_sha
            replay=0
            with torch.no_grad():
                u=values[0];item=items[split][u['id']]
                for i,j in [(0,0),(1,1),(0,2),(3,2)]:
                    pred,generation=greedy(model,prefix(model,item,i,j,cache,job['arm']),tokenizer)
                    reference=raw[u['id'],i,j]
                    assert pred==reference['prediction'] and generation==reference['generation'],'Locked greedy replay differs'
                    replay+=1
            groups={};n=0;correct=0;free=0;ties=0;rescued=0;lost=0;started=time.perf_counter()
            with torch.no_grad(),(dest/'scores.jsonl').open('x',encoding='utf8',buffering=1) as f:
                for count,u in enumerate(values,1):
                    item=items[split][u['id']];group={};groups[u['id']]=group
                    for i in range(4):
                        for j in (0,1):
                            reference=raw[u['id'],i,j]
                            assert reference['gold']==u['answers'][i][j]
                            p=prefix(model,item,i,j,cache,job['arm']);records=[]
                            candidates=item['answers'][i][:2]
                            assert len(candidates[0])==len(candidates[1])
                            for candidate in candidates:
                                output=logits(model,p,candidate)
                                truth=torch.tensor(candidate,device=model.device)
                                lp=output.log_softmax(-1).gather(-1,truth.unsqueeze(-1)).squeeze(-1)
                                assert bool(torch.isfinite(lp).all())
                                records.append(dict(mean=float(lp.mean()),sum=float(lp.sum()),token_logp=lp.tolist(),
                                    token_ids=candidate,teacher_forced_argmax=output.argmax(-1).tolist()))
                            decision=choice_decision([r['mean'] for r in records]);ok=decision==j
                            greedy_ok=bool(em_f1(reference['prediction'],reference['gold'])[0])
                            n+=1;correct+=ok;free+=greedy_ok;ties+=decision is None;rescued+=(ok and not greedy_ok);lost+=(greedy_ok and not ok)
                            group[i,j]=(ok,greedy_ok)
                            record=dict(unit_id=u['id'],source_id=u['source_id'],article=u['article'],level=u['level'],
                                context_index=i,question_index=j,gold=reference['gold'],prediction=reference['prediction'],
                                candidate_values=u['answers'][i][:2],candidate_scores=records,choice=decision,
                                choice_correct=ok,greedy_correct=greedy_ok,gold_margin=records[j]['mean']-records[1-j]['mean'])
                            f.write(json.dumps(record,allow_nan=False)+'\n')
                    if count==1 or count%16==0 or count==len(values):emit(phase='scoring',job=job['name'],split=split,units=count,total_units=len(values),rows=n,seconds=time.perf_counter()-started)
            target=lambda k:sum(all(g[i,j][k] for i in (0,1) for j in (0,1)) for g in groups.values())
            result=dict(job=job['name'],split=split,rows=n,choice_correct=correct,greedy_correct=free,ties=ties,
                wrong_generation_correct_choice=rescued,correct_generation_wrong_choice=lost,
                choice_target_joint=target(0),greedy_target_joint=target(1),groups=len(groups),
                exact_replay_rows=replay,wall_seconds=time.perf_counter()-started,scores_sha256=sha(dest/'scores.jsonl'),
                raw_predictions_sha256=sha(rawpath),protocol_sha256=sha(protocol),checkpoint_sha256=original['pins'][job['checkpoint']])
            write_json(dest/'receipt.json',result);emit(phase='cell_complete',**result)
    assert cache.digest()==cache_sha
    emit(phase='complete',cells=20)
if __name__=='__main__':main()
