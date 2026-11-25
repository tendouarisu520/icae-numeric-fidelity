"""One frozen inference pass on each of ten existing readers."""
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path('E:/Research/ICAE-ConditionalMemory-20261021')
OUT = ROOT / 'results/confirmation-20261104'
CFG = OUT / 'protocol.json'

def read(path):
    return json.loads(path.read_text(encoding='utf8'))

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4194304), b''):
            h.update(chunk)
    return h.hexdigest()

def status(**values):
    print(json.dumps(dict(time=time.time(), **values)), flush=True)

def main():
    os.chdir(ROOT)
    for name, sub in {'TEMP':'temp','TMP':'temp','HF_HOME':'cache/huggingface','TORCH_HOME':'cache/torch','CUDA_CACHE_PATH':'cache/cuda','TRITON_CACHE_DIR':'cache/triton','TORCHINDUCTOR_CACHE_DIR':'cache/torchinductor','HF_DATASETS_CACHE':'cache/datasets','HUGGINGFACE_HUB_CACHE':'cache/huggingface/hub','XDG_CACHE_HOME':'cache','NUMBA_CACHE_DIR':'cache/numba'}.items():
        os.environ[name]=str(ROOT/sub)
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
    sys.path.insert(0,str(ROOT/'src'))
    cfg=read(CFG)
    assert digest(Path(__file__))==cfg['runner']
    for name,expected in cfg['pins'].items():
        assert digest(ROOT/name)==expected,name
    status(phase='verified_code_data_models_and_ten_weights')
    from binding.repro_probe import configure_environment,configure_torch
    configure_environment('det_auto')
    configure_torch('det_auto')
    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from binding.model import load_icae
    from binding.v11_pipeline import read_rows,write_json
    from binding.v11_training import load_model
    from binding.v12_reader import build_ram_cache
    from binding.v13_training import build_units,evaluate
    from binding.v13_data import prepare
    from binding.v9_model import load_state,adapter_state
    from binding.v4_model import state_identity
    base=ROOT/'data/v11-natural-frozen-20261025'
    rows=read_rows(base/'confirmation.jsonl')
    assert len(rows)==len({u['id'] for u in rows})==64 and len({u['article'] for u in rows})==22
    for split in ('train','development'):
        others=read_rows(base/(split+'.jsonl'))
        for k in ('id','article'):
            assert {u[k] for u in rows}.isdisjoint(u[k] for u in others)
    units=build_units(rows,'development')
    for unit in units: unit['split']='confirmation'
    assert len(units)==256 and len({u['id'] for u in units})==256
    path=OUT/'confirmation_units.jsonl'
    data=''.join(json.dumps(u,ensure_ascii=False)+'\n' for u in units).encode()
    if path.exists():assert path.read_bytes()==data
    else:path.write_bytes(data)
    tok=AutoTokenizer.from_pretrained(str(ROOT/'models/mistral'),local_files_only=True,use_fast=False)
    items={u['id']:prepare(u,tok) for u in units}
    remaining=[]
    for job in cfg['jobs']:
        dest=OUT/job['name']
        if (dest/'receipt.json').exists():
            rec=read(dest/'receipt.json')
            assert rec['protocol']==digest(CFG) and rec['predictions']==digest(dest/'predictions.jsonl')
        else:
            assert not dest.exists(), 'Partial cell exists; audit before retry'
            remaining.append(job)
    cache=None
    if any(job['arm']=='compressed' for job in remaining):
        status(phase='encode_frozen_contexts')
        enc,_=load_icae(str(ROOT/'models/mistral'),str(ROOT/'models/icae/mistral_7b_ft_icae.safetensors'),device='cuda',dtype=torch.bfloat16)
        enc.requires_grad_(False);enc.eval()
        cache=build_ram_cache(enc,[ctx for v in items.values() for ctx in v['contexts']],lambda **v:status(**v))
        cache_sha=cache.digest()
        del enc
        gc.collect();torch.cuda.empty_cache()
    if remaining:
        model,tok=load_model('full',73)
        model.eval()
        if cache is not None:assert torch.equal(model.ft.detach().cpu(),cache.ft)
        for job in remaining:
            dest=OUT/job['name']
            load_state(model,load_file(str(ROOT/job['checkpoint']),device='cpu'))
            assert state_identity(adapter_state(model))['sha256']==job['identity']
            status(job=job['name'],phase='start')
            with torch.no_grad():
                result=evaluate(model,tok,units,items,cache if job['arm']=='compressed' else None,job['arm'],dest,lambda **v:status(job=job['name'],**v))
            assert result['natural']['rows']==448 and result['probe']['rows']==3072
            write_json(dest/'receipt.json',dict(protocol=digest(CFG),units=digest(path),predictions=result['predictions_sha256'],checkpoint=cfg['pins'][job['checkpoint']],cache=cache_sha if job['arm']=='compressed' else None))
            status(job=job['name'],phase='complete')
        if cache is not None:assert cache.digest()==cache_sha
    assert all((OUT/job['name']/'receipt.json').exists() for job in cfg['jobs'])
    status(phase='all_ten_complete')
if __name__=='__main__': main()

