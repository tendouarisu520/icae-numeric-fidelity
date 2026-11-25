"""Serialize the exposure diagnostics after the already running choice job."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

R=Path('E:/Research/ICAE-ConditionalMemory-20261021')
O=R/'results/v16-code-fidelity-20261106'
def status(**v):
    v=dict(time=time.time(),pid=os.getpid(),**v);print(json.dumps(v),flush=True)
    (O/'exposure_queue_status.json').write_text(json.dumps(v),encoding='utf8')
def main():
    os.chdir(R);os.environ['PYTHONPATH']=str(R/'src')
    lock=json.loads((R/'temp/v16/exposure_lock.json').read_text())
    for name,pin in lock['files'].items():
        assert hashlib.sha256((R/name).read_bytes()).hexdigest()==pin,name
    for name,sub in {'TEMP':'temp','TMP':'temp','HF_HOME':'cache/huggingface','TORCH_HOME':'cache/torch','CUDA_CACHE_PATH':'cache/cuda','TRITON_CACHE_DIR':'cache/triton','TORCHINDUCTOR_CACHE_DIR':'cache/torchinductor','HF_DATASETS_CACHE':'cache/datasets','HUGGINGFACE_HUB_CACHE':'cache/huggingface/hub','XDG_CACHE_HOME':'cache','NUMBA_CACHE_DIR':'cache/numba'}.items():os.environ[name]=str(R/sub)
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
    path=O/'choice/status.json';last=None;last_change=time.time()
    while True:
        current=json.loads(path.read_text(encoding='utf8'))
        if current['phase']=='complete':break
        if current!=last:last=current;last_change=time.time()
        if time.time()-last_change>1800:raise RuntimeError('Choice status unchanged30min; inspect before training')
        status(phase='waiting_for_choice',choice=current)
        time.sleep(20)
    # No shared CUDA tensors survive from the previous process before the resource check.
    import torch
    deadline=time.time()+120
    while torch.cuda.mem_get_info()[0]<23*1024**3:
        if time.time()>deadline:raise RuntimeError('GPU not free after choice; do not interrupt other work')
        status(phase='waiting_for_choice_cuda_release')
        time.sleep(5)
    jobs=[]
    for seed in (73,97):
        name=f'main-compressed-sft-seed{seed}';out=R/'runs/v16-repeated-exposure-20261106'/name
        if out.exists():raise RuntimeError('Existing job requires explicit audit; no automatic restart')
        log=R/f'logs/v16-exposure-{seed}-{time.strftime("%Y%m%d-%H%M%S")}'
        with Path(str(log)+'.log').open('x') as stdout,Path(str(log)+'.err.log').open('x') as stderr:
            proc=subprocess.Popen([sys.executable,'-B','-m','binding.v16_training','--phase','main','--arm','compressed','--method','sft','--seed',str(seed)],cwd=R,stdout=stdout,stderr=stderr)
            status(phase='training',job=name,child_pid=proc.pid,log=str(log),output=str(out),completed_jobs=jobs)
            code=proc.wait()
        if code!=0:raise RuntimeError(f'{name} failed exit{code}; see {log}')
        summary=json.loads((out/'summary.json').read_text())
        assert summary['status']=='success' and summary['optimizer_steps']==1024 and summary['step512_exact_replay']
        jobs.append(dict(job=name,summary_sha256=hashlib.sha256((out/'summary.json').read_bytes()).hexdigest(),plateau=summary['plateau_diagnostic']))
    status(phase='complete',jobs=jobs)
if __name__=='__main__':
    try:main()
    except BaseException as error:status(phase='failed',error=str(error));raise
