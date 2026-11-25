"""Adopt the existing V11 trainer and run remaining cells with a tested full entry."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT=Path('E:/Research/ICAE-ConditionalMemory-20261021')
RUNS=ROOT/'runs/v11-natural-20261025'
LOGS=ROOT/'logs/v11-natural-20261025'
PLAN=ROOT/'docs/v11-continuation-plan-20261025.json'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    os.chdir(ROOT);sys.path.insert(0,str(ROOT/'src'))
    from binding.v11_pipeline import sha,write_json
    plan=json.loads(PLAN.read_text(encoding='utf8'))
    original=json.loads((ROOT/'docs/v11-run-plan-20261025.json').read_text(encoding='utf8'))
    pinned={**original['source_files'],**plan['source_files']}
    def check():
        if any(sha(ROOT/p)!=expected for p,expected in pinned.items()):raise ValueError('Pinned source changed')
        if sha(ROOT/'data/v11-natural-frozen-20261025/manifest.json')!=plan['cohort_manifest_sha256']:
            raise ValueError('Cohort changed')
    check()
    state=json.loads((RUNS/'queue-status.json').read_text(encoding='utf8'))
    if args.preflight:
        print(json.dumps(dict(status='preflight_pass',current_job=state.get('current_job'),
            current_progress=state.get('current_progress'),pinned_files=len(pinned),original_sources_unchanged=True)),flush=True)
        return
    if state['status']!='running':raise ValueError('Only an ongoing queue can be adopted')
    snapshot=RUNS/'queue-before-full-entry-handover.json'
    if snapshot.exists():raise ValueError('Handover already attempted; inspect rather than duplicate')
    write_json(snapshot,state)
    state.update(controller_history=[dict(pid=state['pid'],retired_reason='Full generation API compatibility entry')],
                 pid=os.getpid(),continuation_plan_sha256=sha(PLAN),full_entry='binding.v11_full_entry')
    def update(**values):
        state.update(values,updated=time.time());write_json(RUNS/'queue-status.json',state)
        print(json.dumps({k:state.get(k) for k in ('status','phase','current_job','current_progress','scientific_starts','updated')}),flush=True)
    env=dict(os.environ)
    env.update(PYTHONPATH=str(ROOT/'src'),PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED='73',HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',TEMP=str(ROOT/'temp'),TMP=str(ROOT/'temp'),HF_HOME=str(ROOT/'cache/huggingface'),TORCH_HOME=str(ROOT/'cache/torch'))
    env.pop('CUDA_VISIBLE_DEVICES',None)
    def resources():
        start=time.time()
        while True:
            if shutil.disk_usage(ROOT).free<300_000_000:raise RuntimeError('Less300MB free; preserve artifacts')
            result=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
            free=int(result.stdout.strip().splitlines()[0])
            if free>=23500:return
            update(phase='waiting_for_gpu_memory',free_mib=free)
            if time.time()-start>12*3600:raise TimeoutError('GPU not released; no user application stopped')
            time.sleep(30)
    update(phase='adopting_existing_trainer')
    try:
        reference={}
        for seed in (73,97):
            for arm,method in (('encoder','sft'),('encoder','relation'),('full','sft')):
                check();name=f'main-{arm}-{method}-seed{seed}'
                folder=RUNS/name;record=RUNS/(name+'-process.json')
                out=LOGS/(name+'.stdout.log');err=LOGS/(name+'.stderr.log')
                if not record.exists():
                    if any(p.exists() for p in (folder,out,err)):raise ValueError('Partial unrecorded job:'+name)
                    resources();check()
                    if state['scientific_starts']>=6:raise RuntimeError('Six-run limit reached')
                    entry='binding.v11_full_entry' if arm=='full' else 'binding.v11_training'
                    arguments=['-m',entry,'--phase','main','--arm',arm,'--method',method,'--seed',str(seed),'--output',str(folder)]
                    command=[str(ROOT/'envs/icae/Scripts/python.exe'),'-B',str(ROOT/'src/scripts/run_recorded.py'),
                             '--record',str(record),'--','-B',*arguments]
                    with out.open('x',encoding='utf8') as stdout,err.open('x',encoding='utf8') as stderr:
                        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
                    state['scientific_starts']+=1
                    update(phase='running',current_job=name,wrapper_pid=child.pid,stdout=str(out),stderr=str(err),process_record=str(record))
                    deadline=time.time()+30
                    while not record.exists():
                        if child.poll() is not None or time.time()>deadline:raise RuntimeError('Recording wrapper failed:'+name)
                        time.sleep(.2)
                elif name!=state.get('current_job') and not (folder/'summary.json').exists():
                    raise ValueError('Cannot adopt an unrelated unfinished job:'+name)
                deadline=time.time()+12*3600
                while True:
                    actual=json.loads(record.read_text(encoding='utf8'))
                    progress=json.loads((folder/'status.json').read_text(encoding='utf8')) if (folder/'status.json').exists() else None
                    update(phase='running',current_job=name,current_progress=progress,process_record=str(record),
                        stdout=str(out),stderr=str(err),stdout_bytes=out.stat().st_size,stderr_bytes=err.stat().st_size,
                        last_log_write=max(out.stat().st_mtime,err.stat().st_mtime))
                    if actual['status'] not in ('starting','running'):break
                    if time.time()>deadline:raise TimeoutError('Job exceeded12hours:'+name)
                    time.sleep(20)
                if actual['status']!='success' or actual['exit_code']!=0:raise RuntimeError('Recorded job failed:'+name)
                result=json.loads((folder/'summary.json').read_text(encoding='utf8'))
                config=json.loads((folder/'config.json').read_text(encoding='utf8'))
                if result['status']!='success' or result['optimizer_steps']!=1024 or not result['reload_exact'] or not result['frozen_parameters_unchanged']:
                    raise RuntimeError('Incomplete1024 cell:'+name)
                if arm=='encoder' and method=='sft':reference[seed]=(result['initial_adapter_identity'],config['schedule'])
                if config['schedule']!=reference[seed][1]:raise RuntimeError('Exposure differs')
                if arm=='encoder' and result['initial_adapter_identity']!=reference[seed][0]:raise RuntimeError('Encoder initialization differs')
                if not any(c['job']==name for c in state['completed']):
                    state['completed'].append(dict(job=name,exit_code=0,record_sha256=sha(record),finished=time.time()))
                update(current_progress=None)
        check();update(status='success',phase='complete',current_job=None,finished=time.time())
    except BaseException as error:
        update(status='failed',phase='failed',error=str(error),traceback=traceback.format_exc(),finished=time.time());raise


if __name__=='__main__':main()
