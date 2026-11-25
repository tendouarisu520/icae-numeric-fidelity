"""Wait for V10, run actual calibration, then six fixed natural-memory cells."""
import json
import hashlib
import os
from pathlib import Path
import subprocess
import shutil
import sys
import time
import traceback

ROOT=Path('E:/Research/ICAE-ConditionalMemory-20261021')
RUNS=ROOT/'runs/v11-natural-20261025';LOGS=ROOT/'logs/v11-natural-20261025'
PYTHON=ROOT/'envs/icae/Scripts/python.exe'


def main():
    if ROOT.drive.lower()!='e:':raise ValueError('E only')
    os.chdir(ROOT);sys.path.insert(0,str(ROOT/'src'))
    from binding.v11_pipeline import sha,write_json
    from binding.v10_comparison import collect
    plan=json.loads((ROOT/'docs/v11-run-plan-20261025.json').read_text(encoding='utf8'))
    pinned={relative:sha(ROOT/relative) for relative in plan['source_files']}
    if pinned!=plan['source_files']:raise ValueError('Deployed source differs from fixed plan')
    cohort_sha=sha(ROOT/'data/v11-natural-frozen-20261025/manifest.json')
    RUNS.mkdir(exist_ok=False);LOGS.mkdir(exist_ok=False)
    state=dict(status='running',phase='waiting_for_v10',pid=os.getpid(),started=time.time(),completed=[],scientific_starts=0,
               maximum_scientific_starts=6,cohort_manifest_sha256=cohort_sha)
    def update(**values):
        state.update(values,updated=time.time());write_json(RUNS/'queue-status.json',state);print(json.dumps(state),flush=True)
    env=dict(os.environ)
    env.update(PYTHONPATH=str(ROOT/'src'),PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED='73',HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1',TEMP=str(ROOT/'temp'),TMP=str(ROOT/'temp'),HF_HOME=str(ROOT/'cache/huggingface'),TORCH_HOME=str(ROOT/'cache/torch'))
    env.pop('CUDA_VISIBLE_DEVICES',None)
    def check_sources():
        if any(sha(ROOT/relative)!=expected for relative,expected in pinned.items()):raise ValueError('Pinned V11 source changed')
        if sha(ROOT/'data/v11-natural-frozen-20261025/manifest.json')!=cohort_sha:raise ValueError('Cohort changed')
    def resources():
        started=time.time()
        while True:
            if shutil.disk_usage(ROOT).free<300_000_000:raise RuntimeError('Less300MB free on E; preserve existing files')
            result=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
            if int(result.stdout.strip().splitlines()[0])>=23500:return
            update(phase='waiting_for_gpu_memory',free_mib=int(result.stdout.strip().splitlines()[0]))
            if time.time()-started>12*3600:raise RuntimeError('GPU memory unavailable for12hours; no user process stopped')
            time.sleep(30)
    def run(name,arguments,scientific=False):
        check_sources()
        if scientific:
            state['scientific_starts']+=1
            if state['scientific_starts']>6:raise RuntimeError('Run budget exhausted')
        record=RUNS/(name+'-process.json');out=LOGS/(name+'.stdout.log');err=LOGS/(name+'.stderr.log')
        if any(p.exists() for p in (record,out,err)):raise ValueError('Existing run; no overwrite or retry')
        command=[str(PYTHON),'-B',str(ROOT/'src/scripts/run_recorded.py'),'--record',str(record),'--','-B',*arguments]
        with out.open('x',encoding='utf8') as stdout,err.open('x',encoding='utf8') as stderr:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
            update(phase='running',current_job=name,wrapper_pid=child.pid,stdout=str(out),stderr=str(err),process_record=str(record),current_progress=None)
            while child.poll() is None:
                path=RUNS/name/'status.json'
                progress=json.loads(path.read_text(encoding='utf8')) if path.exists() else None
                update(current_progress=progress,stdout_bytes=out.stat().st_size,stderr_bytes=err.stat().st_size,
                       last_log_write=max(out.stat().st_mtime,err.stat().st_mtime))
                time.sleep(20)
        actual=json.loads(record.read_text(encoding='utf8'))
        if child.returncode or actual['status']!='success' or actual['exit_code']!=0:raise RuntimeError('Job failed:'+name)
        state['completed'].append(dict(job=name,exit_code=0,record_sha256=sha(record),finished=time.time()))
        update(current_progress=None)
    def model_job(phase,arm,method,seed):
        resources();name=f'{phase}-{arm}-{method}-seed{seed}'
        run(name,['-m','binding.v11_training','--phase',phase,'--arm',arm,'--method',method,'--seed',str(seed),'--output',str(RUNS/name)],phase=='main')
        result=json.loads((RUNS/name/'summary.json').read_text(encoding='utf8'))
        if result['status']!='success' or not result['reload_exact'] or not result['frozen_parameters_unchanged']:raise RuntimeError('Incomplete model result:'+name)
        if phase=='overfit' and not result['overfit_pass']:raise RuntimeError('Eight-unit overfit target missed; inspect recorded predictions before main starts')
        if phase=='main' and result['optimizer_steps']!=1024:raise RuntimeError('Main endpoint incomplete')
        return result
    update()
    try:
        deadline=time.time()+12*3600
        while True:
            q=json.loads((ROOT/'runs/v10-coefficient-20261025/queue-status.json').read_text(encoding='utf8'))
            if q['status']=='failed':raise RuntimeError('V10 failed; GPU handover cancelled')
            record=json.loads((ROOT/'runs/v10-coefficient-20261025/queue-process.json').read_text(encoding='utf8'))
            if q['status']=='success' and record['status']=='success' and record['exit_code']==0:break
            update(phase='waiting_for_v10',v10_current_job=q.get('current_job'),v10_progress=q.get('current_progress'))
            if time.time()>deadline:raise RuntimeError('V10 has not completed within12hours')
            time.sleep(30)
        report=collect(ROOT);update(v10_comparison=str(ROOT/'results/v10-coefficient-comparison-20261025/comparison.json'))
        for relative,expected in plan['model_files'].items():
            update(phase='verify_model_files',model_file=relative)
            digest=hashlib.sha256()
            with (ROOT/relative).open('rb') as handle:
                for block in iter(lambda:handle.read(4*1024*1024),b''):digest.update(block)
            if digest.hexdigest()!=expected:raise ValueError('Official model file differs:'+relative)
        run('unit-tests',['-m','unittest','discover','-s',str(ROOT/'src/tests'),'-p','test_v11*.py','-v'])
        text=(LOGS/'unit-tests.stderr.log').read_text(encoding='utf8')
        if 'Ran 20 tests' not in text or 'skipped' in text or '\nOK\n' not in text:raise RuntimeError('All20 tests must execute including neural tests')
        model_job('smoke','encoder','relation',73)
        model_job('overfit','encoder','sft',73)
        reference={}
        for seed in (73,97):
            for arm,method in (('encoder','sft'),('encoder','relation'),('full','sft')):
                result=model_job('main',arm,method,seed)
                config=json.loads((RUNS/f'main-{arm}-{method}-seed{seed}'/'config.json').read_text(encoding='utf8'))
                if method=='sft' and arm=='encoder':reference[seed]=(result['initial_adapter_identity'],config['schedule'])
                if config['schedule']!=reference[seed][1]:raise RuntimeError('Methods had unequal unit exposure')
                if arm=='encoder' and result['initial_adapter_identity']!=reference[seed][0]:raise RuntimeError('Encoder initialization unequal')
        check_sources();update(status='success',phase='complete',current_job=None,finished=time.time())
    except BaseException as error:
        update(status='failed',phase='failed',error=str(error),traceback=traceback.format_exc(),finished=time.time());raise


if __name__=='__main__':main()
