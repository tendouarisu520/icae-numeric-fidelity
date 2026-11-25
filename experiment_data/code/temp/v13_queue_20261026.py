"""Recorded, bounded answerable-reading queue; no original artifacts modified."""
from pathlib import Path
import ctypes
import ctypes.wintypes as wt
import json
import os
import shutil
import struct
import subprocess
import sys
import time
import traceback

ROOT=Path('E:/Research/ICAE-ConditionalMemory-20261021')
RUNS=ROOT/'runs/v13-answerable-reading-20261026'
LOGS=ROOT/'logs/v13-answerable-reading-20261026'

def main():
    os.chdir(ROOT);sys.path.insert(0,str(ROOT/'src'))
    from binding.v11_pipeline import write_json,sha
    plan_path=ROOT/'docs/v13-run-plan-20261026.json';plan=json.loads(plan_path.read_text(encoding='utf8'));plan_sha=sha(plan_path)
    def pins():
        if sha(plan_path)!=plan_sha:raise ValueError('Run plan changed')
        if any(sha(ROOT/n)!=h for n,h in plan['source_files'].items()):raise ValueError('Pinned source changed')
        for split,expected in plan['original_data_hashes'].items():
            if sha(ROOT/'data/v11-natural-frozen-20261025'/split)!=expected:raise ValueError('Original frozen data changed')
    pins()
    if json.loads((ROOT/'runs/v12-frozen-reader-20261025/queue-status.json').read_text(encoding='utf8'))['status']!='success':raise ValueError('Finish original V12 queue first')
    RUNS.mkdir(exist_ok=False);LOGS.mkdir(exist_ok=False)
    state=dict(status='running',phase='preflight',pid=os.getpid(),completed=[],scientific_starts=0,maximum_scientific_starts=8,started=time.time())
    def update(**values):
        state.update(values,updated=time.time());write_json(RUNS/'queue-status.json',state);print(json.dumps(state),flush=True)
    env=dict(os.environ);env.update(PYTHONPATH=str(ROOT/'src'),PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED='73',HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',TEMP=str(ROOT/'temp'),TMP=str(ROOT/'temp'),HF_HOME=str(ROOT/'cache/huggingface'),TORCH_HOME=str(ROOT/'cache/torch'))
    env.pop('CUDA_VISIBLE_DEVICES',None)
    def resources():
        start=time.time()
        while True:
            if shutil.disk_usage(ROOT).free<450_000_000:raise RuntimeError('Efree below450MB; do not delete old artifacts')
            buffer=ctypes.create_string_buffer(64);struct.pack_into('I',buffer,0,64)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(buffer):raise OSError('RAM status unavailable')
            ram=struct.unpack_from('Q',buffer,16)[0]
            result=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
            free=int(result.stdout.strip().splitlines()[0])
            if free>=23500 and ram>=12*1024**3:return
            update(phase='waiting_for_resources',free_gpu_mib=free,available_ram_bytes=ram)
            if time.time()-start>12*3600:raise TimeoutError('Resources unavailable; no user process stopped')
            time.sleep(30)
    def cpu(pid):
        api=ctypes.windll.kernel32;api.OpenProcess.restype=ctypes.c_void_p
        handle=api.OpenProcess(0x1000,False,pid)
        if not handle:return None
        times=[wt.FILETIME() for _ in range(4)]
        value=sum((t.dwHighDateTime<<32)|t.dwLowDateTime for t in times[2:])/10_000_000 if api.GetProcessTimes(ctypes.c_void_p(handle),*(ctypes.byref(t) for t in times)) else None
        api.CloseHandle(ctypes.c_void_p(handle));return value
    def run(name,arguments,scientific=False):
        pins();record=RUNS/(name+'-process.json');out=LOGS/(name+'.stdout.log');err=LOGS/(name+'.stderr.log')
        if any(p.exists() for p in (record,out,err)):raise ValueError('Existing job; no overwrite/retry')
        if scientific:
            if state['scientific_starts']>=8:raise RuntimeError('Eight-cell budget exhausted')
            state['scientific_starts']+=1
        cmd=[str(ROOT/'envs/icae/Scripts/python.exe'),'-B',str(ROOT/'src/scripts/run_recorded.py'),'--record',str(record),'--','-B',*arguments]
        with out.open('x',encoding='utf8') as stdout,err.open('x',encoding='utf8') as stderr:
            child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
        update(phase='running',current_job=name,wrapper_pid=child.pid,stdout=str(out),stderr=str(err),process_record=str(record))
        deadline=time.time()+12*3600
        while child.poll() is None:
            status=RUNS/name/'status.json';progress=json.loads(status.read_text(encoding='utf8')) if status.exists() else None
            last=max(out.stat().st_mtime,err.stat().st_mtime);stale=time.time()-last
            update(current_progress=progress,stdout_bytes=out.stat().st_size,stderr_bytes=err.stat().st_size,
                last_log_write=last,stale_log_seconds=stale,possible_stall=stale>300,
                child_cpu_seconds=cpu(progress['pid']) if progress else None)
            if time.time()>deadline:raise TimeoutError('Job exceeded12hours; inspect owned job:'+name)
            time.sleep(20)
        actual=json.loads(record.read_text(encoding='utf8'))
        if child.returncode or actual['status']!='success' or actual['exit_code']!=0:raise RuntimeError('Job failed:'+name)
        state['completed'].append(dict(job=name,exit_code=0,record_sha256=sha(record)));update(current_progress=None)
    update()
    try:
        if shutil.disk_usage(ROOT).free<700_000_000:raise RuntimeError('Initial Efree below700MB')
        run('unit-tests',['-m','unittest','discover','-s',str(ROOT/'src/tests'),'-p','test_v13*.py','-v'])
        result=(LOGS/'unit-tests.stderr.log').read_text(encoding='utf8')
        if 'Ran 13 tests' not in result or '\nOK\n' not in result or 'skipped' in result:raise RuntimeError('All13 actual tests must pass with0skips')
        run('prepare',['-m','binding.v13_training','--phase','prepare'])
        jobs=[('smoke','compressed','relation',73),('overfit','full','sft',73),('overfit','compressed','sft',73)]
        jobs += [('main',arm,method,seed) for seed in (73,97) for arm,method in
            (('compressed','sft'),('compressed','assignment'),('compressed','relation'),('full','sft'))]
        initial={};cache_sha=None
        for phase,arm,method,seed in jobs:
            resources();name=f'{phase}-{arm}-{method}-seed{seed}'
            run(name,['-m','binding.v13_training','--phase',phase,'--arm',arm,'--method',method,'--seed',str(seed)],phase=='main')
            summary=json.loads((RUNS/name/'summary.json').read_text(encoding='utf8'))
            if summary['status']!='success' or not all(summary[k] for k in ('cache_unchanged','frozen_parameters_unchanged','reload_exact')):raise ValueError('Calibration failed:'+name)
            if phase=='overfit' and not summary['overfit_pass']:raise ValueError('Overfit gate failed')
            if phase=='main':
                if summary['optimizer_steps']!=512:raise ValueError('Registered512 endpoint required')
                if seed in initial and initial[seed]!=summary['initial_reader_identity']:raise ValueError('Reader initialization differs')
                initial[seed]=summary['initial_reader_identity']
                if arm=='compressed':
                    if cache_sha is not None and cache_sha!=summary['cache_sha256']:raise ValueError('Memory differs across methods/seeds')
                    cache_sha=summary['cache_sha256']
        cells=[]
        for seed in (73,97):
            for arm,method in (('compressed','sft'),('compressed','assignment'),('compressed','relation'),('full','sft')):
                p=RUNS/f'main-{arm}-{method}-seed{seed}'/'summary.json';s=json.loads(p.read_text(encoding='utf8'))
                cells.append(dict(arm=arm,method=method,seed=seed,fixed512=s['evaluations']['512'],best_development_step=s['best_development_step'],
                    training_seconds=s['cumulative_training_seconds'],peak_allocated_mib=s['peak_allocated_mib'],summary_sha256=sha(p)))
        index={(c['arm'],c['method'],c['seed']):c for c in cells};comparisons={}
        for method in ('assignment','relation'):
            comparisons[method]={}
            for seed in (73,97):
                a=index['compressed',method,seed]['fixed512'];b=index['compressed','sft',seed]['fixed512']
                macro=lambda x:sum(m['rates']['target_joint'] for m in x['probe_by_level'].values())/3
                gain=macro(a)-macro(b);natural=a['natural']['known_f1']-b['natural']['known_f1']
                locality=a['probe']['rates']['locality_joint']-b['probe']['rates']['locality_joint']
                comparisons[method][str(seed)]=dict(target_joint_gain=gain,natural_known_f1_delta=natural,locality_delta=locality,
                    continuation_gate=gain>=.05-1e-9 and natural>=-.02-1e-9 and locality>=-.02-1e-9)
        pins();output=ROOT/'results/v13-answerable-reading-20261026';output.mkdir(exist_ok=False)
        write_json(output/'comparison.json',dict(status='complete',endpoint=512,cells=cells,comparisons=comparisons,
            confirmation_evaluated=False,reserved_test_accessed=False))
        update(status='success',phase='complete',current_job=None,finished=time.time())
    except BaseException as error:
        update(status='failed',phase='failed',error=str(error),traceback=traceback.format_exc(),finished=time.time());raise

if __name__=='__main__':main()
