"""Bounded sequential fixed-memory reader queue with source and runtime gates."""
import ctypes
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time
import traceback

ROOT=Path('E:/Research/ICAE-ConditionalMemory-20261021')
RUNS=ROOT/'runs/v12-frozen-reader-20261025';LOGS=ROOT/'logs/v12-frozen-reader-20261025'


def main():
    os.chdir(ROOT);sys.path.insert(0,str(ROOT/'src'))
    from binding.v11_pipeline import sha,write_json
    plan=json.loads((ROOT/'docs/v12-run-plan-20261025.json').read_text(encoding='utf8'))
    def check_sources():
        if any(sha(ROOT/n)!=h for n,h in plan['source_files'].items()):raise ValueError('Pinned source changed')
        if sha(ROOT/'data/v11-natural-frozen-20261025/manifest.json')!=plan['cohort_manifest_sha256']:raise ValueError('Cohort changed')
    check_sources()
    if json.loads((ROOT/'runs/v11-natural-20261025/queue-status.json').read_text(encoding='utf8'))['status']!='success':
        raise ValueError('Six original cells must finish first')
    RUNS.mkdir(exist_ok=False);LOGS.mkdir(exist_ok=False)
    state=dict(status='running',phase='preflight',pid=os.getpid(),scientific_starts=0,maximum_scientific_starts=6,completed=[],started=time.time())
    def update(**values):
        state.update(values,updated=time.time());write_json(RUNS/'queue-status.json',state)
        print(json.dumps({k:state.get(k) for k in ('status','phase','pid','current_job','current_progress','scientific_starts','updated','error')}),flush=True)
    env=dict(os.environ);env.update(PYTHONPATH=str(ROOT/'src'),PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED='73',HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',TEMP=str(ROOT/'temp'),TMP=str(ROOT/'temp'),HF_HOME=str(ROOT/'cache/huggingface'),TORCH_HOME=str(ROOT/'cache/torch'))
    env.pop('CUDA_VISIBLE_DEVICES',None)
    def resources():
        start=time.time()
        while True:
            if shutil.disk_usage(ROOT).free<450_000_000:raise RuntimeError('E free below450MB; do not delete results')
            buffer=ctypes.create_string_buffer(64);struct.pack_into('I',buffer,0,64)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(buffer):raise OSError('Cannot inspect RAM')
            ram=struct.unpack_from('Q',buffer,16)[0]
            result=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
            gpu=int(result.stdout.strip().splitlines()[0])
            if gpu>=23500 and ram>=8*1024**3:return
            update(phase='waiting_for_resources',free_gpu_mib=gpu,available_ram_bytes=ram)
            if time.time()-start>12*3600:raise TimeoutError('Resources unavailable; no user process stopped')
            time.sleep(30)
    def run(name,arguments,scientific=False):
        check_sources();record=RUNS/(name+'-process.json');out=LOGS/(name+'.stdout.log');err=LOGS/(name+'.stderr.log')
        if any(p.exists() for p in (record,out,err)):raise ValueError('Existing job; do not overwrite or retry')
        if scientific:
            if state['scientific_starts']>=6:raise RuntimeError('Six-cell budget exhausted')
            state['scientific_starts']+=1
        command=[str(ROOT/'envs/icae/Scripts/python.exe'),'-B',str(ROOT/'src/scripts/run_recorded.py'),'--record',str(record),'--','-B',*arguments]
        with out.open('x',encoding='utf8') as stdout,err.open('x',encoding='utf8') as stderr:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
        update(phase='running',current_job=name,wrapper_pid=child.pid,stdout=str(out),stderr=str(err),process_record=str(record))
        deadline=time.time()+12*3600
        while child.poll() is None:
            path=RUNS/name/'status.json';progress=json.loads(path.read_text(encoding='utf8')) if path.exists() else None
            last=max(out.stat().st_mtime,err.stat().st_mtime);stale=time.time()-last
            cpu=None
            if progress is not None:
                # CPU movement is sampled through Windows' process interface without adding dependencies.
                import ctypes.wintypes as wt
                api=ctypes.windll.kernel32;api.OpenProcess.restype=ctypes.c_void_p
                handle=api.OpenProcess(0x1000,False,progress['pid'])
                if handle:
                    times=[wt.FILETIME() for _ in range(4)]
                    if api.GetProcessTimes(ctypes.c_void_p(handle),*(ctypes.byref(t) for t in times)):
                        cpu=sum((t.dwHighDateTime<<32)|t.dwLowDateTime for t in times[2:])/10_000_000
                    api.CloseHandle(ctypes.c_void_p(handle))
            update(current_progress=progress,stdout_bytes=out.stat().st_size,stderr_bytes=err.stat().st_size,
                last_log_write=last,stale_log_seconds=stale,possible_stall=stale>300,child_cpu_seconds=cpu)
            if time.time()>deadline:raise TimeoutError('Job exceeded12hours; inspect owned child:'+name)
            time.sleep(20)
        actual=json.loads(record.read_text(encoding='utf8'))
        if child.returncode or actual['status']!='success' or actual['exit_code']!=0:raise RuntimeError('Job failed:'+name)
        state['completed'].append(dict(job=name,record_sha256=sha(record),exit_code=0));update(current_progress=None)
    update()
    try:
        if shutil.disk_usage(ROOT).free<700_000_000:raise RuntimeError('Initial Efree below700MB')
        run('unit-tests',['-m','unittest','discover','-s',str(ROOT/'src/tests'),'-p','test_v12*.py','-v'])
        text=(LOGS/'unit-tests.stderr.log').read_text(encoding='utf8')
        if 'Ran 8 tests' not in text or '\nOK\n' not in text or 'skipped' in text:raise ValueError('All8 tests must run including neural tests')
        for phase,source,seed in [('smoke','sft512',73)]+[('main',source,seed) for seed in (73,97) for source in ('sft512','sft1024','relation1024')]:
            resources();name=f'{phase}-{source}-seed{seed}'
            run(name,['-m','binding.v12_training','--phase',phase,'--source',source,'--seed',str(seed)],phase=='main')
            summary=json.loads((RUNS/name/'summary.json').read_text(encoding='utf8'))
            if not all(summary[k] for k in ('cache_unchanged','source_checkpoints_unchanged','frozen_parameters_unchanged','reload_exact')):
                raise ValueError('Frozen/reload calibration failed')
            if summary['optimizer_steps']!=(2 if phase=='smoke' else 1024):raise ValueError('Step budget differs')
        check_sources();cells=[]
        for seed in (73,97):
            for source in ('sft512','sft1024','relation1024'):
                summary=json.loads((RUNS/f'main-{source}-seed{seed}'/'summary.json').read_text(encoding='utf8'))
                cells.append(dict(source=source,seed=seed,baseline=summary['baseline'],
                    trained=summary['evaluations']['1024']['development'],diagnostics=summary['diagnostics'],
                    training_seconds=summary['cumulative_training_seconds'],peak_allocated_mib=summary['peak_allocated_mib']))
        out=ROOT/'results/v12-frozen-reader-comparison-20261025';out.mkdir(exist_ok=False)
        write_json(out/'comparison.json',dict(status='complete',fixed_reader_endpoint=1024,cells=cells,
            confirmation_evaluated=False,reserved_test_accessed=False))
        update(status='success',phase='complete',current_job=None,finished=time.time())
    except BaseException as error:
        update(status='failed',phase='failed',error=str(error),traceback=traceback.format_exc(),finished=time.time());raise


if __name__=='__main__':main()
