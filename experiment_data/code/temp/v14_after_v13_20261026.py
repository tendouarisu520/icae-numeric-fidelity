"""At most two prespecified followups, only after the original eight-cell study."""
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
RUNS=ROOT/'runs/v14-bottleneck-reading-20261026'
OLD=ROOT/'runs/v13-answerable-reading-20261026'
LOGS=ROOT/'logs/v14-bottleneck-reading-20261026'

def read_json(path):
    for attempt in range(6):
        try:return json.loads(path.read_text(encoding='utf8'))
        except (json.JSONDecodeError,PermissionError):
            if attempt==5:raise
            time.sleep(.2)

def main():
    os.chdir(ROOT);sys.path.insert(0,str(ROOT/'src'))
    from binding.v11_pipeline import write_json,read_rows,sha
    from binding.v14_protocol import should_run
    from binding.v13_error_profile import profile,paired_intervals
    plan_path=ROOT/'docs/v14-run-plan-20261026.json';plan=read_json(plan_path);plan_sha=sha(plan_path)
    def pins():
        if sha(plan_path)!=plan_sha:raise ValueError('V14 plan changed')
        if any(sha(ROOT/n)!=h for n,h in plan['source_files'].items()):raise ValueError('Pinned source changed')
        if any(sha(ROOT/n)!=h for n,h in plan['data_files'].items()):raise ValueError('Frozen training/development data changed')
    pins();RUNS.mkdir(exist_ok=False);LOGS.mkdir(exist_ok=False)
    state=dict(status='running',phase='preflight',pid=os.getpid(),completed=[],scientific_starts=0,
        maximum_scientific_starts=2,started=time.time(),confirmation_evaluated=False,reserved_test_accessed=False)
    def update(**values):
        state.update(values,updated=time.time());temporary=RUNS/'queue-status.tmp'
        write_json(temporary,state);temporary.replace(RUNS/'queue-status.json');print(json.dumps(state),flush=True)
    env=dict(os.environ);env.update(PYTHONPATH=str(ROOT/'src'),PYTHONDONTWRITEBYTECODE='1',PYTHONHASHSEED='73',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TEMP=str(ROOT/'temp'),TMP=str(ROOT/'temp'),
        HF_HOME=str(ROOT/'cache/huggingface'),TORCH_HOME=str(ROOT/'cache/torch'),
        CUDA_CACHE_PATH=str(ROOT/'cache/cuda'),TRITON_CACHE_DIR=str(ROOT/'cache/triton'),XDG_CACHE_HOME=str(ROOT/'cache/xdg'))
    env.pop('CUDA_VISIBLE_DEVICES',None)
    for name in ('TEMP','TMP','HF_HOME','TORCH_HOME','CUDA_CACHE_PATH','TRITON_CACHE_DIR','XDG_CACHE_HOME'):
        Path(env[name]).mkdir(parents=True,exist_ok=True)
    def resources():
        start=time.time()
        while True:
            free_disk=shutil.disk_usage(ROOT).free
            if free_disk<450_000_000:raise RuntimeError('Efree below450MB; no old artifacts deleted')
            buffer=ctypes.create_string_buffer(64);struct.pack_into('I',buffer,0,64)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(buffer):raise OSError('RAM status unavailable')
            ram=struct.unpack_from('Q',buffer,16)[0]
            result=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],
                capture_output=True,text=True,check=True)
            free=int(result.stdout.strip().splitlines()[0])
            if free>=23500 and ram>=12*1024**3:return
            update(phase='waiting_for_resources',free_gpu_mib=free,available_ram_bytes=ram,Efree=free_disk)
            if time.time()-start>12*3600:raise TimeoutError('Resources unavailable; no user process stopped')
            time.sleep(30)
    def cpu(pid):
        api=ctypes.windll.kernel32;api.OpenProcess.restype=ctypes.c_void_p
        handle=api.OpenProcess(0x1000,False,pid)
        if not handle:return None
        times=[wt.FILETIME() for _ in range(4)]
        value=sum((t.dwHighDateTime<<32)|t.dwLowDateTime for t in times[2:])/10_000_000 \
            if api.GetProcessTimes(ctypes.c_void_p(handle),*(ctypes.byref(t) for t in times)) else None
        api.CloseHandle(ctypes.c_void_p(handle));return value
    def run(name,arguments,scientific=False):
        pins();record=RUNS/(name+'-process.json');out=LOGS/(name+'.stdout.log');err=LOGS/(name+'.stderr.log')
        if any(p.exists() for p in (record,out,err)):raise ValueError('Existing job; no overwrite/retry')
        if scientific:
            if state['scientific_starts']>=2:raise RuntimeError('Two-cell budget exhausted')
            state['scientific_starts']+=1
        cmd=[str(ROOT/'envs/icae/Scripts/python.exe'),'-B',str(ROOT/'src/scripts/run_recorded.py'),
            '--record',str(record),'--','-B',*arguments]
        with out.open('x',encoding='utf8') as stdout,err.open('x',encoding='utf8') as stderr:
            child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
        update(phase='running',current_job=name,wrapper_pid=child.pid,stdout=str(out),stderr=str(err),process_record=str(record))
        deadline=time.time()+12*3600;last_cpu=None;last_log=None;stable=0
        while child.poll() is None:
            status=RUNS/name/'status.json';progress=read_json(status) if status.exists() else None
            actual=read_json(record) if record.exists() else {}
            pid=progress['pid'] if progress else actual.get('child_pid');current_cpu=cpu(pid) if pid else None
            latest=max(out.stat().st_mtime,err.stat().st_mtime);stale=time.time()-latest
            stable=stable+1 if current_cpu==last_cpu and latest==last_log else 0
            update(current_progress=progress,stdout_bytes=out.stat().st_size,stderr_bytes=err.stat().st_size,
                last_log_write=latest,stale_log_seconds=stale,child_cpu_seconds=current_cpu,
                diagnostic='stalled_cpu_and_log' if stable>=15 and stale>300 else 'progress_or_short_gap',
                recent_stderr=err.read_text(encoding='utf8',errors='replace')[-1500:] if stale>300 else None)
            last_cpu=current_cpu;last_log=latest
            if stable>=30 and stale>600:raise RuntimeError('Owned job CPU and logs unchanged10minutes; inspect:'+name)
            if time.time()>deadline:raise TimeoutError('Job exceeded12hours; inspect owned job:'+name)
            time.sleep(20)
        actual=read_json(record)
        if child.returncode or actual['status']!='success' or actual['exit_code']!=0:raise RuntimeError('Job failed:'+name)
        state['completed'].append(dict(job=name,exit_code=0,record_sha256=sha(record)));update(current_progress=None)
    update()
    try:
        run('unit-tests',['-m','unittest','discover','-s',str(ROOT/'src/tests'),'-p','test_v14*.py','-v'])
        tests=(LOGS/'unit-tests.stderr.log').read_text(encoding='utf8')
        if 'Ran 7 tests' not in tests or '\nOK\n' not in tests or 'skipped' in tests:raise ValueError('All7 real tests required with0skips')
        deadline=time.time()+48*3600
        while True:
            pins();original=read_json(OLD/'queue-status.json')
            if original['status']=='failed':raise RuntimeError('Original V13 queue failed:'+original.get('error','unknown'))
            if original['status']=='success':break
            progress=original.get('current_progress') or {}
            update(phase='waiting_for_original_eight',current_job=None,original_job=original.get('current_job'),
                original_step=progress.get('optimizer_steps'),original_phase=progress.get('phase'),
                original_updated=original.get('updated'),Efree=shutil.disk_usage(ROOT).free)
            if time.time()>deadline:raise TimeoutError('Original queue not complete after48hours; no GPU training started')
            time.sleep(30)
        source=ROOT/'results/v13-answerable-reading-20261026/comparison.json';comparison=read_json(source)
        decision=should_run(comparison);output=ROOT/'results/v14-bottleneck-reading-20261026';output.mkdir(exist_ok=False)
        write_json(output/'trigger-decision.json',dict(run_followup=decision,original_comparison_sha256=sha(source),
            comparisons=comparison['comparisons'],rule=plan['trigger']))
        if not decision:
            update(status='success',phase='skipped_original_goal_met',current_job=None,finished=time.time());return
        resources();run('smoke-compressed-bottleneck-seed73',['-m','binding.v14_training','--phase','smoke'])
        smoke=read_json(RUNS/'smoke-compressed-bottleneck-seed73/summary.json')
        if smoke['optimizer_steps']!=2 or smoke['source_replay_exact']!=2 or smoke['status']!='success' or \
            not all(smoke[k] for k in ('cache_unchanged','frozen_parameters_unchanged','reload_exact')):
            raise ValueError('Real7B smoke calibration failed')
        units=read_rows(ROOT/'data/v13-answerable-relations-20261026/development.jsonl');cells=[];comparisons={}
        for seed in (73,97):
            resources();name=f'main-compressed-bottleneck-seed{seed}'
            run(name,['-m','binding.v14_training','--phase','main','--seed',str(seed)],True)
            path=RUNS/name/'summary.json';s=read_json(path);base=read_json(OLD/f'main-compressed-sft-seed{seed}/summary.json')
            relation=read_json(OLD/f'main-compressed-relation-seed{seed}/summary.json')
            if s['status']!='success' or s['optimizer_steps']!=512 or not all(s[k] for k in ('cache_unchanged','frozen_parameters_unchanged','reload_exact')):
                raise ValueError('Fixed512 calibration failed:'+name)
            if s['initial_reader_identity']!=base['initial_reader_identity'] or s['cache_sha256']!=base['cache_sha256']:
                raise ValueError('Initial reader or frozen memories differ from original baseline')
            result=s['evaluations']['512'];metric_path=RUNS/name/'evaluation-512/metrics.json';metric=read_json(metric_path)
            raw=RUNS/name/'evaluation-512/predictions.jsonl'
            if sha(raw)!=metric['predictions_sha256']:raise ValueError('Predictions checksum changed')
            p=profile(read_rows(raw),units);write_json(output/(name+'-profile.json'),p)
            comparisons[str(seed)]={}
            for method,baseline in (('sft',base),('relation',relation)):
                old_profile_path=ROOT/f'results/v13-binding-error-analysis-20261026/main-compressed-{method}-seed{seed}--512.json'
                old_profile=read_json(old_profile_path) if old_profile_path.exists() else profile(
                    read_rows(OLD/f'main-compressed-{method}-seed{seed}/evaluation-512/predictions.jsonl'),units)
                b=baseline['evaluations']['512'];macro=lambda x:sum(m['rates']['target_joint'] for m in x['probe_by_level'].values())/3
                gain=macro(result)-macro(b);natural=result['natural']['known_f1']-b['natural']['known_f1']
                local=result['probe']['rates']['locality_joint']-b['probe']['rates']['locality_joint']
                comparisons[str(seed)][method]=dict(target_joint_gain=gain,natural_known_f1_delta=natural,locality_delta=local,
                    continuation_gate=gain>=.05-1e-9 and natural>=-.02-1e-9 and local>=-.02-1e-9,
                    paired_development_interval=paired_intervals(old_profile['source_scores'],p['source_scores']))
            cells.append(dict(seed=seed,arm='compressed',method='bottleneck',fixed512=result,
                best_development_step=s['best_development_step'],training_seconds=s['cumulative_training_seconds'],
                peak_allocated_mib=s['peak_allocated_mib'],summary_sha256=sha(path),errors=p['errors']))
            write_json(output/'comparison-partial.json',dict(status='partial',endpoint=512,cells=cells,comparisons=comparisons))
        pins();write_json(output/'comparison.json',dict(status='complete',endpoint=512,cells=cells,comparisons=comparisons,
            original_comparison_sha256=sha(source),two_seed_gate_vs_sft=all(comparisons[str(s)]['sft']['continuation_gate'] for s in (73,97)),
            confirmation_evaluated=False,reserved_test_accessed=False,automatic_additional_search=False))
        update(status='success',phase='complete',current_job=None,finished=time.time())
    except BaseException as error:
        update(status='failed',phase='failed',error=str(error),traceback=traceback.format_exc(),finished=time.time());raise

if __name__=='__main__':main()
