"""V20 queue: one bounded serial queue. No retries, no restarts, nothing done to other processes.

Order
  preflight (CPU) -> smoke (GPU, two training-split units)
  part A  crossed-view choice scoring
  part B  32-update replay of the original seed-73 runs (sft, bottleneck), then seeds 101, 131, 151 (sft, bottleneck)
  part C  readouts of the three new SFT readers
  collect (CPU)   frozen analysis and evidence archive
Parts A and B are independent: a failure in part A is recorded and part B still runs. Part C needs all of part B.
A resource gate precedes every GPU job; if the GPU, RAM or disk is not free within 30 minutes the part stops and
nothing else on the machine is touched.
"""
from common import *
import ctypes
import shutil
import subprocess
import sys
import traceback


def resources():
    text = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True)
    free = int(text.strip().splitlines()[0])

    class Memory(ctypes.Structure):
        _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong)] + \
                   [(k, ctypes.c_ulonglong) for k in ('totalPhys', 'availPhys', 'totalPage', 'availPage',
                                                      'totalVirtual', 'availVirtual', 'availExtended')]
    m = Memory()
    m.dwLength = ctypes.sizeof(m)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return dict(gpu_free_mib=free, disk_free_gib=round(shutil.disk_usage(R).free / 1024 ** 3, 2),
                ram_free_gib=round(m.availPhys / 1024 ** 3, 2))


def main():
    environment()
    O.mkdir(parents=True, exist_ok=True)
    verify_lock()
    state = O / 'queue_status.json'
    assert not state.exists(), 'Queue already exists; inspect before resuming'
    env = dict(os.environ)
    env.update(PYTHONPATH=str(R / 'src'), PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='73',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    env.pop('CUDA_VISIBLE_DEVICES', None)
    done = []
    failures = []
    started = time.time()

    def beat(**values):
        emit(state, queue_started=started, completed=done, failures=failures, **values)

    def gate(label):
        waited = time.time()
        while True:
            res = resources()
            if res['gpu_free_mib'] >= 23500 and res['ram_free_gib'] >= 12 and res['disk_free_gib'] >= 20:
                return res
            beat(phase='waiting_for_resources', job=label, resources=res)
            if time.time() - waited > 1800:
                raise RuntimeError(f'Resources not free for {label} after 30 minutes: {res}; nothing was stopped')
            time.sleep(60)

    def run(label, args, gpu=False, output=None):
        verify_lock()
        res = gate(label) if gpu else resources()
        stamp = time.strftime('%Y%m%d-%H%M%S')
        log = R / f'logs/v20-{label}-{stamp}'
        begin = time.time()
        with Path(str(log) + '.log').open('x') as stdout, Path(str(log) + '.err.log').open('x') as stderr:
            p = subprocess.Popen([sys.executable, '-X', 'utf8', '-B', *args], cwd=R, env=env, stdout=stdout,
                                 stderr=stderr, creationflags=0x08000000)
            while True:
                beat(phase='running', job=label, child_pid=p.pid, output=output, stdout=str(log) + '.log',
                     stderr=str(log) + '.err.log', job_started=begin, resources_at_start=res)
                try:
                    p.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    pass
        if p.returncode:
            raise RuntimeError(f'{label} exited {p.returncode}; see {log}.err.log; no retry')
        done.append(dict(job=label, seconds=round(time.time() - begin, 1), child_pid=p.pid, exit_code=0,
                         stdout=str(log) + '.log'))
        beat(phase='job_complete', job=label)

    run('preflight', ['temp/v20/preflight.py'], output=str(O / 'preflight.json'))
    run('smoke', ['temp/v20/evaluate.py', 'smoke'], True, str(O / 'smoke'))
    try:
        run('crossed-choice', ['temp/v20/evaluate.py', 'crossed_choice'], True, str(O / 'crossed_choice'))
    except Exception as e:
        failures.append(dict(part='A', error=str(e)))
        beat(phase='part_failed', job='crossed-choice')
    try:
        for method in ('sft', 'bottleneck'):
            run(f'replay-{method}-seed73', ['temp/v20/seed_train.py', 'replay', method, '73'], True,
                str(RUNS / 'replay32'))
            assert read(RUNS / 'replay32' / f'replay_check-{method}.json')['pass']
        for seed in NEW_SEEDS:
            pair = {}
            for method in ('sft', 'bottleneck'):
                name = f'main-compressed-{method}-seed{seed}'
                assert not (RUNS / name).exists()
                run(name, ['temp/v20/seed_train.py', 'main', method, str(seed)], True, str(RUNS / name))
                s = read(RUNS / name / 'summary.json')
                assert s['status'] == 'success' and s['optimizer_steps'] == 512 and s['seed'] == seed
                assert all(s[k] for k in ('cache_unchanged', 'frozen_parameters_unchanged', 'reload_exact'))
                pair[method] = s
            assert pair['sft']['initial_reader_identity'] == pair['bottleneck']['initial_reader_identity']
            assert pair['sft']['cache_sha256'] == pair['bottleneck']['cache_sha256']
        run('new-seed-readouts', ['temp/v20/evaluate.py', 'new_seeds'], True, str(O / 'new_seeds'))
    except Exception as e:
        failures.append(dict(part='B/C', error=str(e)))
        beat(phase='part_failed', job='seed-extension')
    run('collect', ['temp/v20/collect.py'], output=str(O / 'completed_evidence.zip'))
    beat(phase='complete' if not failures else 'complete_with_failures', finished=time.time())


if __name__ == '__main__':
    try:
        main()
    except BaseException as e:
        state = O / 'queue_status.json'
        previous = read(state) if state.exists() else {}
        if previous.get('phase') not in ('complete', 'complete_with_failures') and previous.get('pid') in (None, os.getpid()):
            emit(state, phase='failed', error=repr(e), traceback=traceback.format_exc()[-3000:],
                 completed=previous.get('completed', []), failures=previous.get('failures', []))
        raise
