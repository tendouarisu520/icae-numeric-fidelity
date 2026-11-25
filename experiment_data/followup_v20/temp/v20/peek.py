"""Read-only progress view for the V20 queue. Reads append-only logs only; never opens a status file that a
running job replaces."""
from pathlib import Path
import json
import subprocess
import sys
import time

R = Path('E:/Research/ICAE-ConditionalMemory-20261021')


def last_json(path):
    try:
        with open(path, 'rb') as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 65536))
            lines = f.read().decode('utf8', 'replace').splitlines()
    except OSError as e:
        return dict(error=str(e))
    for line in reversed(lines):
        line = line.strip()
        if line.startswith('{') and line.endswith('}'):
            try:
                return json.loads(line)
            except ValueError:
                continue
    return {}


def main():
    logs = sorted((R / 'logs').glob('v20-queue-*.out.log'))
    if not logs:
        print('Q none')
        return
    q = last_json(logs[-1])
    now = time.time()
    done = q.get('completed', [])
    print('Q phase=%s job=%s age=%ds up=%dmin done=%d fail=%d pid=%s child=%s' % (
        q.get('phase'), q.get('job'), now - q.get('time', now), (now - q.get('queue_started', now)) / 60, len(done),
        len(q.get('failures', [])), q.get('pid'), q.get('child_pid')))
    for d in done[-12:]:
        print('D %s %.1fmin' % (d['job'], d['seconds'] / 60))
    for f in q.get('failures', []):
        print('F %s %s' % (f.get('part'), str(f.get('error'))[:200]))
    if q.get('error'):
        print('E %s' % str(q['error'])[:300])
    if q.get('stdout') and q.get('phase') == 'running':
        j = last_json(q['stdout'])
        keys = ('status', 'phase', 'mode', 'job', 'cohort', 'optimizer_steps', 'units', 'total_units', 'cells_done',
                'evaluation_units', 'evaluation_total', 'cache_completed', 'cache_total', 'error')
        print('J ' + ' '.join('%s=%s' % (k, j[k]) for k in keys if k in j)[:230])
        stamp = j.get('updated', j.get('time'))
        if stamp:
            print('J age=%ds' % (now - stamp))
        err = Path(q['stderr'])
        if err.exists() and err.stat().st_size:
            print('S stderr_bytes=%d' % err.stat().st_size)
    try:
        g = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
        print('G used_mib,util=' + g.strip())
    except Exception as e:  # noqa: BLE001
        print('G error %r' % (e,))
    if '--tail' in sys.argv and q.get('stderr'):
        text = Path(q['stderr']).read_text(encoding='utf8', errors='replace')[-1500:]
        for line in text.splitlines()[-12:]:
            print('| ' + line[:110])


if __name__ == '__main__':
    main()
