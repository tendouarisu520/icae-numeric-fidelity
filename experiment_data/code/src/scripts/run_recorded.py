"""Record background child PID and exact exit status independently of terminal."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--record', required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    path = Path(args.record).resolve()
    if path.exists() or path.drive.lower() != 'e:' or not args.command or args.command[0] != '--':
        raise ValueError('New E: record and -- child arguments required')
    command = [sys.executable] + args.command[1:]
    state = dict(parent_pid=os.getpid(), command=command, status='starting', started=time.time())
    def save():
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2), encoding='utf-8')
        temporary.replace(path)
    save()
    try:
        child = subprocess.Popen(command, stdout=sys.stdout, stderr=sys.stderr,
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        state.update(status='running', child_pid=child.pid); save()
        code = child.wait()
        state.update(status='success' if code == 0 else 'failed', exit_code=code)
    except BaseException:
        state.update(status='failed', traceback=traceback.format_exc(), exit_code=1)
        raise
    finally:
        state['finished'] = time.time(); save()
    raise SystemExit(state['exit_code'])


if __name__ == '__main__':
    main()
