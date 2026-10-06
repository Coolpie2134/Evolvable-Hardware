"""Console-independent campaign launcher with bounded crash recovery.

python tools/supervise_campaign.py --launch
The campaign's STOP file pauses normally; it is never overridden.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/multi_target_600gen_20260915'


def supervise(command, directory, delay=30, poll=15, max_restarts=3):
    """Retain each exit code and retry at most three unexpected exits."""
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    with (directory / 'supervisor-events.jsonl').open('a', encoding='utf-8') as events:
        def record(**data):
            events.write(json.dumps(dict(time=time.time(), supervisor_pid=os.getpid(), **data)) + '\n')
            events.flush()
        for attempt in range(max_restarts + 1):
            if (directory / 'STOP').exists():
                record(event='paused')
                return 0
            log_path = directory / f'supervised-{stamp}-{attempt}.log'
            with log_path.open('w', encoding='utf-8') as log:
                child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, close_fds=True,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                record(event='started', child_pid=child.pid, attempt=attempt, log=str(log_path))
                while child.poll() is None:
                    record(event='alive', child_pid=child.pid)
                    time.sleep(poll)
                record(event='exited', child_pid=child.pid, returncode=child.returncode)
            if child.returncode == 0 or (directory / 'STOP').exists():
                return child.returncode
            if attempt < max_restarts:
                time.sleep(delay)
        record(event='restart_limit_reached')
        return child.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.launch:
        with (OUT / 'supervisor-launch.log').open('a', encoding='utf-8') as log:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve())],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                close_fds=True, creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
        print(f'Detached supervisor PID: {child.pid}', flush=True)
        return 0
    import msvcrt
    with (OUT / 'supervisor.lock').open('a+b') as lock:
        lock.seek(0)
        if not lock.read(1):
            lock.write(b'0')
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        return supervise([sys.executable, str(ROOT / 'tools/multi_target_campaign.py'), '--retry-errors'], OUT)


if __name__ == '__main__':
    raise SystemExit(main())
