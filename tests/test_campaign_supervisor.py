import json
from pathlib import Path
import sys
import tempfile
import unittest
from tools.supervise_campaign import supervise


class SupervisorTests(unittest.TestCase):
    def test_crash_is_logged_and_resumed(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            script = path / 'child.py'
            script.write_text('from pathlib import Path\nimport sys\np=Path(__file__).with_suffix(".count")\nn=int(p.read_text()) if p.exists() else 0\np.write_text(str(n+1))\nsys.exit(7 if n==0 else 0)\n')
            self.assertEqual(supervise([sys.executable, str(script)], path, delay=0, poll=.01), 0)
            events = [json.loads(line) for line in (path/'supervisor-events.jsonl').read_text().splitlines()]
            self.assertEqual([e['returncode'] for e in events if e['event']=='exited'], [7, 0])

    def test_persistent_failure_is_bounded(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            self.assertEqual(supervise([sys.executable, '-c', 'raise SystemExit(8)'], path, delay=0, poll=.01, max_restarts=1), 8)
            events = [json.loads(line) for line in (path/'supervisor-events.jsonl').read_text().splitlines()]
            self.assertEqual(sum(e['event']=='started' for e in events), 2)
            self.assertEqual(events[-1]['event'], 'restart_limit_reached')

    def test_stop_file_is_respected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path/'STOP').touch()
            self.assertEqual(supervise(['must-not-launch'], path), 0)
