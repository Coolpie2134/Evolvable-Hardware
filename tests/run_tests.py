"""
tests/run_tests.py - run the whole test suite with bare Python (no pytest needed).

    py tests/run_tests.py

Discovers every tests/test_*.py, runs its `test_*` functions, and reports a
combined pass/fail summary. Exit code is non-zero if any test fails, so it can
gate a commit or CI step. (If pytest is installed, `py -m pytest` also works.)
"""
import importlib.util
import inspect
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def _load(path):
    spec = importlib.util.spec_from_file_location(
        "suite_" + os.path.splitext(os.path.basename(path))[0], path)
    module = importlib.util.module_from_spec(spec)
    # importlib's low-level loader does not register the module for us.
    # Dataclasses and other runtime type resolvers look up cls.__module__ in
    # sys.modules while the file is executing, so bare-runner modules must be
    # visible there just like modules loaded through normal import machinery.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _MonkeyPatch:
    """Minimal pytest-compatible setattr fixture for the bare-Python runner."""

    def __init__(self):
        self._changes = []

    def setattr(self, target, name, value):
        self._changes.append((target, name, getattr(target, name)))
        setattr(target, name, value)

    def undo(self):
        for target, name, value in reversed(self._changes):
            setattr(target, name, value)
        self._changes.clear()


def _run(function):
    parameters = tuple(inspect.signature(function).parameters)
    if not parameters:
        function()
        return
    if parameters == ('monkeypatch',):
        monkeypatch = _MonkeyPatch()
        try:
            function(monkeypatch)
        finally:
            monkeypatch.undo()
        return
    raise TypeError(
        'bare runner does not provide fixture(s): %s' %
        ', '.join(parameters))


def main():
    files = sorted(f for f in os.listdir(HERE)
                   if f.startswith('test_') and f.endswith('.py'))
    total = passed = 0
    failures = []
    for fname in files:
        # A module that cannot even import is one failure, not a dead sweep:
        # the remaining files still have to report.
        try:
            module = _load(os.path.join(HERE, fname))
        except Exception as e:                     # noqa: BLE001
            total += 1
            print("\n%s  (failed to import)" % fname)
            print("  ERROR  %s: %s" % (type(e).__name__, e))
            failures.append("%s\n%s" % (fname, traceback.format_exc()))
            continue
        tests = [v for k, v in sorted(vars(module).items())
                 if k.startswith('test_') and callable(v)]
        print("\n%s  (%d tests)" % (fname, len(tests)))
        for fn in tests:
            total += 1
            try:
                _run(fn)
                print("  PASS  %s" % fn.__name__)
                passed += 1
            except Exception as e:                 # noqa: BLE001
                kind = "FAIL" if isinstance(e, AssertionError) else "ERROR"
                print("  %s  %s: %s" % (kind, fn.__name__, e))
                failures.append("%s::%s\n%s" % (fname, fn.__name__,
                                                traceback.format_exc()))
    print("\n" + "=" * 50)
    print("%d/%d tests passed across %d files" % (passed, total, len(files)))
    if failures:
        print("\nFailures:\n" + "\n".join(failures))
    return 0 if passed == total else 1


if __name__ == '__main__':
    raise SystemExit(main())
