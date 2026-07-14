"""
tests/run_tests.py — run the whole test suite with bare Python (no pytest needed).

    py tests/run_tests.py

Discovers every tests/test_*.py, runs its `test_*` functions, and reports a
combined pass/fail summary. Exit code is non-zero if any test fails, so it can
gate a commit or CI step. (If pytest is installed, `py -m pytest` also works.)
"""
import importlib.util
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
    spec.loader.exec_module(module)
    return module


def main():
    files = sorted(f for f in os.listdir(HERE)
                   if f.startswith('test_') and f.endswith('.py'))
    total = passed = 0
    failures = []
    for fname in files:
        module = _load(os.path.join(HERE, fname))
        tests = [v for k, v in sorted(vars(module).items())
                 if k.startswith('test_') and callable(v)]
        print("\n%s  (%d tests)" % (fname, len(tests)))
        for fn in tests:
            total += 1
            try:
                fn()
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
