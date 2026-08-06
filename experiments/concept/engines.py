"""
experiments/concept/engines.py - GUI bridge for the concept simulations.

Each concept sim is a self-driving CLI loop (`main()`). Rather than reimplement
each one's selection/crossover, we let `main()` run as-is in a background thread
and tap it through a tiny module-level hook: when `sim.GUI` is set to a GuiBridge,
`main()` streams a per-generation stats dict via `GUI.report(...)` and yields
control via `GUI.checkpoint()` (which supports pause/stop). When `sim.GUI` is
None the sim behaves exactly as the original CLI.

The genuinely-shared terminal I/O lives in experiments/concept/common/terminal.py (imported
by every sim); only the divergent selection/fitness code stays per-sim.

stats dict (produced by each sim's hook):
    generation, fitnesses(list), best_fitness, mean_fitness, max_fitness,
    best_grid (rows of ints), target_grid (rows of ints | None),
    best_stop, extra(dict)

board kind:  '2d' grid (proof7/multi9/...) or '1d' strip (linear12/species2).
"""
from __future__ import annotations
import queue
import threading
import importlib
import traceback


class _Stop(Exception):
    """Raised inside a sim's hook to unwind main() when the GUI stops it."""


class GuiBridge:
    """Owned by an engine, assigned to a sim's module-level GUI hook."""

    def __init__(self, maxqueue=4):
        self.q        = queue.Queue(maxsize=maxqueue)
        self._go      = threading.Event(); self._go.set()
        self._stopped = False

    # called from the sim thread -------------------------------------------------
    def report(self, stats):
        while True:
            if self._stopped:
                raise _Stop()
            try:
                self.q.put(stats, timeout=0.1)
                return
            except queue.Full:
                continue

    def checkpoint(self):
        if self._stopped:
            raise _Stop()
        while not self._go.wait(timeout=0.1):
            if self._stopped:
                raise _Stop()
        if self._stopped:
            raise _Stop()

    # called from the GUI thread -------------------------------------------------
    def pause(self):  self._go.clear()
    def resume(self): self._go.set()
    def stop(self):   self._stopped = True; self._go.set()


class HookEngine:
    """Runs one sim's main() in a thread, bridged to the GUI via GuiBridge."""

    def __init__(self, cfg):
        self.cfg        = cfg
        self.name       = cfg["name"]
        self.board_kind = cfg["board"]
        self.mod        = importlib.import_module(cfg["module"])
        self.bridge     = None
        self.thread     = None

    def start(self, popsize=None, paused=False):
        self.bridge = GuiBridge()
        if paused:
            self.bridge.pause()
        if popsize:
            setattr(self.mod, self.cfg["popsize_attr"], int(popsize))
        for attr, val in self.cfg.get("overrides", {}).items():
            setattr(self.mod, attr, val)
        self.mod.GUI = self.bridge
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            self.mod.main()          # sim silences its own output via NullIO fout
        except _Stop:
            pass
        except BaseException:
            try:
                self.bridge.q.put({"error": traceback.format_exc()}, timeout=1.0)
            except Exception:
                pass
        finally:
            self.mod.GUI = None

    @property
    def q(self):
        return self.bridge.q if self.bridge else None

    def pause(self):
        if self.bridge: self.bridge.pause()

    def resume(self):
        if self.bridge: self.bridge.resume()

    def stop(self):
        if self.bridge: self.bridge.stop()

    def is_running(self):
        return bool(self.thread and self.thread.is_alive())


# -- sim registry --------------------------------------------------------------
# popsize_attr: the module global the engine shrinks for interactivity.
# overrides:    other module globals to set before running (kept modest).

CONFIGS = {
    "proof7":    {"module": "experiments.concept.proof7",    "board": "2d",
                  "popsize_attr": "POPSIZE", "popsize_default": 150,
                  "overrides": {"SOLUTIONSPACE": 2048}},
    "multi9":    {"module": "experiments.concept.multi9",    "board": "2d",
                  "popsize_attr": "POPSIZE", "popsize_default": 150},
    "mhier1":    {"module": "experiments.concept.mhier1",    "board": "2d",
                  "popsize_attr": "POPSIZE", "popsize_default": 150},
    "coderack7": {"module": "experiments.concept.coderack7", "board": "2d",
                  "popsize_attr": "POPSIZE", "popsize_default": 60},
    "hierarch4": {"module": "experiments.concept.hierarch4", "board": "2d",
                  "popsize_attr": "POPSIZE", "popsize_default": 60},
    "linear12":  {"module": "experiments.concept.linear12",  "board": "1d",
                  "popsize_attr": "POPSIZE", "popsize_default": 200},
    "species2":  {"module": "experiments.concept.species2",  "board": "1d",
                  "popsize_attr": "POPSIZE", "popsize_default": 200},
}
for _name, _c in CONFIGS.items():
    _c["name"] = _name

DEFAULT_SIM = "multi9"


def make_engine(name):
    return HookEngine(CONFIGS[name])


def available_engines():
    """Return {name: (cfg_or_None, status)} probing import + GUI-hook support."""
    out = {}
    for name, cfg in CONFIGS.items():
        try:
            mod = importlib.import_module(cfg["module"])
            if getattr(mod, "GUI_READY", False):
                out[name] = (cfg, "available")
            else:
                out[name] = (None, "no GUI hook yet")
        except Exception as exc:
            out[name] = (None, "unavailable (%s: %s)" % (type(exc).__name__, exc))
    return out
