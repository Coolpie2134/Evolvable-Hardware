# Evolvable Hardware

Evolve developmental rules that grow and simulate asynchronous circuits. The
desktop app supports Nervous net (analog tri-circuit), Functional NV Net (FNV),
LUT, and spiking-neuron substrates.

## Start here

1. Download and extract the repository, or clone it.
2. Install Python 3.10 or newer, including Tkinter for the desktop interface.
3. Open a terminal in this folder and install the dependencies:

   ```sh
   python -m venv .venv
   ```

   Activate the environment on Windows:

   ```bat
   .venv\Scripts\activate
   ```

   Or on macOS/Linux:

   ```sh
   source .venv/bin/activate
   ```

   Then install and launch:

   ```sh
   python -m pip install -r requirements.txt
   python app.py
   ```

On Windows, `py` can be used instead of `python` to create the environment.
Use Command Prompt if PowerShell blocks the activation script. Linux Python
installations may need the operating system's Tkinter package (`python3-tk` on
Debian/Ubuntu). Check Tkinter with `python -m tkinter`.

## Run an experiment

In the app, choose a substrate and target, set the population and generation
budget, and start the run. Use Stop to finish early and Load Saved to reopen a
checkpoint. Runs and exported figures are written to `results/` automatically.
You do not need any previously generated results to use the application.

For terminal benchmarks:

```sh
python benchmark.py --list-architectures
python benchmark.py --list-targets --architectures nervous,fnv,lut
python benchmark.py --architectures nervous,fnv,lut --targets "Half adder (temporal)" --seeds 3 --gens 200 --pop 60 --dry-run
```

Remove `--dry-run` to execute the displayed plan. Use `python benchmark.py --help`
for output locations, resume options, worker counts, and other settings.
Training fitness and held-out certification are separate results; a perfect
training score alone does not establish generalization.

## Files and folders

| Path | Purpose |
| --- | --- |
| `app.py` | Start the desktop application. |
| `benchmark.py` | Run terminal benchmarks. |
| `run_tests.py` | Run the test suite without installing pytest. |
| `requirements.txt` | Python dependencies. |
| `ui/` | Desktop interface and circuit visualization. |
| `runtime/` | Shared evolution controller, configuration, checkpoints, and workers. |
| `substrates/` | Circuit representations, developmental growth, and simulators. |
| `tests/` | Automated tests and test fixtures. |
| `tools/` | Additional maintained audit and analysis utilities. |
| `experiments/` | Historical concept/reference implementations; not needed to launch the app. |
| `results/` | Generated local output; excluded from Git. |

Additional utilities are listed in [tools/README.md](tools/README.md). The
application code stays in its packages; the root launchers provide the common
commands without duplicating implementation.

## Check the installation

```sh
python run_tests.py
```

The suite includes simulator physics, scoring, growth, checkpoint, and UI tests.
GUI tests need a working desktop display. Pytest is optional.

Nervous runs use `tri3` with `paper_analog`. Checkpoints requiring retired digital
NV engines are rejected rather than silently replayed under different physics.

## Sharing a download

Use the repository's source download or `git archive` to share the tracked code.
Do not include `.venv/`, `tmp/`, caches, or your local `results/` folder in a
manual ZIP. These are generated locally and are not installation requirements.
