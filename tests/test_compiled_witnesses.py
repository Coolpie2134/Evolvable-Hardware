"""Exact production witnesses for the output-rooted LUT and nervous genomes."""

from runtime.config import nv_run_config
from substrates.lut.branched_synthesis import synthesize_branched_truth_table
from substrates.lut.ga import evaluate_lut_full
from substrates.lut.state_synthesis import (
    synthesize_branched_dynamic as synthesize_lut_dynamic)
from substrates.nervous.certification import certify
from substrates.nervous.ga import evaluate_nv_full
from substrates.nervous.logic_synthesis import synthesize_branched_logic
from substrates.nervous.state_synthesis import (
    synthesize_branched_dynamic as synthesize_nervous_dynamic)
from tools.benchmark import targets_for_backend


def test_every_lut_truth_table_compiles_to_exact_fitness():
    targets = targets_for_backend('lut')
    compiled = 0
    for target in targets.values():
        if not (getattr(target, 'combinational_cases', ())
                or getattr(target, 'temporal_logic_cases', ())):
            continue
        target.lut_io_mode = 'source_pads'
        genome = synthesize_branched_truth_table(
            target, chromosome_count=2, max_telomere=18)
        fitness, cases = evaluate_lut_full(genome, target)
        assert fitness == 1.0 and min(cases) == 1.0
        compiled += 1
    assert compiled == 30


def test_dynamic_lut_witnesses_are_exact():
    targets = targets_for_backend('lut')
    names = (
        'Oscillator', 'Pattern (1000)', 'Coincidence (2-in)',
        'Temporal XOR (2-in)', 'Sequence A->B', 'Veto gate', 'Burst x3',
        'Echo (delay 3)', 'Pair detector (gap 2)', 'Gated D latch',
        'SR latch')
    for name in names:
        target = targets[name]
        target.lut_io_mode = 'source_pads'
        genome = synthesize_lut_dynamic(target, max_telomere=18)
        fitness, cases = evaluate_lut_full(genome, target)
        assert fitness == 1.0 and min(cases) == 1.0


def test_nervous_logic_and_dynamic_witnesses_are_exact():
    targets = targets_for_backend('nervous', 'paper_analog')
    logic = (
        'Coincidence (2-in)', 'Temporal XOR (2-in)',
        'Half adder (temporal)', '2:1 MUX (temporal)',
        'Majority-3 (temporal)',
        'Parity-3 (XOR3) (temporal)', 'AND (temporal)', 'XOR (temporal)',
        'NAND (temporal)', 'NOR (temporal)', 'XNOR (temporal)')
    dynamic = (
        'Oscillator', 'Pattern (1000)', 'Burst x3',
        'Sequence A->B', 'Veto gate',
        'Echo (delay 3)', 'Pair detector (gap 2)',
        'Refractory filter (3 seconds)')
    for name in logic + dynamic:
        target = targets[name]
        target.pulse_config = nv_run_config().pulse
        genome = (synthesize_branched_logic(target, max_telomere=24)
                  if name in logic else
                  synthesize_nervous_dynamic(target, max_telomere=24))
        fitness, cases = evaluate_nv_full(genome, target)
        assert fitness == 1.0 and min(cases) == 1.0


def test_echo_oracle_applies_its_declared_delay_once_and_certifies():
    for backend, compiler, evaluator, model in (
            ('lut', synthesize_lut_dynamic, evaluate_lut_full, None),
            ('nervous', synthesize_nervous_dynamic,
             evaluate_nv_full, 'paper_analog')):
        target = targets_for_backend(backend, model)['Echo (delay 3)']
        if backend == 'lut':
            target.lut_io_mode = 'source_pads'
        else:
            target.pulse_config = nv_run_config().pulse
        genome = compiler(target)
        fitness, _cases = evaluator(genome, target)
        result = certify(genome, target, train=fitness, backend=backend)
        assert fitness == 1.0
        assert result['verdict'] == 'CERTIFIED'
        assert result['holdouts'] == [1.0, 1.0, 1.0]
