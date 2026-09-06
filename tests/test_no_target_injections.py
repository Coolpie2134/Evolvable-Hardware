"""Guardrails for unaided evolutionary benchmarks."""

import inspect
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from substrates.fnv import ga as fnv_ga
from substrates.lut import ga as lut_ga
from substrates.nervous import ga as nervous_ga
from substrates.snn import ga as snn_ga


def test_breeders_cannot_accept_prebuilt_rescue_genomes():
    for breeder in (
            fnv_ga.next_population, lut_ga.next_population,
            nervous_ga.next_population, snn_ga.next_population):
        assert 'rescue_candidates' not in inspect.signature(breeder).parameters


def test_target_compiler_modules_are_absent():
    forbidden = (
        'substrates/lut/synthesis.py',
        'substrates/lut/branched_synthesis.py',
        'substrates/lut/state_synthesis.py',
        'substrates/nervous/branched_synthesis.py',
        'substrates/nervous/logic_synthesis.py',
        'substrates/nervous/state_synthesis.py',
    )
    assert not [name for name in forbidden if (ROOT / name).exists()]


def test_fnv_reproduction_has_no_target_answer_channel():
    parameters = inspect.signature(fnv_ga.next_population).parameters
    assert 'target' not in parameters
    assert 'focus_families' not in parameters
    source = inspect.getsource(fnv_ga.next_population)
    assert 'preferred_signature' not in source
    assert 'evaluate_functional_full' not in source


def test_fnv_has_no_constructor_or_mutation_terminal_tropism():
    source = (ROOT / 'substrates/fnv/construction_ga.py').read_text(
        encoding='utf-8')
    forbidden = (
        '_connect_terminal_step',
        'LOGIC_SCAFFOLD_GENES',
        'MAX_LOGIC_SCAFFOLD_GENES',
        'FEEDBACK_CLOSE_PROBABILITY',
    )
    assert not [name for name in forbidden if name in source]
