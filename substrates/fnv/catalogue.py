"""Permanent component catalogue for the Functional NV Net (FNV).

Every entry is a physically distinct component-and-routing form.  The numeric
IDs are therefore part of the substrate definition: existing entries are never
renumbered and new entries may only be appended.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
from typing import Iterable

DIRECTIONS = ("L", "R", "D")

LOGIC = "LOGIC"
DELAY = "DELAY"
NORMALIZER = "NORMALIZER"
HOLD = "HOLD"
C_ELEMENT = "C_ELEMENT"
TOGGLE = "TOGGLE"
GATED_OSCILLATOR = "GATED_OSCILLATOR"

FAMILIES = (
    LOGIC,
    DELAY,
    NORMALIZER,
    HOLD,
    C_ELEMENT,
    TOGGLE,
    GATED_OSCILLATOR,
)
DEFAULT_FAMILIES = frozenset(FAMILIES)


@dataclass(frozen=True)
class ComponentType:
    """One immutable catalogue entry.

    ``inputs`` is ordered when the behavior has distinct A/B roles (VETO) and
    otherwise follows the catalogue's canonical direction order.  Unary forms
    have one input.  A component has one internal output value which is driven
    onto every direction in ``outputs``.
    """

    id: int
    name: str
    family: str
    behavior: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    delay: int = 1
    duration: int = 0
    high_time: int = 0
    low_time: int = 0


_components: list[ComponentType] = [
    ComponentType(0, "EMPTY", "EMPTY", "EMPTY", (), (), delay=0)
]


def _add(name: str, family: str, behavior: str, inputs: Iterable[str],
         outputs: Iterable[str], *, delay: int = 1, duration: int = 0,
         high_time: int = 0, low_time: int = 0) -> None:
    _components.append(ComponentType(
        len(_components), name, family, behavior, tuple(inputs), tuple(outputs),
        delay=delay, duration=duration, high_time=high_time,
        low_time=low_time))


def _remaining(*used: str) -> tuple[str, ...]:
    return tuple(direction for direction in DIRECTIONS if direction not in used)


# IDs 1..15: two-input logic.  Symmetric gates have three output-direction
# variants; VETO has both A/B role assignments for each output direction.
for _behavior in ("AND", "OR", "XOR"):
    for _out in DIRECTIONS:
        _ins = _remaining(_out)
        _add(f"{_behavior}_{''.join(_ins)}_TO_{_out}", LOGIC, _behavior,
             _ins, (_out,))
for _out in DIRECTIONS:
    _a, _b = _remaining(_out)
    _add(f"VETO_{_a}_NOT_{_b}_TO_{_out}", LOGIC, "VETO",
         (_a, _b), (_out,))
    _add(f"VETO_{_b}_NOT_{_a}_TO_{_out}", LOGIC, "VETO",
         (_b, _a), (_out,))


def _add_unary_forms(prefix: str, family: str, behavior: str, *,
                     duration: int = 0, high_time: int = 0,
                     low_time: int = 0) -> None:
    """Append the nine routes of one fixed unary component form.

    Each of three input directions has two one-output variants and one
    two-output variant.
    """
    for _inp in DIRECTIONS:
        _outs = _remaining(_inp)
        for _out in _outs:
            _add(f"{prefix}_{_inp}_TO_{_out}", family, behavior,
                 (_inp,), (_out,), duration=duration,
                 high_time=high_time, low_time=low_time)
        _add(f"{prefix}_{_inp}_TO_{''.join(_outs)}", family, behavior,
             (_inp,), _outs, duration=duration,
             high_time=high_time, low_time=low_time)


# IDs 16..33: exact edge transport delays.
for _duration in (1, 2):
    _add_unary_forms(f"DELAY{_duration}", DELAY, "DELAY",
                     duration=_duration)

# IDs 34..51: one pulse per inactive->active episode, with canonical width.
for _duration in (1, 2):
    _add_unary_forms(f"NORMALIZER{_duration}", NORMALIZER, "NORMALIZER",
                     duration=_duration)

# IDs 52..69: transport the level and retain it for N ticks after input falls.
for _duration in (1, 2):
    _add_unary_forms(f"HOLD{_duration}", HOLD, "HOLD",
                     duration=_duration)

# IDs 70..72: level-sensitive Muller C-elements.
for _out in DIRECTIONS:
    _ins = _remaining(_out)
    _add(f"C_ELEMENT_{''.join(_ins)}_TO_{_out}", C_ELEMENT, "C_ELEMENT",
         _ins, (_out,))

# IDs 73..81: rising-edge T flip-flops.
_add_unary_forms("TOGGLE", TOGGLE, "TOGGLE")

# IDs 82..117: enable-gated oscillators.  Low enable always returns to low;
# there is deliberately no free-running oscillator family.
for _high in (1, 2):
    for _low in (1, 2):
        _add_unary_forms(
            f"GOSC_H{_high}_L{_low}", GATED_OSCILLATOR,
            "GATED_OSCILLATOR", high_time=_high, low_time=_low)

COMPONENTS = tuple(_components)
BY_ID = {component.id: component for component in COMPONENTS}
BY_NAME = {component.name: component for component in COMPONENTS}
# Compact public dictionary for decoding the permanent number printed inside
# every FNV node.  Keep this derived from COMPONENTS so it cannot drift from the
# executable component catalogue.
NODE_TYPE_DICTIONARY = {
    component.id: component.name for component in COMPONENTS
}
MAX_COMPONENT_ID = len(COMPONENTS) - 1
_IDS_BY_FAMILY = {
    family: tuple(entry.id for entry in COMPONENTS if entry.family == family)
    for family in FAMILIES
}

if len(COMPONENTS) != 118:  # catalogue edits must be conscious and append-only
    raise RuntimeError("the initial FNV catalogue must contain exactly 118 types")


def component(component_id: int) -> ComponentType:
    try:
        return BY_ID[int(component_id)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"unknown FNV component type: {component_id!r}") from exc


def normalise_families(families: Iterable[str] | None) -> frozenset[str]:
    enabled = DEFAULT_FAMILIES if families is None else frozenset(
        str(family).upper() for family in families)
    unknown = enabled.difference(FAMILIES)
    if unknown:
        raise ValueError("unknown FNV component families: %s" %
                         ", ".join(sorted(unknown)))
    return frozenset(enabled)


@lru_cache(maxsize=None)
def _enabled_component_ids(enabled: frozenset[str],
                           include_empty: bool) -> tuple[int, ...]:
    ids = tuple(
        component_id
        for family in FAMILIES if family in enabled
        for component_id in _IDS_BY_FAMILY[family]
    )
    return ((0,) + ids) if include_empty else ids


def enabled_component_ids(families: Iterable[str] | None = None,
                          *, include_empty: bool = False) -> tuple[int, ...]:
    return _enabled_component_ids(
        normalise_families(families), bool(include_empty))


def _catalogue_payload() -> list[dict]:
    return [asdict(entry) for entry in COMPONENTS]


CATALOGUE_HASH = hashlib.sha256(json.dumps(
    _catalogue_payload(), sort_keys=True, separators=(",", ":")
).encode("utf-8")).hexdigest()


def verify_catalogue_hash(value: str) -> None:
    if str(value) != CATALOGUE_HASH:
        raise ValueError(
            "FNV catalogue hash mismatch: this checkpoint uses a different "
            "component-to-type-ID mapping")


@lru_cache(maxsize=None)
def state_distance(left: int, right: int) -> int:
    """Categorical/physical distance used by associative development.

    Numeric type IDs are labels, not bit fields.  Matching therefore compares
    component descriptions: exact identity is closest; related routes/timings
    in one family are nearer than unrelated component families.
    """
    a, b = component(left), component(right)
    if a.id == b.id:
        return 0
    if a.id == 0 or b.id == 0:
        return 8
    distance = 0 if a.family == b.family else 8
    if a.behavior != b.behavior:
        distance += 3
    distance += len(set(a.inputs).symmetric_difference(b.inputs))
    distance += len(set(a.outputs).symmetric_difference(b.outputs))
    if a.inputs != b.inputs:  # preserves asymmetric input roles
        distance += 1
    distance += abs(a.duration - b.duration)
    distance += abs(a.high_time - b.high_time)
    distance += abs(a.low_time - b.low_time)
    return max(1, distance)


# The catalogue is permanent and small (118 entries), while development asks
# this same distance question millions of times. Materialize the complete
# immutable table once so the growth hot loop performs four indexed reads
# instead of four cached Python function calls per gene comparison.
STATE_DISTANCES = tuple(
    tuple(state_distance(left, right) for right in range(len(COMPONENTS)))
    for left in range(len(COMPONENTS))
)
# The matrix supersedes those temporary cache entries on every hot path. Drop
# their key tuples so each persistent worker keeps one compact table, not both
# representations.
state_distance.cache_clear()


@lru_cache(maxsize=None)
def _local_component_ids(component_id: int,
                         enabled: frozenset[str]) -> tuple[int, ...]:
    """Closest enabled alternatives for the high-probability local mutation."""
    current = component(component_id)
    candidates = [
        candidate
        for family in FAMILIES if family in enabled
        for candidate in _IDS_BY_FAMILY[family]
        if candidate != current.id
        and (current.id == 0 or family == current.family)
    ]
    if not candidates:
        return enabled_component_ids(enabled)
    distances = STATE_DISTANCES[current.id]
    best = min(distances[candidate] for candidate in candidates)
    return tuple(candidate for candidate in candidates
                 if distances[candidate] == best)


def local_component_ids(component_id: int,
                        families: Iterable[str] | None = None) -> tuple[int, ...]:
    return _local_component_ids(
        int(component_id), normalise_families(families))


def quiescent_component(component_type: ComponentType) -> bool:
    """Catalogue-level power-on invariant.

    All-zero inputs and state must remain all-zero.  This is explicit rather
    than inferred from names so future catalogue additions have a testable
    admission gate.
    """
    return component_type.behavior in {
        "EMPTY", "AND", "OR", "XOR", "VETO", "DELAY", "NORMALIZER",
        "HOLD", "C_ELEMENT", "TOGGLE", "GATED_OSCILLATOR",
    }


assert all(quiescent_component(entry) for entry in COMPONENTS)
