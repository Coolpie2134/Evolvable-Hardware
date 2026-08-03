"""Physical function banks for the four-input directional LUT substrate.

Every entry is still an ordinary 16-bit LUT truth table indexed as
``N | S<<1 | E<<2 | W<<3``.  Families change the inventory evolution may
install; they do not add parameters or alter LUT runtime physics.
"""
from __future__ import annotations

import random
from itertools import combinations, permutations

INPUTS = ("N", "S", "E", "W")
INPUT_BITS = {name: index for index, name in enumerate(INPUTS)}

ROUTING = "ROUTING"
AND = "AND"
OR = "OR"
XOR = "XOR"
VETO = "VETO"
THRESHOLD = "THRESHOLD"
MUX = "MUX"
UNRESTRICTED = "UNRESTRICTED"

FUNCTION_FAMILIES = (
    ROUTING, AND, OR, XOR, VETO, THRESHOLD, MUX, UNRESTRICTED,
)
DEFAULT_FUNCTION_FAMILIES = (UNRESTRICTED,)


def _table(function) -> int:
    value = 0
    for index in range(16):
        bits = tuple((index >> bit) & 1 for bit in range(4))
        if function(bits):
            value |= 1 << index
    return value


INPUT_TABLES = {
    name: _table(lambda bits, bit=bit: bits[bit])
    for name, bit in INPUT_BITS.items()
}


def _subset_tables(operator) -> tuple[int, ...]:
    values = []
    for width in (2, 3, 4):
        for names in combinations(INPUTS, width):
            indices = tuple(INPUT_BITS[name] for name in names)
            values.append(_table(
                lambda bits, indices=indices: operator(
                    tuple(bits[index] for index in indices))))
    return tuple(dict.fromkeys(values))


_AND_TABLES = _subset_tables(lambda values: all(values))
_OR_TABLES = _subset_tables(lambda values: any(values))
_XOR_TABLES = _subset_tables(lambda values: sum(values) & 1)

_VETO_TABLES = tuple(dict.fromkeys(
    _table(lambda bits, a=INPUT_BITS[a], b=INPUT_BITS[b]:
           bits[a] and not bits[b])
    for a, b in permutations(INPUTS, 2)
))

_THRESHOLD_TABLES = tuple(dict.fromkeys((
    *(
        _table(lambda bits, indices=tuple(INPUT_BITS[name] for name in names):
               sum(bits[index] for index in indices) >= 2)
        for names in combinations(INPUTS, 3)
    ),
    _table(lambda bits: sum(bits) >= 2),
    _table(lambda bits: sum(bits) >= 3),
)))

_MUX_TABLES = tuple(dict.fromkeys(
    _table(lambda bits, select=INPUT_BITS[select],
                  high=INPUT_BITS[high], low=INPUT_BITS[low]:
           bits[high] if bits[select] else bits[low])
    for select in INPUTS
    for high, low in permutations(
        tuple(name for name in INPUTS if name != select), 2)
))

FAMILY_TABLES = {
    ROUTING: tuple(INPUT_TABLES[name] for name in INPUTS),
    AND: _AND_TABLES,
    OR: _OR_TABLES,
    XOR: _XOR_TABLES,
    VETO: _VETO_TABLES,
    THRESHOLD: _THRESHOLD_TABLES,
    MUX: _MUX_TABLES,
}
_FAMILY_SETS = {
    family: frozenset(values) for family, values in FAMILY_TABLES.items()
}

if any(not values for values in FAMILY_TABLES.values()):
    raise RuntimeError("every named LUT function family must contain a table")
if any(value & 1 for values in FAMILY_TABLES.values() for value in values):
    raise RuntimeError("named LUT gate banks must remain quiescent at all-zero input")


def normalise_function_families(families=None) -> tuple[str, ...]:
    selected = (
        DEFAULT_FUNCTION_FAMILIES if families is None
        else tuple(str(family).upper() for family in families))
    unknown = set(selected).difference(FUNCTION_FAMILIES)
    if unknown:
        raise ValueError(
            "unknown LUT function families: %s" % ", ".join(sorted(unknown)))
    if not selected:
        raise ValueError("at least one LUT function family must be enabled")
    if len(set(selected)) != len(selected):
        raise ValueError("LUT function families may not be repeated")
    return tuple(
        family for family in FUNCTION_FAMILIES if family in selected)


def unrestricted_only(families=None) -> bool:
    return normalise_function_families(families) == (UNRESTRICTED,)


def table_families(value: int, families=None) -> tuple[str, ...]:
    """Enabled named families containing ``value``.

    ``UNRESTRICTED`` is returned only when no more-specific enabled bank owns
    the table, keeping local mutations inside a named family when possible.
    """
    value = int(value) & 0xFFFF
    enabled = normalise_function_families(families)
    named = tuple(
        family for family in enabled
        if family != UNRESTRICTED and value in _FAMILY_SETS[family])
    if named:
        return named
    return ((UNRESTRICTED,) if UNRESTRICTED in enabled else ())


def allowed_function_table(value: int, families=None) -> bool:
    value = int(value) & 0xFFFF
    if value == 0:  # the dead/off table is permanently available
        return True
    enabled = normalise_function_families(families)
    return (
        UNRESTRICTED in enabled
        or any(value in _FAMILY_SETS[family] for family in enabled)
    )


def _family_values(family: str):
    return range(1 << 16) if family == UNRESTRICTED else FAMILY_TABLES[family]


def random_function_table(families=None, *, allow_off: bool = True) -> int:
    """Family-first table sampling, with OFF reachable but deliberately rare."""
    enabled = normalise_function_families(families)
    if allow_off and random.random() < 0.02:
        return 0
    family = random.choice(enabled)
    values = _family_values(family)
    return values[random.randrange(len(values))]


def _nearest_table(value: int, values, forbidden=frozenset()) -> int:
    candidates = [
        candidate for candidate in values
        if candidate != value and candidate not in forbidden]
    if not candidates:
        raise ValueError("selected LUT function bank has no alternative table")
    distances = [(int(candidate ^ value).bit_count(), candidate)
                 for candidate in candidates]
    best = min(distance for distance, _candidate in distances)
    return random.choice([
        candidate for distance, candidate in distances if distance == best])


def project_function_table(value: int, families=None) -> int:
    """Project an existing executable table into a family-first inventory.

    A named family is chosen before its nearest table, preventing larger banks
    from dominating conversion of the dense ontogeny seed population.
    """
    value = int(value) & 0xFFFF
    enabled = normalise_function_families(families)
    if value == 0 or enabled == (UNRESTRICTED,):
        return value
    family = random.choice(enabled)
    if family == UNRESTRICTED:
        return value
    values = FAMILY_TABLES[family]
    distances = [(int(candidate ^ value).bit_count(), candidate)
                 for candidate in values]
    best = min(distance for distance, _candidate in distances)
    return random.choice([
        candidate for distance, candidate in distances if distance == best])


def mutate_function_table(value: int, families=None, *,
                          forbidden=()) -> int:
    """Local family-aware mutation of one executable truth table."""
    value = int(value) & 0xFFFF
    enabled = normalise_function_families(families)
    forbidden = frozenset(int(item) & 0xFFFF for item in forbidden)

    if value and 0 not in forbidden and random.random() < 0.03:
        return 0

    current = table_families(value, enabled)
    if current and random.random() < 0.78:
        family = random.choice(current)
    else:
        alternatives = tuple(
            family for family in enabled if family not in current)
        family = random.choice(alternatives or enabled)

    if family == UNRESTRICTED:
        # Preserve the ordinary LUT's useful Hamming-local landscape.
        for _ in range(64):
            width = random.randint(1, 3)
            candidate = value
            for bit in random.sample(range(16), width):
                candidate ^= 1 << bit
            if candidate != value and candidate not in forbidden:
                return candidate
        return _nearest_table(value, range(1 << 16), forbidden)
    return _nearest_table(value, FAMILY_TABLES[family], forbidden)


def enabled_named_tables(families=None) -> frozenset[int]:
    """Finite named-table union used by audits and GUI descriptions."""
    enabled = normalise_function_families(families)
    return frozenset({
        value for family in enabled if family != UNRESTRICTED
        for value in FAMILY_TABLES[family]
    })
