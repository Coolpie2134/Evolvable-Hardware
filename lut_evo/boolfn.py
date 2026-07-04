"""
lut_evo/boolfn.py — make a 16-bit LUT understandable as boolean logic.

Each live LUT cell holds four 16-bit lookup tables (Ln, Ls, Le, Lw). A single
16-bit table is not an opaque number: it is a boolean function of the FOUR bits
the cell receives from its neighbours. The dynamics (lut.py `LutSim.step`) build
a 4-bit index from those neighbour bits — weight 1 = the bit from the NORTH
neighbour, 2 = SOUTH, 4 = EAST, 8 = WEST — and the LUT's output bit is
`(lut >> index) & 1`. So the table is exactly the truth table of

        out = f(N, S, E, W)

over the neighbour inputs N/S/E/W. This module turns the raw 16-bit word into
that function: a minimised sum-of-products expression (Quine–McCluskey), a short
human summary, and a raw truth string — so the genome and its lookup tables read
as logic instead of hex.
"""
from __future__ import annotations

# Neighbour-input variables in LUT-index bit order (see LutSim.step):
#   bit0 = from N, bit1 = from S, bit2 = from E, bit3 = from W.
INPUT_NAMES = ('N', 'S', 'E', 'W')
_NOT = '¬'      # ¬  (Latin-1 / cp1252 safe, so saved genome files are fine)
_AND = '·'      # ·
_OR  = ' + '


def minterms(lut):
    """The input indices (0..15) for which the LUT outputs 1."""
    lut &= 0xFFFF
    return [i for i in range(16) if (lut >> i) & 1]


def truth_string(lut):
    """The 16 output bits in index order 0..15 (index = N + 2S + 4E + 8W)."""
    lut &= 0xFFFF
    return ''.join(str((lut >> i) & 1) for i in range(16))


def popcount(lut):
    return bin(lut & 0xFFFF).count('1')


# ── Quine–McCluskey minimisation (only 4 variables — tiny and exact) ────────────

def _prime_implicants(ones):
    """Prime implicants of the ON-set `ones`, each as (value, dashmask) where a
    set bit in dashmask marks a "don't care" position."""
    groups = {(m, 0) for m in ones}
    primes = set()
    while groups:
        nxt, used = set(), set()
        terms = list(groups)
        for i in range(len(terms)):
            v1, d1 = terms[i]
            for j in range(i + 1, len(terms)):
                v2, d2 = terms[j]
                if d1 != d2:
                    continue
                diff = v1 ^ v2
                if diff and (diff & (diff - 1)) == 0:   # differ in exactly one bit
                    nd = d1 | diff
                    nxt.add((v1 & ~nd, nd))
                    used.add((v1, d1))
                    used.add((v2, d2))
        for t in groups:
            if t not in used:
                primes.add(t)
        groups = nxt
    return primes


def _implicant_minterms(pi):
    """All input indices matched by a prime implicant (value, dashmask)."""
    v, d = pi
    fixed = (~d) & 0xF
    return [m for m in range(16) if (m & fixed) == (v & fixed)]


def _cover(ones, primes):
    """Choose implicants covering every minterm: essentials first, then greedy."""
    cov = {pi: set(_implicant_minterms(pi)) for pi in primes}
    remaining, avail, chosen = set(ones), set(primes), []
    while remaining:
        essentials = set()
        for m in list(remaining):
            covering = [pi for pi in avail if m in cov[pi]]
            if len(covering) == 1:
                essentials.add(covering[0])
        if essentials:
            for pi in essentials:
                chosen.append(pi)
                remaining -= cov[pi]
                avail.discard(pi)
            continue
        best = max(avail, key=lambda pi: len(cov[pi] & remaining))
        chosen.append(best)
        remaining -= cov[best]
        avail.discard(best)
    return chosen


def _fmt_implicant(pi, names):
    v, d = pi
    lits = [(names[b] if (v >> b) & 1 else _NOT + names[b])
            for b in range(len(names)) if not (d >> b) & 1]
    return _AND.join(lits) if lits else '1'


def lut_sop(lut, names=INPUT_NAMES):
    """Minimised sum-of-products for the LUT as a function of the neighbour
    inputs, e.g. ``¬N·E + S``. Constants collapse to ``'0'`` / ``'1'``. Verified
    exact — falls back to the raw minterm SOP if minimisation ever disagrees."""
    lut &= 0xFFFF
    ones = minterms(lut)
    if not ones:
        return '0'
    if len(ones) == 16:
        return '1'
    chosen = _cover(ones, _prime_implicants(ones))
    covered = set()
    for pi in chosen:
        covered |= set(_implicant_minterms(pi))
    if covered == set(ones):                       # verified: exact cover
        terms = sorted((_fmt_implicant(pi, names) for pi in chosen),
                       key=lambda s: (len(s), s))
        return _OR.join(terms)
    # fallback (should never run): canonical minterm expansion, always correct
    terms = []
    for m in ones:
        lits = [(names[b] if (m >> b) & 1 else _NOT + names[b])
                for b in range(len(names))]
        terms.append(_AND.join(lits))
    return _OR.join(terms)


def lut_summary(lut):
    """One-line human description of a 16-bit LUT: the boolean expression plus a
    density hint. ``0`` reads as a dead direction; ``0xFFFF`` as constant-on."""
    lut &= 0xFFFF
    if lut == 0:
        return 'dead (always 0)'
    if lut == 0xFFFF:
        return 'always 1'
    return '%s   [%d/16 on]' % (lut_sop(lut), popcount(lut))


def lut_class_char(lut):
    """A one-glyph class for a LUT, for compact per-wedge labelling on the net:
    '' dead · '1' constant-on · a variable (buffer, e.g. 'N') or ¬variable
    (inverter) · '·' single product (AND of ≥2) · '+' sum (≥2 terms)."""
    lut &= 0xFFFF
    ones = minterms(lut)
    if not ones:
        return ''
    if len(ones) == 16:
        return '1'
    terms = lut_sop(lut).split(_OR)
    if len(terms) == 1:
        lits = terms[0].split(_AND)
        return lits[0] if len(lits) == 1 else _AND
    return '+'


def karnaugh(lut):
    """A 4x4 Karnaugh-style ON/OFF grid of the LUT as ASCII — rows vary EW,
    columns vary NS — for an at-a-glance view of the raw table. '#' = 1, '.' = 0."""
    lut &= 0xFFFF
    rows = []
    for ew in range(4):                # E,W bits -> row
        cells = []
        for ns in range(4):            # N,S bits -> column
            idx = ns | (ew << 2)
            cells.append('#' if (lut >> idx) & 1 else '.')
        rows.append(' '.join(cells))
    return rows


if __name__ == '__main__':      # exhaustive self-test: SOP must match every LUT
    def _eval(expr, n, s, e, w):
        if expr == '0':
            return 0
        if expr == '1':
            return 1
        env = {'N': n, 'S': s, 'E': e, 'W': w}
        for term in expr.split(_OR):
            ok = True
            for lit in term.split(_AND):
                neg = lit.startswith(_NOT)
                val = env[lit[-1]]
                if (val == 1) if neg else (val == 0):   # literal fails the term
                    ok = False
                    break
            if ok:
                return 1
        return 0
    bad = 0
    for lut in range(1 << 16):
        expr = lut_sop(lut)
        for idx in range(16):
            n, s, e, w = idx & 1, (idx >> 1) & 1, (idx >> 2) & 1, (idx >> 3) & 1
            if _eval(expr, n, s, e, w) != ((lut >> idx) & 1):
                bad += 1
                break
    print('self-test: %d / 65536 LUTs mismatched' % bad)
