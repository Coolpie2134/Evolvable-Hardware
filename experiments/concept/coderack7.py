#!/usr/bin/env python3
#------------------------------------------------------#
# Proof-of concept indirect-mapping genetic algorithm
# June 27, 2004 --- July 1, 2004
# Version "coderack7"
#   Generated from "coderack6".
#   -- Modularize the "main" routine.
#
# Python translation of coderack7.c
#------------------------------------------------------#
# NOTES:  The initial state should be "biased" in
#   orientation by declaring some input or output
#   nodes on specific sides.  Otherwise, one must
#   consider a solution to be valid in any of the
#   8 orientations.
#
#   Symmetry is broken arbitrarily by first-come,
#   first-served.  For evaluations having the same
#   score, the first one executed takes precedence.
#
#   It is interesting to note that this program
#   solved the "S"-shape problem handily, even
#   though it was rotated and flipped with respect
#   to the fitness calculation!
#
#   Need to do something to keep the size of the
#   genome small.
#------------------------------------------------------#

import os
import random
import sys
import select
import functools

# Genuinely-shared, platform-guarded terminal I/O (identical across all sims).
from experiments.concept.common.terminal import (setnodelay, setnormal,
                                      read_nonblocking, read_blocking_char, NullIO)

# GUI bridge: when a GUI engine sets this, main() streams per-generation stats
# through it and takes pause/stop from it instead of the terminal. None = CLI.
GUI = None
GUI_READY = True    # main() implements the GUI hook

#----------------*
# Some constants
#----------------*

POPSIZE = 4096         # total population size
STOPGROWTH = 15         # avoid unconstrained growth
STOPFIT = 0             # fitness bonus for having constrained growth
                        # (currently assigning 0 fitness when divergent)

MINCHROMO = 1           # initialization only
MAXCHROMO = 5           # initialization only

MINRULES = 1            # initialization only
MAXRULES = 3            # initialization only

MINEVALUATORS = 1       # initialization only
MAXEVALUATORS = 3       # initialization only
MINOPERATORS = 1        # initialization only
MAXOPERATORS = 2        # initialization only

INIT_MUTATE_RATE = 20      # initialization only
INIT_FITNESS_SCALE = 2     # initialization only

KERNEL = 5     # maximum spacing that can be
                # referenced by an initial rule
                # e.g., 5 means current position +/-2
                # mutations can override this. . .

RESULTSPACE = 10                   # number of "standard" symbols used
SPECIALSYMBOLS = 2                  # number of boundary marker types
BOUNDSYMBOL = RESULTSPACE           # boundary symbol marker
INPUTSYMBOL = RESULTSPACE + 1        # boundary symbol marker: input
SPECIES = 4

SIZEX = 8       # solution space size in x
SIZEY = 8       # solution space size in y
CELLSXY = SIZEX * SIZEY


def ABS(a):
    return -a if a < 0 else a


GEN_POP = 1
GEN_START = 2
GEN_CHEM = 4
GEN_SOLN = 8
GEN_ALL = GEN_POP | GEN_START | GEN_CHEM | GEN_SOLN

VIEW_BEST = 0    # display individual with best fitness
VIEW_MODAL = 1    # display individual with modal fitness

NUM_RULESET_RULES = 3
RULES_PRUNE = 0     # mutation ruleset promoting smaller genome
RULES_GROW = 1       # mutation ruleset promoting larger genome
RULES_MODIFY = 2      # mutation ruleset promoting gene changes

#----------------------------------------------#
# Definition of coderack evaluators/operators
#----------------------------------------------#

NUM_EVALUATORS = 4
EVAL_EXACT = 0x00
EVAL_MATCH = 0x01
EVAL_AND = 0x02
EVAL_OR = 0x03

NUM_OPERATORS = 2
OP_REPLACE = 0x80
OP_COPY = 0x81

# Name the more useful orientations

ORIENT_360 = 0x0f
ORIENT_ALL = 0xff
ORIENT_FLIP = 0xf0

USEFUL_ORIENTATIONS = [ORIENT_ALL, ORIENT_FLIP, ORIENT_360]

#--------------------------------------------------#
# An "opeval" (evaluator or operator expression) is represented
# as a small dict in place of the C `opeval` struct:
#   {"code": int, "orient": int, "score": int,
#    "relx": int, "rely": int, "symbol": int,
#    "relx2": int, "rely2": int}
#
# A "gene" is represented as a list [evaluators, operators], where
# each of evaluators/operators is a Python list of opeval dicts
# (taking the place of the linked lists in C).
#
# A "chromosome" (chromo struct in C) is represented as a dict:
#   {"tag": <single-char string>, "genes": [gene, ...], "split": int}
#--------------------------------------------------#


class Individual:
    __slots__ = ("genome", "bitstream", "fitness", "scaled", "stop", "species", "tag")

    def __init__(self):
        self.genome = []     # list of chromo dicts
        self.bitstream = [[0] * SIZEY for _ in range(SIZEX)]
        self.fitness = 0
        self.scaled = 0
        self.stop = 0
        self.species = 0
        self.tag = [0] * 7


#------------------*
# Global variables
#------------------*

cellinit = [[[0] * SIZEY for _ in range(SIZEX)] for _ in range(SPECIES)]
chemistry = [[0] * RESULTSPACE for _ in range(SPECIES)]

# 'S' pattern
solution1 = [
    [1, 1, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0, 1, 0],
    [1, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 0, 1],
    [0, 1, 0, 1, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 1, 1, 1],
]

# 'X' pattern
solution2 = [
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 0, 0, 0, 0, 1, 1],
    [1, 0, 1, 0, 0, 1, 0, 1],
    [0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0],
    [1, 0, 1, 0, 0, 1, 0, 1],
    [1, 1, 0, 0, 0, 0, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
]

# NOTE: the original C indexes solution1/solution2 as solution1[x][y]
# using the literal initializer above, which in C row-major layout
# means the first index selects the row. We mirror that exactly:
# solution1[x][y] in the C code reads row x, column y of the table
# as written above.


#----------------------------------------------------------------------#
# Terminal raw-mode helpers (replace setnodelay/setnormal from termios)
#----------------------------------------------------------------------#

# Terminal raw-mode helpers now come from experiments.concept.common.terminal (imported above).


#----------------------------------------------------------------------#
# Routine to set the random seed from the current time in microseconds
#----------------------------------------------------------------------#

def setrandom():
    random.seed()


#----------------------------------------------#
# Determine fitness from the "solution" vector
# Balance ones and zeros, and allow the
# "reverse" solution to have equal fitness.
#
# Each species may have its own solution.
#----------------------------------------------#

# These mirror the C 'static' locals inside find_solution(), which
# persist across calls and are computed once (t0/t1 are set up the
# first time find_solution() is called, based on whichever species
# happens to be passed in first).
_find_solution_t0 = 0
_find_solution_t1 = 0


def find_solution(species, bitstream):
    global _find_solution_t0, _find_solution_t1

    oddspec = species & 0x1

    if _find_solution_t0 == 0:
        t1 = 0
        for y in range(SIZEY):
            for x in range(SIZEX):
                sbit = solution1[x][y] if oddspec else solution2[x][y]
                if sbit:
                    t1 += 1

        t0 = CELLSXY - t1

        # Invert t0 and t1 and scale
        t0 = (CELLSXY * CELLSXY) // t0
        t1 = (CELLSXY * CELLSXY) // t1
        _find_solution_t0 = t0
        _find_solution_t1 = t1
    else:
        t0 = _find_solution_t0
        t1 = _find_solution_t1

    # Quick check: All 1's or all 0's imply (usually) that no genes
    # have matching conditions, so give this a very low fitness.
    b0 = 0
    for y in range(SIZEY):
        for x in range(SIZEX):
            if bitstream[x][y]:
                b0 += 1
    if b0 == CELLSXY or b0 == 0:
        return 0

    f0 = f1 = 0
    r0 = r1 = 0

    for y in range(SIZEY):
        for x in range(SIZEX):
            sbit = solution1[x][y] if oddspec else solution2[x][y]
            if sbit:
                if bitstream[x][y]:
                    f1 += 1
                else:
                    r1 += 1
            else:
                if not bitstream[x][y]:
                    f0 += 1
                else:
                    r0 += 1

    # fitness must be between 0 and CELLSXY
    ffit = (t1 * f1) + (t0 * f0)
    rfit = (t1 * r1) + (t0 * r0)
    return (ffit // (2 * CELLSXY)) if (ffit > rfit) else (rfit // (2 * CELLSXY))


def genecopy(src):
    """Return a deep copy of a gene [evaluators, operators]."""
    evaluators, operators = src
    new_evaluators = [dict(e) for e in evaluators]
    new_operators = [dict(o) for o in operators]
    return [new_evaluators, new_operators]


#------------------------------------------------------#
# Free operators and evaluators belonging to a gene
#------------------------------------------------------#

def freegene(gp):
    # No-op in Python (garbage collected); kept for structural fidelity.
    pass


#----------------------------------------------#
# Re-orient a relative x-coordinate
#----------------------------------------------#

def x_oriented(relx, rely, orient):
    resx = rely if orient >= 4 else relx
    if orient & 1:
        resx = -resx
    return resx


#----------------------------------------------#
# Re-orient a relative y-coordinate
#----------------------------------------------#

def y_oriented(relx, rely, orient):
    resy = relx if orient >= 4 else rely
    if orient & 1:
        resy = -resy
    return resy


#-------------------------------------------------#
# Growth algorithm and fitness evaluation routine
#-------------------------------------------------#

def evaluate(organism, fout):
    # This error already flagged
    if not organism.genome:
        organism.fitness = 0
        return

    cellarray = [[0] * SIZEY for _ in range(SIZEX)]
    outputbits = [[0] * SIZEY for _ in range(SIZEX)]

    # cellnext[x][y] holds {"gene": gene_or_None, "orient": int,
    #                        "escore": int, "symbol": int}
    cellnext = [[None] * SIZEY for _ in range(SIZEX)]

    # Initialize cell array

    for y in range(SIZEY):
        for x in range(SIZEX):
            cellarray[x][y] = cellinit[organism.species][x][y]

    if fout:
        for y in range(SIZEY):
            if y == 0:
                fout.write(" 0: ")
            else:
                fout.write("    ")
            for x in range(SIZEX):
                fout.write(chr(cellarray[x][y] + ord('a')))
            fout.write("\n")
        fout.write("\n")

    # Apply growth algorithm

    s = 0
    for s in range(STOPGROWTH):

        # Setup: Reset the score at each array position

        for y in range(SIZEY):
            for x in range(SIZEX):
                cellnext[x][y] = {
                    "gene": None,
                    "orient": 0,
                    "escore": -1000,
                    "symbol": cellarray[x][y],
                }

        # Pass 1: Evaluate the score at this position for each gene

        for y in range(SIZEY):
            for x in range(SIZEX):
                for cp in organism.genome:
                    for gp in cp["genes"]:
                        evaluators, operators = gp
                        for orient in range(8):
                            score = 0
                            for ev in evaluators:
                                if not (ev["orient"] & (1 << orient)):
                                    continue
                                absx = x + x_oriented(ev["relx"], ev["rely"], orient)
                                absy = y + y_oriented(ev["relx"], ev["rely"], orient)
                                if absx > SIZEX or absy > SIZEY or absx < -1 or absy < -1:
                                    continue
                                elif absx == 0 or absx == SIZEX:
                                    symbol1 = INPUTSYMBOL    # breaks symmetry
                                elif absy == SIZEY or absy == 0:
                                    symbol1 = BOUNDSYMBOL
                                else:
                                    symbol1 = cellarray[absx][absy]

                                symbol2 = None
                                if ev["code"] in (EVAL_MATCH, EVAL_AND, EVAL_OR):
                                    absx2 = x + x_oriented(ev["relx2"], ev["rely2"], orient)
                                    absy2 = y + y_oriented(ev["relx2"], ev["rely2"], orient)
                                    if absx2 > SIZEX or absy2 > SIZEY or absx2 < -1 or absy2 < -1:
                                        continue
                                    elif absx2 == 0 or absx2 == SIZEX:
                                        symbol2 = INPUTSYMBOL
                                    elif absy2 == SIZEY or absy2 == 0:
                                        symbol2 = BOUNDSYMBOL
                                    else:
                                        symbol2 = cellarray[absx2][absy2]

                                code = ev["code"]
                                if code == EVAL_EXACT:
                                    if symbol1 == ev["symbol"]:
                                        score += ev["score"]
                                elif code == EVAL_MATCH:
                                    if symbol1 == symbol2:
                                        score += ev["score"]
                                elif code == EVAL_AND:
                                    if symbol1 == ev["symbol"] and symbol2 == ev["symbol"]:
                                        score += ev["score"]
                                elif code == EVAL_OR:
                                    if symbol1 == ev["symbol"] or symbol2 == ev["symbol"]:
                                        score += ev["score"]

                            if score > cellnext[x][y]["escore"]:
                                cellnext[x][y]["escore"] = score
                                cellnext[x][y]["orient"] = orient
                                cellnext[x][y]["gene"] = gp   # save for quick reference

        # Pass 2: Operators

        for y in range(SIZEY):
            for x in range(SIZEX):
                gp = cellnext[x][y]["gene"]
                orient = cellnext[x][y]["orient"]
                if gp is None:
                    continue
                _evaluators, operators = gp
                for op in operators:
                    code = op["code"]
                    if code == OP_REPLACE:
                        cellnext[x][y]["symbol"] = op["symbol"]
                    elif code == OP_COPY:
                        absx = x + x_oriented(op["relx"], op["rely"], orient)
                        absy = y + y_oriented(op["relx"], op["rely"], orient)
                        if absx >= SIZEX or absy >= SIZEY:
                            continue
                        if absx < 0 or absy < 0:
                            continue
                        cellnext[x][y]["symbol"] = cellarray[absx][absy]

        if fout:
            for y in range(SIZEY):
                if y == 0:
                    fout.write("%2d: " % (s + 1))
                else:
                    fout.write("    ")
                for x in range(SIZEX):
                    fout.write(chr(cellnext[x][y]["symbol"] + ord('a')))
                fout.write("\n")
            fout.write("\n")

        # Check whether result was the same. If so, we can stop
        # Copy final result back into the cell

        match = 0
        for y in range(SIZEY):
            for x in range(SIZEX):
                nsymbol = cellnext[x][y]["symbol"]
                if cellarray[x][y] == nsymbol:
                    match += 1
                else:
                    cellarray[x][y] = nsymbol
                outputbits[x][y] = chemistry[organism.species][nsymbol]
        if match == CELLSXY:
            break
    else:
        s = STOPGROWTH

    # End of growth.  If we stopped early because growth was
    # stopped "naturally", add STOPFIT to the initial fitness value.
    # fitness = STOPFIT if s < STOPGROWTH else 0   (disabled in original)

    if s == STOPGROWTH:
        organism.fitness = 0
        return
    else:
        fitness = STOPFIT

    # Determine the resulting bitstream, and find the
    # fitness value for that result.

    for y in range(SIZEY):
        for x in range(SIZEX):
            organism.bitstream[x][y] = outputbits[x][y]

    fitness += find_solution(organism.species, organism.bitstream)
    organism.fitness = fitness
    organism.stop = s

    if fout:
        for y in range(SIZEY):
            if y == 0:
                fout.write("\n      bitstream: ")
            else:
                fout.write("                 ")
            for x in range(SIZEX):
                fout.write('.' if organism.bitstream[x][y] == 0 else 'X')
            if y == 0:
                fout.write("  fitness: %d  species: %d\n" % (fitness, organism.species))
            else:
                fout.write("\n")


#----------------------------------------------#
# Randomly generate allowed orientations
#----------------------------------------------#

def random_orient():
    return USEFUL_ORIENTATIONS[random.randrange(len(USEFUL_ORIENTATIONS))]


#----------------------------------------------#
# Randomly generate an evaluator/operator
#----------------------------------------------#

def random_opeval():
    oe = {
        "code": 0,
        "orient": random_orient(),
        "relx": random.randrange(KERNEL) - 2,
        "rely": random.randrange(KERNEL) - 2,
        "score": 0,
        "symbol": 0,
        "relx2": random.randrange(KERNEL) - 2,
        "rely2": random.randrange(KERNEL) - 2,
    }
    score = random.randrange(4) - 2
    if score <= 0:
        score -= 1
    oe["score"] = score
    return oe


#----------------------------------------------#
# Randomly generate an evaluator
#----------------------------------------------#

def random_evaluator():
    ev = random_opeval()
    ev["code"] = random.randrange(NUM_EVALUATORS) + EVAL_EXACT     # 0x00
    # evaluators include BOUNDSYMBOL and input/output symbols
    ev["symbol"] = random.randrange(RESULTSPACE + SPECIALSYMBOLS)
    return ev


#----------------------------------------------#
# Randomly generate an operator
#----------------------------------------------#

def random_operator():
    op = random_opeval()
    op["code"] = random.randrange(NUM_OPERATORS) + OP_REPLACE   # 0x80
    op["symbol"] = random.randrange(RESULTSPACE)   # no BOUNDSYMBOL
    return op


#----------------------------------------------#
# Randomly generate a gene (rule)
#----------------------------------------------#

def random_gene():
    numevalop = MAXEVALUATORS - MINEVALUATORS
    if numevalop <= 1:
        evaluators_n = MINEVALUATORS
    else:
        evaluators_n = random.randrange(numevalop) + MINEVALUATORS

    evaluators = []
    for _ in range(evaluators_n):
        ev = random_evaluator()
        evaluators.insert(0, ev)

    numevalop = MAXOPERATORS - MINOPERATORS
    if numevalop <= 1:
        operators_n = MINOPERATORS
    else:
        operators_n = random.randrange(numevalop) + MINOPERATORS

    operators = []
    for _ in range(operators_n):
        op = random_operator()
        operators.insert(0, op)

    return [evaluators, operators]


#----------------------------------------------#
# Randomly mutate an organism
#----------------------------------------------#

MUTATION_TYPES = 8

#---------------------------------------------------#
# Probabilities for each ruleset:     0     1    2
#---------------------------------------------------#
# Mutations are:
# 1) remove a rule              11   12    3
# 2) add a new rule              1   12    3
# 3) add an operator to a rule   1   12    3
# 4) remove an operator from a rule 12  12   4
# 5) add an evaluator to a rule  1   12    3
# 6) remove an evaluator from a rule 12  12   4
# 7) change an operator         30   14   40
# 8) change an evaluator        30   14   40
#---------------------------------------------------#
# Total cumulative probability:  100  100  100
#---------------------------------------------------#

RULE_WEIGHTS = [
    [11, 1, 1, 13, 1, 13, 30, 30],     # ruleset 0
    [12, 12, 12, 12, 12, 12, 14, 14],  # ruleset 1
    [3, 3, 3, 4, 3, 4, 40, 40],        # ruleset 2
]
CUMULATIVE_PROB = [
    [11, 12, 13, 26, 27, 40, 70, 100],
    [12, 24, 36, 48, 60, 72, 86, 100],
    [3, 6, 9, 13, 16, 20, 60, 100],
]


def random_mutate(organism, ruleset):
    numchromo = len(organism.genome)

    if numchromo == 0:
        sys.stderr.write("Bad error (?) No genome!\n")
        return

    c = random.randrange(numchromo)
    cp = organism.genome[c]

    numrules = len(cp["genes"])

    if numrules == 0:
        mtype = 2     # can only add a rule!
    else:
        mtype = 0
        while mtype == 0:
            w = random.randrange(100)    # random choice, in percentage points
            mtype = 0
            for t in range(MUTATION_TYPES):
                if w < CUMULATIVE_PROB[ruleset][t]:
                    mtype = t
                    break
            else:
                mtype = MUTATION_TYPES
            mtype += 1

            # Exception: Don't remove a rule if there's only one left
            if mtype == 1 and numrules == 1:
                mtype = 0

    if mtype == 1:    # Find a rule in the genome to remove
        rule = random.randrange(numrules)
        del cp["genes"][rule]
        if rule < cp["split"]:
            cp["split"] -= 1

    else:   # Move to a specific rule
        rule = random.randrange(numrules)
        gp = cp["genes"][rule]
        evaluators, operators = gp

        numop = len(operators)
        numeval = len(evaluators)

        op_idx = random.randrange(numop) if numop > 0 else None
        eval_idx = random.randrange(numeval) if numeval > 0 else None

        if mtype == 2:    # Add a new rule
            newg = random_gene()
            cp["genes"].insert(rule + 1, newg)

        elif mtype == 3:  # Add an operator to an existing rule
            if op_idx is None:
                return
            newop = random_operator()
            operators.insert(op_idx + 1, newop)

        elif mtype == 4:  # Remove an operator from an existing rule
            if op_idx is None:
                return
            if op_idx + 1 < len(operators):
                del operators[op_idx + 1]

        elif mtype == 5:  # Add an evaluator to an existing rule
            if eval_idx is None:
                return
            neweval = random_evaluator()
            evaluators.insert(eval_idx + 1, neweval)

        elif mtype == 6:  # Remove an evaluator from an existing rule
            if eval_idx is None:
                return
            if eval_idx + 1 < len(evaluators):
                del evaluators[eval_idx + 1]

        elif mtype >= 7:  # Change an operator/evaluator
            if mtype == 7:
                if op_idx is None:
                    return
                oe = operators[op_idx]
                is_op = True
            else:
                if eval_idx is None:
                    return
                oe = evaluators[eval_idx]
                is_op = False

            # Change: 1) code, 2) orient, 3) score, 4) relx, 5) rely
            # 6) symbol, 7) relx2, 8) rely2
            r = random.randrange(8)
            if r == 0:
                bit = 1
                old = oe["code"] & bit
                oe["code"] &= ~bit
                if old == 0:
                    oe["code"] |= bit
            elif r == 1:
                oe["orient"] = random_orient()
            elif r == 2:
                rr = random.randrange(2)
                oe["score"] += 1 if rr else -1
            elif r == 3:
                rr = random.randrange(2)
                oe["relx"] += 1 if rr else -1
            elif r == 4:
                rr = random.randrange(2)
                oe["rely"] += 1 if rr else -1
            elif r == 5:
                rr = random.randrange(2)
                oe["relx2"] += 1 if rr else -1
            elif r == 6:
                rr = random.randrange(2)
                oe["rely2"] += 1 if rr else -1
            elif r == 7:
                if is_op:
                    rr = random.randrange(RESULTSPACE)
                else:
                    rr = random.randrange(RESULTSPACE + SPECIALSYMBOLS)
                oe["symbol"] = rr


#----------------------------------------------#
# Randomly generate a new organism
#----------------------------------------------#

def random_organism(organism):
    organism.fitness = 0
    organism.stop = 0
    organism.genome = []
    for j in range(SIZEY):
        for i in range(SIZEX):
            organism.bitstream[i][j] = 0

    organism.tag = [random.randrange(100000) for _ in range(7)]

    c = MINCHROMO + random.randrange(MAXCHROMO - MINCHROMO + 1)
    for g in range(c):
        genes = []

        r = MINRULES + random.randrange(MAXRULES - MINRULES + 1)
        for _ in range(r):
            gp = random_gene()
            genes.insert(0, gp)

        split = (r // 2) + random.randrange(2)
        if split == r:
            split -= 1
        elif split == 0:
            split += 1

        cp = {"tag": chr(ord('a') + g), "genes": genes, "split": split}
        organism.genome.insert(0, cp)

    organism.species = random.randrange(SPECIES)


#----------------------------------------------#
# Print the chromosome table and growth
# pattern of an organism.
#----------------------------------------------#

def print_stuff(organism, fout):
    fout.write("   Chromosome table:\n")

    for cp in organism.genome:
        for gp in cp["genes"]:
            evaluators, operators = gp
            for ev in evaluators:
                code = ev["code"]
                if code == EVAL_EXACT:
                    fout.write(
                        'eval exact "%c" at (%d, %d)\n'
                        % (chr(ord('a') + ev["symbol"]), ev["relx"], ev["rely"])
                    )
                elif code == EVAL_MATCH:
                    fout.write(
                        "eval match (%d, %d) with (%d, %d)\n"
                        % (ev["relx"], ev["rely"], ev["relx2"], ev["rely2"])
                    )
                elif code == EVAL_AND:
                    fout.write(
                        'eval (%d, %d) AND (%d, %d) are "%c"\n'
                        % (ev["relx"], ev["rely"], ev["relx2"], ev["rely2"],
                           chr(ord('a') + ev["symbol"]))
                    )
                elif code == EVAL_OR:
                    fout.write(
                        'eval (%d, %d) OR (%d, %d) are "%c"\n'
                        % (ev["relx"], ev["rely"], ev["relx2"], ev["rely2"],
                           chr(ord('a') + ev["symbol"]))
                    )
            for op in operators:
                code = op["code"]
                if code == OP_REPLACE:
                    fout.write(
                        'op replace "%c" at (%d, %d)\n'
                        % (chr(ord('a') + op["symbol"]), op["relx"], op["rely"])
                    )
                elif code == OP_COPY:
                    fout.write(
                        "op copy from (%d, %d) to (%d, %d)\n"
                        % (op["relx"], op["rely"], op["relx2"], op["rely2"])
                    )
        fout.write("\n\n")

    fout.write("\n   Growth:\n")
    evaluate(organism, fout)


#----------------------------------------------#
# Free all allocated memory for an individual
#----------------------------------------------#

def freeindividual(organism):
    # No-op in Python (garbage collected); kept for structural fidelity.
    organism.genome = []


#----------------------------------------------#
# Free all allocated memory for a population
#----------------------------------------------#

def freepop(population):
    # No-op in Python (garbage collected); kept for structural fidelity.
    pass


#----------------------------------------------#
# qsort sorting routine for the tag list;
# the return values ensure that the list will
# end up ordered greatest to least.
#----------------------------------------------#

def _tagcomp(a, b):
    if a > b:
        return -1
    return 1


#----------------------------------------------#
# Check for the equality of pairs of numbers
#----------------------------------------------#

def quadcheck(a1, b1, a2, b2):
    if a1 == a2 and b1 == b2:
        return True
    elif a1 == b2 and b1 == a2:
        return True
    return False


#----------------------------------------------#
# Routine that checks if two individuals have
# the same ancestors to two generations back.
#----------------------------------------------#

def ancestors(org1, org2):
    if quadcheck(org1.tag[1], org1.tag[2], org2.tag[1], org2.tag[2]):
        return True
    elif quadcheck(org1.tag[3], org1.tag[4], org2.tag[3], org2.tag[4]):
        return True
    elif quadcheck(org1.tag[5], org1.tag[6], org2.tag[5], org2.tag[6]):
        return True
    elif quadcheck(org1.tag[3], org1.tag[4], org2.tag[5], org2.tag[6]):
        return True
    elif quadcheck(org1.tag[5], org1.tag[6], org2.tag[3], org2.tag[4]):
        return True
    return False


#----------------------------------------------#
#----------------------------------------------#

def do_one_generation(population, ruleset, fitness_scale):
    """Returns (pairings_x, pairings_y) lists, each of length POPSIZE."""

    # Evaluate fitness for each organism in the population.

    totalfit = 0
    for i in range(POPSIZE):
        organism = population[i]
        evaluate(organism, None)
        organism.scaled = organism.fitness
        for _ in range(fitness_scale - 1):
            organism.scaled *= organism.fitness
        totalfit += organism.scaled

    # Degenerate generation (everyone scored 0): fall back to uniform selection
    # so the proportional split below doesn't divide by zero.
    if totalfit == 0:
        for i in range(POPSIZE):
            population[i].scaled = 1
        totalfit = POPSIZE

    # Each individual gets to take part in a number of matings
    # according to (individual fitness / population fitness) times
    # (population size) times 2.  First-come, first-served.

    pairings_x = [None] * POPSIZE
    pairings_y = [None] * POPSIZE

    residual = 0
    binsleft = POPSIZE
    for i in range(POPSIZE):
        organism = population[i]
        fitshare = organism.scaled * POPSIZE * 2
        residual += (fitshare % totalfit)
        matings = fitshare // totalfit

        if residual > 0:
            matings += 1
            residual -= totalfit

        tries = 0
        s = 0
        while s < matings:
            x = random.randrange(binsleft)
            if pairings_x[x] is None:
                pairings_x[x] = organism
            else:
                # This block is necessary to prevent mating an organism
                # with itself or another individual with the same genome.

                if ancestors(pairings_x[x], organism):
                    tries += 1
                    if tries < 10:
                        s -= 1
                    else:
                        y = 0
                        found = False
                        while y < binsleft:
                            if pairings_x[y] is None:
                                pairings_x[y] = organism
                                tries = 0
                                found = True
                                break
                            elif not ancestors(pairings_x[y], organism):
                                x = y
                                tries = 0
                                found = True
                                break
                            y += 1
                        if not found:
                            found2 = False
                            for y2 in range(y, POPSIZE):
                                if not ancestors(pairings_x[x], pairings_x[y2]):
                                    pairings_y[x] = pairings_x[y2]
                                    found2 = True
                                    break
                            if not found2:
                                for y3 in range(POPSIZE):
                                    if pairings_x[y3] is not organism:
                                        while ancestors(pairings_x[y3], organism):
                                            random_mutate(pairings_x[y3], ruleset)
                                        break

                if not ancestors(pairings_x[x], organism):
                    save_x = pairings_x[binsleft - 1]
                    save_y = pairings_y[binsleft - 1]
                    pairings_x[binsleft - 1] = pairings_x[x]
                    pairings_y[binsleft - 1] = organism
                    pairings_x[x] = save_x
                    pairings_y[x] = save_y

                    binsleft -= 1
                    if binsleft == 0:
                        break
            s += 1
        if binsleft == 0:
            break

    return pairings_x, pairings_y


#----------------------------------------------#
#----------------------------------------------#

def make_next_generation(pairings_x, pairings_y, mutation_rate, ruleset):
    nextgen = [Individual() for _ in range(POPSIZE)]

    for i in range(POPSIZE):
        organism = nextgen[i]
        organism.fitness = 0
        organism.genome = []

        # Get parents from the "pairings" table
        parentx = pairings_x[i]
        parenty = pairings_y[i]

        # If one or both parents is undeclared, then
        # randomly generate a new organism.

        if parentx is None or parenty is None:
            random_organism(organism)
            continue

        # Set the tag to contain the index of the parents & grandparents
        organism.tag[1] = parentx.tag[0]
        organism.tag[2] = parenty.tag[0]

        organism.tag[3] = parentx.tag[1]
        organism.tag[4] = parentx.tag[2]

        organism.tag[5] = parenty.tag[1]
        organism.tag[6] = parenty.tag[2]

        # Crossover combination for multiple chromosomes.

        # Step 1 & 2: List all of the chromosome tags
        # (assume all chromosomes in an individual have unique tags)

        taglist = [cpx["tag"] for cpx in parentx.genome]
        for cpy in parenty.genome:
            if cpy["tag"] not in taglist:
                taglist.append(cpy["tag"])

        # Tags should be ordered from highest-numbered tag to lowest
        taglist.sort(key=functools.cmp_to_key(_tagcomp))

        # Step 3: For each chromosome tag, create a new chromosome
        # in the offspring.

        for tag in taglist:

            # Flip a coin to determine which parent donates the top
            # portion of the chromosome.  We do this simply by
            # swapping parentx and parenty.

            px, py = parentx, parenty
            if random.randrange(2) == 0:
                px, py = parenty, parentx

            cpx = None
            for cand in px.genome:
                if cand["tag"] == tag:
                    cpx = cand
                    break

            cpy = None
            for cand in py.genome:
                if cand["tag"] == tag:
                    cpy = cand
                    break

            new_genes = []
            new_tag = None
            new_split = 0

            if cpx is not None:
                for s in range(cpx["split"]):
                    new_genes.append(genecopy(cpx["genes"][s]))
                new_tag = cpx["tag"]
                new_split = cpx["split"]
            else:
                new_split = 0

            if cpy is not None:
                for s in range(cpy["split"], len(cpy["genes"])):
                    new_genes.append(genecopy(cpy["genes"][s]))
                new_tag = cpy["tag"]

            # It is possible for the chromosome to be empty
            if len(new_genes) > 0:
                cnew = {"tag": new_tag, "genes": new_genes, "split": new_split}
                organism.genome.insert(0, cnew)

        organism.species = parentx.species

    # Random mutation
    if mutation_rate:
        for i in range(POPSIZE):
            if random.randrange(100) < mutation_rate:
                random_mutate(nextgen[i], ruleset)

    return nextgen


#----------------------------------------------#
#----------------------------------------------#

def rebuild_stuff(rtype, population):
    if rtype & GEN_START:
        for sidx in range(SPECIES):
            cellival = random.randrange(RESULTSPACE)
            for y in range(SIZEY):
                for x in range(SIZEX):
                    cellinit[sidx][x][y] = cellival

    # Generate the "chemistry", or mapping from symbols to output bits

    if rtype & GEN_CHEM:
        for sidx in range(SPECIES):
            while True:
                btot = 0
                for x in range(RESULTSPACE):
                    chemistry[sidx][x] = random.randrange(2)
                    btot += chemistry[sidx][x]
                if 0 < btot < RESULTSPACE:
                    break

    # Generate a population of random chromosomes

    if rtype & GEN_POP:
        population = [Individual() for _ in range(POPSIZE)]
        for i in range(POPSIZE):
            random_organism(population[i])

    return population


#----------------------------------------------#
# Here's the application program. . .
#----------------------------------------------#

def main():
    global _find_solution_t0, _find_solution_t1

    fout = NullIO() if GUI is not None else sys.stdout
    setrandom()

    statfile = None if GUI is not None else open("stat.dat", "w")

    population = None

    rtype = GEN_ALL
    mutation_rate = INIT_MUTATE_RATE
    fitness_scale = INIT_FITNESS_SCALE
    view_mode = VIEW_BEST
    ruleset = RULES_PRUNE

    fd = sys.stdin.fileno()

    while True:    # corresponds to the "rebuild:" label in the C source
        population = rebuild_stuff(rtype, population)

        fout.write("When running, key h=print command help\n")
        fout.write("Hit any key to start:")
        fout.flush()
        if GUI is None:
            sys.stdin.read(1)
        fout.write("\n\n")
        setnodelay(fd)

        generation = 1
        rebuild_requested = False

        while True:
            pairings_x, pairings_y = do_one_generation(population, ruleset, fitness_scale)

            # Done with the population.  Report results

            fout.write("\nGeneration %d: " % generation)

            # Bin the population by fitness level and report
            fitness_bins = [0] * (CELLSXY + STOPFIT + 1)
            for i in range(POPSIZE):
                organism = population[i]
                fitness_bins[organism.fitness] += 1

            # Which is the largest nonzero bin?
            bintop = CELLSXY + STOPFIT
            while bintop >= 0:
                if fitness_bins[bintop] > 0:
                    break
                bintop -= 1

            binbot = STOPFIT + 1
            while binbot < CELLSXY + STOPFIT + 1:
                if fitness_bins[binbot] > 0:
                    break
                binbot += 1

            if view_mode == VIEW_MODAL:
                # Which bin represents the largest portion of the population?
                maxbin = 0
                for i in range(1, CELLSXY + STOPFIT + 1):
                    if fitness_bins[i] > fitness_bins[maxbin]:
                        maxbin = i
            else:
                maxbin = bintop

            if GUI is not None:
                fits = [o.fitness for o in population]
                typ  = next((o for o in population if o.fitness == maxbin), population[0])
                GUI.report({
                    "generation":   generation,
                    "fitnesses":    fits,
                    "best_fitness": max(fits),
                    "mean_fitness": sum(fits) / len(fits),
                    "max_fitness":  CELLSXY + STOPFIT,
                    "best_grid":    [[typ.bitstream[x][y] for x in range(SIZEX)]
                                     for y in range(SIZEY)],
                    "target_grid":  None,
                    "best_stop":    typ.stop,
                    "extra":        {},
                })
                GUI.checkpoint()

            # Print results for first organism in this bin
            for i in range(POPSIZE):
                organism = population[i]
                if organism.fitness == maxbin:
                    fout.write(
                        "%s organism fitness %d stop %d bits "
                        % ("Typical" if view_mode == VIEW_MODAL else "Best",
                           maxbin, organism.stop)
                    )
                    for y in range(SIZEY):
                        for x in range(SIZEX):
                            fout.write('.' if organism.bitstream[x][y] == 0 else 'X')
                    fout.write("\n")
                    print_stuff(organism, fout)
                    break

            # The histogram has been made rather complicated so that
            # the summary is more concise and easier to read.

            fout.write("\nGeneration %d Population statistics:\n" % generation)
            if bintop != CELLSXY + STOPFIT:
                fout.write("%2d    0\n /\n" % (CELLSXY + STOPFIT))
            for i in range(bintop, binbot - 1, -1):
                fout.write("%2d %4d" % (i, fitness_bins[i]))
                y = fitness_bins[i] // 100
                for _ in range(y):
                    fout.write("*")
                fout.write("\n")
            if maxbin < binbot:
                fout.write(" /\n%2d %4d" % (maxbin, fitness_bins[maxbin]))
                y = fitness_bins[maxbin] // 100
                for _ in range(y):
                    fout.write("*")
                fout.write("\n")
            if maxbin > 0 and fitness_bins[0] > 0:
                if maxbin > 1 and binbot > 1:
                    fout.write(" /\n")
                fout.write(" 0 %4d" % fitness_bins[0])
                y = fitness_bins[0] // 100
                for _ in range(y):
                    fout.write("*")
                fout.write("\n")
            fout.write("\n")

            if statfile:
                for i in range(CELLSXY + STOPFIT + 1):
                    statfile.write("%d " % fitness_bins[i])
                statfile.write("\n")

            fout.flush()

            # Check terminal input status

            while True:
                c = None if GUI is not None else read_nonblocking(fd)
                if c is None:
                    break
                c = c.lower()
                if c == 'w':
                    wf = open("pop.dat", "w")
                    for i in range(POPSIZE):
                        organism = population[i]
                        wf.write("Organism %d:\n" % (i + 1))
                        print_stuff(organism, wf)
                        wf.write(
                            "      tag=%d %d %d %d %d %d %d\n\n"
                            % tuple(organism.tag)
                        )
                        wf.write("\n")
                    wf.close()
                    fout.write("Wrote population file pop.dat.")
                elif c == 'p':
                    sys.stdout.write("Pausing.  Hit <return> to continue.\n")
                    setnormal(fd)
                    sys.stdin.read(1)
                    setnodelay(fd)
                elif c == 'm':
                    if mutation_rate == 1:
                        mutation_rate = 0
                        sys.stdout.write("mutation disabled\n")
                    else:
                        mutation_rate = 1
                        sys.stdout.write("mutation enabled at 1%\n")
                elif c == '-':
                    if mutation_rate:
                        mutation_rate -= 1
                        sys.stdout.write("mutation enabled at %d%%\n" % mutation_rate)
                elif c == '=':
                    mutation_rate += 1
                    sys.stdout.write("mutation enabled at %d%%\n" % mutation_rate)
                elif c == '.':
                    fitness_scale += 1
                    sys.stdout.write("fitness scaled by ^%d\n" % fitness_scale)
                elif c == ',':
                    if fitness_scale > 1:
                        fitness_scale -= 1
                        sys.stdout.write("fitness scaled by 2^%d\n" % fitness_scale)
                    else:
                        sys.stdout.write("Cannot reduce fitness below 1\n")
                elif c == 'v':
                    if view_mode == VIEW_MODAL:
                        view_mode = VIEW_BEST
                        sys.stdout.write(
                            "Display one individual with best fitness per generation.\n"
                        )
                    else:
                        view_mode = VIEW_MODAL
                        sys.stdout.write(
                            "Display one individual with modal fitness per generation.\n"
                        )
                elif c == 'o':
                    if statfile is None:
                        statfile = open("stat.dat", "w")
                        sys.stdout.write("Writing output to stat.dat\n")
                    else:
                        statfile.close()
                        statfile = None
                        sys.stdout.write("Closing stat.dat\n")
                elif c == 's':
                    ruleset = (ruleset + 1) % NUM_RULESET_RULES
                    if ruleset == RULES_PRUNE:
                        sys.stdout.write("Mutation rules promote smaller genome\n")
                    elif ruleset == RULES_GROW:
                        sys.stdout.write("Mutation rules promote larger genome\n")
                    elif ruleset == RULES_MODIFY:
                        sys.stdout.write("Mutation rules promote modified genes\n")
                elif c == 'r':
                    sys.stdout.write("Run again with new:\n")
                    sys.stdout.write("  1) population only\n")
                    sys.stdout.write("  2) population and startpoint\n")
                    sys.stdout.write("  3) population and chemistry\n")
                    sys.stdout.write("  4) population, startpoint, & chemistry\n")
                    sys.stdout.write("  5) everything\n")
                    sys.stdout.write("  *) select what to rerun\n")
                    sys.stdout.write("Choice? ")
                    sys.stdout.flush()
                    c2 = read_blocking_char(fd)
                    sys.stdout.write("%c\n" % c2)
                    if c2 == '1':
                        rtype = GEN_POP
                    elif c2 == '2':
                        rtype = GEN_POP | GEN_START
                    elif c2 == '3':
                        rtype = GEN_POP | GEN_CHEM
                    elif c2 == '4':
                        rtype = GEN_POP | GEN_START | GEN_CHEM
                    elif c2 == '5':
                        rtype = GEN_POP | GEN_START | GEN_CHEM | GEN_SOLN
                    else:
                        rtype = 0
                        sys.stdout.write("    New population? ")
                        sys.stdout.flush()
                        c2 = read_blocking_char(fd)
                        sys.stdout.write("%c\n" % c2)
                        if c2 in ('y', 'Y'):
                            rtype |= GEN_POP
                        sys.stdout.write("    New startpoint? ")
                        sys.stdout.flush()
                        c2 = read_blocking_char(fd)
                        sys.stdout.write("%c\n" % c2)
                        if c2 in ('y', 'Y'):
                            rtype |= GEN_START
                        sys.stdout.write("    New chemistry? ")
                        sys.stdout.flush()
                        c2 = read_blocking_char(fd)
                        sys.stdout.write("%c\n" % c2)
                        if c2 in ('y', 'Y'):
                            rtype |= GEN_CHEM
                    setnormal(fd)
                    rebuild_requested = True
                    break
                elif c in ('h', '?'):
                    fout.write("Command keystrokes:\n")
                    fout.write("r  run again\n")
                    fout.write("m  toggle mutation\n")
                    fout.write("v  toggle view: modal or best fitness\n")
                    fout.write("s  change ruleset weights\n")
                    fout.write("-  decrease mutation\n")
                    fout.write("=  increase mutation\n")
                    fout.write(",  decrease fitness scaling\n")
                    fout.write(".  increase fitness scaling\n")
                    fout.write("w  write population file pop.dat\n")
                    fout.write("o  write statistics file stat.dat\n")
                    fout.write("p  pause\n")
                    fout.write("h  print help\n")
                    fout.write("q  quit\n")
                elif c == 'q':
                    setnormal(fd)
                    fout.write("Done!\n")
                    if statfile:
                        statfile.close()
                    sys.exit(0)

            if rebuild_requested:
                break

            # Create the next generation by crossover recombination

            population = make_next_generation(pairings_x, pairings_y, mutation_rate, ruleset)

            generation += 1

        if rebuild_requested:
            continue
        break


if __name__ == "__main__":
    main()
