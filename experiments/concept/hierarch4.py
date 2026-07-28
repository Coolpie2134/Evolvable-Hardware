#!/usr/bin/env python3
#------------------------------------------------------#
# Proof-of concept indirect-mapping genetic algorithm
# June 27, 2004 --- February 15, 2005
# Version "hierarch4"
#   Generated from "hierarch4".
#
# Python translation of hierarch4.c
#------------------------------------------------------#
# NOTES:
#
# Attempting to unconstrain the size of the
# organism while maintaining the notion of the
# membrane-surrounded cell.  This version starts with
# a single unit and bisects down "arbitrarily" far,
# resulting in an organism with arbitrarily many
# computing cells.  The side of one cell may connect
# to many others.  This is resolved by applying the
# same function to each, but replacing the input with
# the input to the specific cell in question (assuming
# that the cell is functionally simulated, which for
# now it is not).
#
# For this version, we do no computation, but only
# experiment with the diversity created by the
# bisection method.
#------------------------------------------------------#
# Corrections from hierarch1:
# 1) bounds check for neighbors-in-range
# 2) correct check for flip before splitting
#
# Additions to hierarch1:
# 1) corner searches (NE, NW, SE, SW)
#
# Improvements to hierarch2:
# 1) speedup of boundary search (do once per cell)
# note:  This change appears to have corrected
# something that must have been wrong with hieararch1,
# since the symmetric patterns I was expecting now
# suddenly show up.
#
# Additions to hierarch3:
# 2) Added a choice of multiple solutions, to see the
# effect of suddenly changing the fitness landscape.
#------------------------------------------------------#

import os
import random
import sys
import select
import functools

# Genuinely-shared, platform-guarded terminal I/O (identical across all sims).
from concept.common.terminal import (setnodelay, setnormal,
                                      read_nonblocking, read_blocking_char, NullIO)

# GUI bridge: when a GUI engine sets this, main() streams per-generation stats
# through it and takes pause/stop from it instead of the terminal. None = CLI.
GUI = None
GUI_READY = True    # main() implements the GUI hook

#----------------*
# Some constants
#----------------*

POPSIZE = 8192         # total population size
STOPGROWTH = 15         # avoid unconstrained growth
STOPFIT = 1             # fitness bonus for having constrained growth

MINCHROMO = 2           # initialization only (was 2)
MAXCHROMO = 10          # initialization only (was 20)

MINRULES = 1            # initialization only
MAXRULES = 1            # initialization only

MINEVALUATORS = 1       # initialization only
MAXEVALUATORS = 1       # initialization only
MINOPERATORS = 1        # initialization only
MAXOPERATORS = 1        # initialization only

INIT_MUTATE_RATE = 20      # initialization only
INIT_FITNESS_SCALE = 1     # initialization only

RESULTSPACE = 8     # number of "standard" symbols used
SPECIES = 8

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

VIEW_BEST = 0      # display individual with best fitness
VIEW_MODAL = 1      # display individual with modal fitness

NUM_RULESET_RULES = 3
RULES_PRUNE = 0     # mutation ruleset promoting smaller genome
RULES_GROW = 1       # mutation ruleset promoting larger genome
RULES_MODIFY = 2      # mutation ruleset promoting gene changes

#----------------------------------------------#
# Definition of coderack evaluators/operators
#----------------------------------------------#

NUM_EVALUATORS = 3
EVAL_EXACT = 0
EVAL_HAS_N_NEIGHBORS = 1
EVAL_NEIGHBOR_MATCHES = 2

NUM_OPERATORS = 3
OP_REPLACE = 0
OP_SPLIT_NS = 1
OP_SPLIT_EW = 2

# Name the more useful orientations.  These are
# combined with CHAIN options; in a chain of
# evaluators, the first one declares the allowed
# orientations, and the remainder declare that
# their results should be OR'd or AND'ed with
# the rest of the results for that chain.

CHAIN_TYPES = 3
CHAIN_AND = 0
CHAIN_SUM = 1
CHAIN_MAX = 2

# Orientations vs. ordinal representation
# F_ indicates flipped

NORTH = 0
NORTHEAST = 1
EAST = 2
SOUTHEAST = 3
SOUTH = 4
SOUTHWEST = 5
WEST = 6
NORTHWEST = 7
F_NORTH = 8
F_NORTHWEST = 9
F_WEST = 10
F_SOUTHWEST = 11
F_SOUTH = 12
F_SOUTHEAST = 13
F_EAST = 14
F_NORTHEAST = 15

#--------------------------------------------------#
# An "eval" entry is a dict:
#   {"code": int, "chain": int, "orient": int,
#    "weight": int, "symbol": int, "number": int}
#
# An "op" entry is a dict: {"code": int, "symbol": int}
#
# A "gene" is represented as a list [evaluators, operators], where
# each of evaluators/operators is a Python list of the dicts above
# (taking the place of the linked lists in C).
#
# A "chromosome" (chromo struct in C) is represented as a dict:
#   {"tag": <single-char string>, "genes": [gene, ...], "split": int}
#
# A "cell" (corner-stitched cell) is represented as a dict:
#   {"symbol": int, "llx": int, "lly": int, "urx": int, "ury": int,
#    "todo": {"gene": gene_or_None, "orient": int, "escore": int}}
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

cellinit = [0] * SPECIES                       # seed value for target
chemistry = [[0] * RESULTSPACE for _ in range(SPECIES)]

# Global list of cells in an organism (replaces the C linked list)
celllist = []

cursoln = 0

solution = [
    [
        "XXX  X  ",
        " X  X X ",
        "X   X  X",
        "X   X  X",
        "X  X   X",
        "X  X   X",
        " X X  X ",
        "  X  XXX",
    ],
    [
        "X      X",
        "XX    XX",
        "X X  X X",
        "XXXXXXXX",
        "XXXXXXXX",
        "X X  X X",
        "XX    XX",
        "X      X",
    ],
    [
        "   XX   ",
        "   XX   ",
        "XXX  XXX",
        "XXX  XXX",
        "XXXXXXXX",
        "XXX  XXX",
        "XXX  XXX",
        "   XX   ",
    ],
    [
        "XXX  XXX",
        "XX    XX",
        "X X  X X",
        "   XX   ",
        "   XX   ",
        "X X  X X",
        "XX    XX",
        "XXX  XXX",
    ],
    [
        "XXXXXXXX",
        "X      X",
        "X X  X X",
        "X  XX  X",
        "X  XX  X",
        "X X  X X",
        "X      X",
        "XXXXXXXX",
    ],
    [
        "XXXXXXXX",
        "XX    XX",
        " XX    X",
        "  XXX   ",
        "   XXX  ",
        "X    XX ",
        "XX    XX",
        "XXXXXXXX",
    ],
    [
        "XXXXXXX ",
        " XX   XX",
        " XX   XX",
        " XX   XX",
        " XXXXXX ",
        " XX     ",
        " XX     ",
        "XXXX    ",
    ],
    [
        "  XX XX ",
        " XX   XX",
        " XX   XX",
        "  XXXXX ",
        "  XXXXX ",
        " XX   XX",
        " XX   XX",
        "  XX XX ",
    ],
]

# NOTE: hierarch4.c originally had only the first 6 of these listed
# explicitly inside its array-of-8 initializer (with the 7th/8th left
# implicit/short in the C literal in some variants); here we use the
# 8-pattern table exactly as found in the C source for fidelity.


#--------------------------------------------------------------#
# Split a cell in two across the N/S line.
# Return value: 0 on success, -1 on failure to split.
#--------------------------------------------------------------#

def split_NS(origcell):
    midx = origcell["urx"] + origcell["llx"]
    if midx & 0x1:
        return -1
    midx >>= 1     # X midpoint

    newcell = {
        "llx": origcell["llx"],
        "lly": origcell["lly"],
        "ury": origcell["ury"],
        "urx": midx,
        "symbol": 0,
        "todo": {"gene": None, "orient": 0, "escore": -1000},
    }
    origcell["llx"] = midx

    celllist.append(newcell)
    return 0


#--------------------------------------------------------------#
# Split a cell in two across the E/W line.
# Return value: 0 on success, -1 on failure to split.
#--------------------------------------------------------------#

def split_EW(origcell):
    midy = origcell["ury"] + origcell["lly"]
    if midy & 0x1:
        return -1
    midy >>= 1     # Y midpoint

    newcell = {
        "lly": origcell["lly"],
        "llx": origcell["llx"],
        "urx": origcell["urx"],
        "ury": midy,
        "symbol": 0,
        "todo": {"gene": None, "orient": 0, "escore": -1000},
    }
    origcell["lly"] = midy

    celllist.append(newcell)
    return 0


def reorient(a, r):
    return (a + r) & 0x7


#--------------------------------------------------------------#
# This function takes the absolute orientation, turns it into
# a relative orientation, and calls the indicated function
# using the appropriate function of the four above.
#--------------------------------------------------------------#

def enumerate_neighbors(cref):
    """Returns neighbors[orient] -> list of cell dicts, for orient 0..7."""
    neighbors = [[] for _ in range(8)]

    for sc in celllist:
        if sc is cref:
            continue

        if sc["lly"] == cref["ury"]:
            if sc["urx"] > cref["llx"] and sc["llx"] < cref["urx"]:
                neighbors[NORTH].append(sc)
            elif sc["llx"] == cref["urx"]:
                neighbors[NORTHEAST].append(sc)

        if sc["llx"] == cref["urx"]:
            if sc["ury"] > cref["lly"] and sc["lly"] < cref["ury"]:
                neighbors[EAST].append(sc)
            elif sc["ury"] == cref["lly"]:
                neighbors[SOUTHEAST].append(sc)

        if sc["ury"] == cref["lly"]:
            if sc["urx"] > cref["llx"] and sc["llx"] < cref["urx"]:
                neighbors[SOUTH].append(sc)
            elif sc["urx"] == cref["llx"]:
                neighbors[SOUTHWEST].append(sc)

        if sc["urx"] == cref["llx"]:
            if sc["ury"] > cref["lly"] and sc["lly"] < cref["ury"]:
                neighbors[WEST].append(sc)
            elif sc["lly"] == cref["ury"]:
                neighbors[NORTHWEST].append(sc)

    return neighbors


#----------------------------------------------------------------------#
# Terminal raw-mode helpers (replace setnodelay/setnormal from termios)
#----------------------------------------------------------------------#

# Terminal raw-mode helpers now come from concept.common.terminal (imported above).


#----------------------------------------------------------------------#
# Fast, cheap pseudorandom
#----------------------------------------------------------------------#

_prnv = [0] * 256
_pridx = 0


def init_pseudo():
    global _prnv
    _prnv = [random.randrange(255) for _ in range(256)]


def get_pseudo():
    global _pridx
    val = _prnv[_pridx]
    _pridx = (_pridx + 1) & 0xff
    return val


#----------------------------------------------------------------------#
# Routine to set the random seed from the current time in microseconds
#----------------------------------------------------------------------#

def setrandom():
    random.seed()
    init_pseudo()


#----------------------------------------------#
# Determine fitness from the "solution" vector
# Balance ones and zeros, and allow the
# "reverse" solution to have equal fitness.
#
# Each species may have its own solution.
#----------------------------------------------#

_find_solution_t0 = 0
_find_solution_t1 = 0


def find_solution(species, bitstream):
    global cursoln, _find_solution_t0, _find_solution_t1

    # New solution requires new determination of 1s and 0s
    if cursoln < 0:
        _find_solution_t0 = 0
        _find_solution_t1 = 0
        cursoln = (-cursoln) - 1     # 2s complement

    # Initial round---determine total 1s and 0s
    if _find_solution_t0 == 0:
        t1 = 0
        for y in range(SIZEY):
            for x in range(SIZEX):
                sbit = solution[cursoln][x][y]
                if sbit == 'X':
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
            sbit = solution[cursoln][x][y]
            if sbit == 'X':
                if bitstream[x][y]:
                    f1 += 1
                else:
                    r1 += 1
            else:
                if not bitstream[x][y]:
                    f0 += 1
                else:
                    r0 += 1

    ffit = (t1 * f1) + (t0 * f0)
    rfit = (t1 * r1) + (t0 * r0)
    return (ffit // (2 * CELLSXY)) if (ffit > rfit) else (rfit // (2 * CELLSXY))


def genecopy(src):
    """Return a deep copy of a gene [evaluators, operators]."""
    evaluators, operators = src
    new_evaluators = [dict(e) for e in evaluators]
    new_operators = [dict(o) for o in operators]
    return [new_evaluators, new_operators]


def freegene(gp):
    # No-op in Python (garbage collected); kept for structural fidelity.
    pass


def freecell(c):
    # No-op in Python (garbage collected); kept for structural fidelity.
    pass


#-------------------------------------------------#
# Growth algorithm and fitness evaluation routine
#-------------------------------------------------#

def evaluate(organism, fout):
    global celllist

    # This error already flagged
    if not organism.genome:
        organism.fitness = 0
        return

    outputbits = [[0] * SIZEY for _ in range(SIZEX)]

    # Initialize with a single cell (the zygote)

    c = {
        "symbol": cellinit[organism.species],
        "llx": 0,
        "lly": 0,
        "urx": SIZEX,
        "ury": SIZEY,
        "todo": {"gene": None, "orient": 0, "escore": -1000},
    }
    celllist = [c]

    # Apply growth algorithm

    s = 0
    nomatch = False
    for s in range(STOPGROWTH):

        # Setup: Reset the score at each array position

        for cell in celllist:
            cell["todo"] = {"gene": None, "orient": 0, "escore": -1000}

        # Pass 1: Evaluate the score at this position for each gene

        for cell in celllist:
            # For each cell, determine the boundary cells and save the info
            neighbors = enumerate_neighbors(cell)

            for cp in organism.genome:
                for gp in cp["genes"]:
                    evaluators, operators = gp
                    # Evaluate at N,S,E,W orientations only
                    for orient in range(0, 8, 2):
                        score = 0
                        for idx, ev in enumerate(evaluators):
                            eval_result = 0
                            rel_orient = reorient(orient, ev["orient"])

                            code = ev["code"]
                            if code == EVAL_EXACT:
                                if cell["symbol"] == ev["symbol"]:
                                    eval_result = ev["weight"]
                                else:
                                    eval_result = -ev["weight"]

                            elif code == EVAL_HAS_N_NEIGHBORS:
                                number = len(neighbors[rel_orient])
                                eval_result = ev["weight"] - ABS(number - ev["number"])

                            elif code == EVAL_NEIGHBOR_MATCHES:
                                number = 0
                                for nb in neighbors[rel_orient]:
                                    if nb["symbol"] == ev["symbol"]:
                                        number += 1
                                if number > 0:
                                    eval_result = ev["weight"]
                                else:
                                    eval_result = -ev["weight"]

                            # Add a small bit of randomness to the score so that
                            # "directionless" evaluators produce random results
                            # and thus get weeded out of the population.

                            if (get_pseudo() & 0xff) == 0xff:
                                eval_result += 1

                            # If we got here, we need to handle the eval score

                            if idx == 0:
                                score = eval_result
                            else:
                                chain = ev["chain"]
                                if chain == CHAIN_AND:
                                    if eval_result == 0:
                                        score -= ev["weight"]
                                    elif eval_result > score:
                                        score = eval_result
                                elif chain == CHAIN_MAX:
                                    if eval_result > score:
                                        score = eval_result
                                elif chain == CHAIN_SUM:
                                    if eval_result > 0:
                                        score += eval_result

                        if score > cell["todo"]["escore"]:
                            cell["todo"]["escore"] = score
                            cell["todo"]["orient"] = orient
                            cell["todo"]["gene"] = gp   # save for quick reference

        # Pass 2: Operators

        nomatch = False
        for cell in celllist:
            gp = cell["todo"]["gene"]
            if gp is None:
                continue
            orient = cell["todo"]["orient"]
            _evaluators, operators = gp
            for op in operators:
                code = op["code"]
                if code == OP_SPLIT_NS:
                    if orient & 2:
                        didsplit = split_EW(cell)
                    else:
                        didsplit = split_NS(cell)
                    if didsplit == 0:
                        cell["symbol"] = op["symbol"]
                        celllist[0]["symbol"] = op["symbol"]
                        nomatch = True

                elif code == OP_SPLIT_EW:
                    if orient & 2:
                        didsplit = split_NS(cell)
                    else:
                        didsplit = split_EW(cell)
                    if didsplit == 0:
                        cell["symbol"] = op["symbol"]
                        celllist[0]["symbol"] = op["symbol"]
                        nomatch = True

                elif code == OP_REPLACE:
                    if cell["symbol"] != op["symbol"]:
                        cell["symbol"] = op["symbol"]
                        nomatch = True

        # Check whether result was the same.  If so, we can stop
        # Copy final result back into the cell

        if not nomatch:
            break

        if fout:
            for cell in celllist:
                for x in range(cell["llx"], cell["urx"]):
                    for y in range(cell["lly"], cell["ury"]):
                        outputbits[x][y] = cell["symbol"]
            for y in range(SIZEY):
                if y == 0:
                    fout.write("\n      symbols: ")
                else:
                    fout.write("               ")
                for x in range(SIZEX):
                    fout.write(chr(ord('a') + outputbits[x][y]))
                fout.write("\n")
    else:
        s = STOPGROWTH

    # Generate output bits (by mapping cell space onto a grid)

    for cell in celllist:
        nsymbol = cell["symbol"]
        sval = chemistry[organism.species][nsymbol]
        for x in range(cell["llx"], cell["urx"]):
            for y in range(cell["lly"], cell["ury"]):
                outputbits[x][y] = sval

    # End of growth.  If we stopped early because growth was
    # stopped "naturally", add STOPFIT to the initial fitness value.

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

    freecell(celllist)


#----------------------------------------------#
# Randomly generate an evaluator
# We pass the organism so we can extract the
# initial symbol and all possible operator
# result symbols.  There is no point in
# producing an evaluator that does not contain
# one of these symbols.
#----------------------------------------------#

def random_evaluator():
    weight = random.randrange(8) - 4
    if weight >= 0:
        weight += 1
    return {
        "code": random.randrange(NUM_EVALUATORS),
        "number": random.randrange(SIZEX),
        "orient": random.randrange(16),
        "symbol": random.randrange(RESULTSPACE),
        "chain": random.randrange(CHAIN_TYPES),
        "weight": weight,
    }


#----------------------------------------------#
# Randomly generate an operator
#----------------------------------------------#

def random_operator():
    return {
        "code": random.randrange(NUM_OPERATORS),
        "symbol": random.randrange(RESULTSPACE),
    }


#----------------------------------------------#
# Randomly generate a gene (rule)
#----------------------------------------------#

def random_gene():
    numevalop = MAXOPERATORS - MINOPERATORS + 1
    if numevalop <= 1:
        operators_n = MINOPERATORS
    else:
        operators_n = random.randrange(numevalop) + MINOPERATORS

    operators = []
    for _ in range(operators_n):
        op = random_operator()
        operators.insert(0, op)

    numevalop = MAXEVALUATORS - MINEVALUATORS + 1
    if numevalop <= 1:
        evaluators_n = MINEVALUATORS
    else:
        evaluators_n = random.randrange(numevalop) + MINEVALUATORS

    evaluators = []
    for _ in range(evaluators_n):
        ev = random_evaluator()
        evaluators.insert(0, ev)

    return [evaluators, operators]


#----------------------------------------------#
# Randomly mutate an organism
#----------------------------------------------#

MUTATION_TYPES = 9

#---------------------------------------------------#
# Probabilities for each ruleset:     0     1    2
#---------------------------------------------------#
# Mutations are:
# 1) remove a rule                5   11    5
# 2) add a new rule              10   11    5
# 3) replace an existing evaluator 20  11    5
# 4) remove an evaluator from a rule 5  11   5
# 5) add an evaluator to a rule  10   11    5
# 6) replace an existing operator 20  11    5
# 7) remove an operator from a rule 5  11    5
# 8) add an operator to a rule   10   11    5
# 9) modify an evaluator         15   12   60
#---------------------------------------------------#
# Total cumulative probability:  100  100  100
#---------------------------------------------------#

CUMULATIVE_PROB = [
    [5, 15, 35, 40, 50, 70, 75, 85, 100],
    [11, 22, 33, 44, 55, 66, 77, 88, 100],
    [5, 10, 15, 20, 25, 30, 35, 40, 100],
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
            w = random.randrange(100)
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

    else:    # Move to a specific rule
        rule = random.randrange(numrules)
        gp = cp["genes"][rule]
        evaluators, operators = gp

        numop = len(operators)
        numeval = len(evaluators)

        # roe selects an index into operators / evaluators, matching the
        # C code's "for (op = gp->operator; roe > 0; op = op->next) roe--;"
        # (NULL/empty list if there are zero operators or evaluators)
        op_idx = random.randrange(numop) if numop > 0 else None
        eval_idx = random.randrange(numeval) if numeval > 0 else None

        if mtype == 2:    # Add a new rule
            newg = random_gene()
            cp["genes"].insert(rule + 1, newg)

        elif mtype in (3, 4):
            # Remove an evaluator from an existing rule
            # or replace an existing evaluator.  If there
            # is only one evaluator, always replace.
            #
            # C semantics: neweval = eval->next (the evaluator AFTER the
            # selected one). If mtype==3 and neweval exists, splice out
            # neweval (delete the one after the selected index) and stop.
            # Otherwise (mtype==4, or mtype==3 with no next element),
            # delete the selected evaluator itself and fall through to
            # "add_eval" (insert a new evaluator right after where the
            # deleted one was -- i.e. at the same index, or at the head
            # if none remain).

            if eval_idx is None:
                return

            next_idx = eval_idx + 1
            if mtype == 3 and next_idx < numeval:
                del evaluators[next_idx]
            else:
                del evaluators[eval_idx]
                neweval = random_evaluator()
                evaluators.insert(eval_idx, neweval)

        elif mtype == 5:    # Add an evaluator to an existing rule ("add_eval")
            neweval = random_evaluator()
            insert_at = (eval_idx + 1) if eval_idx is not None else 0
            evaluators.insert(insert_at, neweval)

        elif mtype in (6, 7):
            # Remove an operator from an existing rule
            # or replace an existing operator.
            # If there is only one operator, always replace it.
            #
            # NOTE: the original C checks "type == 5" here (not 6), which
            # appears to be a copy-paste bug inherited from the evaluator
            # branch above. Preserved faithfully: this means the
            # "remove next operator" branch never actually triggers from
            # this code path (mtype is 6 or 7 here, never 5), so the
            # else-branch (replace the selected operator) always runs.

            if op_idx is None:
                return

            next_idx = op_idx + 1
            condition_matches_c_bug = False    # "type == 5" can't hold here
            if condition_matches_c_bug and next_idx < numop:
                del operators[next_idx]
            else:
                del operators[op_idx]
                newop = random_operator()
                operators.insert(op_idx, newop)

        elif mtype == 8:    # Add an operator to an existing rule ("add_op")
            newop = random_operator()
            insert_at = (op_idx + 1) if op_idx is not None else 0
            operators.insert(insert_at, newop)

        elif mtype == 9:    # Modify an evaluator weight
            if eval_idx is None:
                return
            ev = evaluators[eval_idx]
            r = random.randrange(2)
            ev["weight"] += 1 if r else -1


#----------------------------------------------#
# Randomly generate a new organism
#----------------------------------------------#

def random_organism(organism):
    organism.fitness = 0
    organism.stop = 0
    organism.genome = []
    organism.species = random.randrange(SPECIES)

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

        # split = (r // 2) + random.randrange(2)   # disabled (temporary) in original C
        split = r // 2
        if split == r:
            split -= 1
        elif split == 0:
            split += 1

        cp = {"tag": chr(ord('a') + g), "genes": genes, "split": split}
        organism.genome.insert(0, cp)


#----------------------------------------------#
#----------------------------------------------#

def print_orient(orient):
    names = {
        NORTH: "N", NORTHEAST: "NE", EAST: "E", SOUTHEAST: "SE",
        SOUTH: "S", SOUTHWEST: "SW", WEST: "W", NORTHWEST: "NW",
        F_NORTH: "*N", F_NORTHEAST: "*NE", F_EAST: "*E", F_SOUTHEAST: "*SE",
        F_SOUTH: "*S", F_SOUTHWEST: "*SW", F_WEST: "*W", F_NORTHWEST: "*NW",
    }
    return names.get(orient, "")


#----------------------------------------------#
# Print the chromosome table and growth
# pattern of an organism.
#----------------------------------------------#

def print_stuff(organism, fout):
    fout.write("   Chromosome table:\n")

    for cp in organism.genome:
        for gp in cp["genes"]:
            evaluators, operators = gp
            for idx, ev in enumerate(evaluators):
                if idx == 0:
                    if ev["chain"]:
                        fout.write("\nAt all rotations plus flipped:\n")
                    else:
                        fout.write("\nAt all rotations:\n")
                else:
                    chain = ev["chain"]
                    if chain == CHAIN_AND:
                        fout.write("AND ")
                    elif chain == CHAIN_SUM:
                        fout.write("SUM ")
                    elif chain == CHAIN_MAX:
                        fout.write("MAX ")

                fout.write("(weight=%d) " % ev["weight"])
                code = ev["code"]
                if code == EVAL_EXACT:
                    fout.write('eval exact "%c"\n' % chr(ord('a') + ev["symbol"]))
                elif code == EVAL_HAS_N_NEIGHBORS:
                    fout.write(
                        "eval has %d neighbors to %s\n"
                        % (ev["number"], print_orient(ev["orient"]))
                    )
                elif code == EVAL_NEIGHBOR_MATCHES:
                    fout.write(
                        'eval neighbor to %s is "%c"\n'
                        % (print_orient(ev["orient"]), chr(ord('a') + ev["symbol"]))
                    )
            for op in operators:
                code = op["code"]
                if code == OP_REPLACE:
                    fout.write('op replace with "%c"\n' % chr(ord('a') + op["symbol"]))
                elif code == OP_SPLIT_NS:
                    fout.write('op split NS type "%c"\n' % chr(ord('a') + op["symbol"]))
                elif code == OP_SPLIT_EW:
                    fout.write('op split EW type "%c"\n' % chr(ord('a') + op["symbol"]))
        fout.write("\n\n")

    fout.write("\n   Growth:\n")
    evaluate(organism, fout)


#----------------------------------------------#
# Free all allocated memory for an individual / population
#----------------------------------------------#

def freeindividual(organism):
    organism.genome = []


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

    totalfit = 0
    for i in range(POPSIZE):
        organism = population[i]
        evaluate(organism, None)
        organism.scaled = organism.fitness
        for _ in range(fitness_scale - 1):
            organism.scaled *= organism.fitness
        totalfit += organism.scaled

    # Degenerate generation (everyone scored 0): uniform selection, avoid /0.
    if totalfit == 0:
        for i in range(POPSIZE):
            population[i].scaled = 1
        totalfit = POPSIZE

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

        parentx = pairings_x[i]
        parenty = pairings_y[i]

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

        taglist = [cpx["tag"] for cpx in parentx.genome]
        for cpy in parenty.genome:
            if cpy["tag"] not in taglist:
                taglist.append(cpy["tag"])

        taglist.sort(key=functools.cmp_to_key(_tagcomp))

        for tag in taglist:
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

            if len(new_genes) > 0:
                cnew = {"tag": new_tag, "genes": new_genes, "split": new_split}
                organism.genome.insert(0, cnew)

        organism.species = parentx.species

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
            cellinit[sidx] = random.randrange(RESULTSPACE)

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
    fout = NullIO() if GUI is not None else sys.stdout
    setrandom()

    statfile = None if GUI is not None else open("stat.dat", "w")

    population = None

    rtype = GEN_ALL
    mutation_rate = INIT_MUTATE_RATE
    fitness_scale = INIT_FITNESS_SCALE
    view_mode = VIEW_BEST
    ruleset = RULES_GROW

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

            fout.write("\nGeneration %d: " % generation)

            fitness_bins = [0] * (CELLSXY + STOPFIT + 1)
            for i in range(POPSIZE):
                organism = population[i]
                fitness_bins[organism.fitness] += 1

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
                elif c in ('1', '2', '3', '4', '5', '6', '7', '8'):
                    global cursoln
                    n = ord(c) - ord('1')
                    cursoln = -n - 1     # 2s complement
                    sys.stdout.write("Target pattern is %s\n" % c)
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
                    fout.write("r   run again\n")
                    fout.write("m   toggle mutation\n")
                    fout.write("v   toggle view: modal or best fitness\n")
                    fout.write("s   change ruleset weights\n")
                    fout.write("-   decrease mutation\n")
                    fout.write("=   increase mutation\n")
                    fout.write(",   decrease fitness scaling\n")
                    fout.write(".   increase fitness scaling\n")
                    fout.write("w   write population file pop.dat\n")
                    fout.write("o   write statistics file stat.dat\n")
                    fout.write("1-8 change target fitness pattern\n")
                    fout.write("p   pause\n")
                    fout.write("h   print help\n")
                    fout.write("q   quit\n")
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
