#!/usr/bin/env python3
#------------------------------------------------------#
# Proof-of concept indirect-mapping genetic algorithm
# August 11, 2007
# Version "multi9"
#   Generated from "multi8".
#   Use a poisson-distributed mutation rate, such
#   that multiple mutations per organism are
#   possible, and there is no upper limit on the
#   mutation rate.
#
# Python translation of multi9.c
#------------------------------------------------------#

import os
import random
import sys
import select
import termios
import tty
import functools

from concept.common.terminal import setnodelay, setnormal, read_nonblocking, read_blocking_char
from concept.common.ga import GEN_POP, GEN_START, GEN_CHEM, GEN_SOLN, GEN_ALL, do_one_generation
from concept.common.history import HIST_LIMIT, HISTORY, hist_compare, update_history
from concept.common.solution_tree import (make_solution, find_solution,
                                          free_solution, find_searchspace, new_tree)

#----------------*
# Some constants
#----------------*

POPSIZE = 8192        # total population size
STOPGROWTH = 30        # avoid unconstrained growth
INTFITSCALE = True     # is fitness scaling integer?
FITSCALE = 1.0         # fitness scaling (1 = unscaled)

MINCHROMO = 3          # for initialization only
MAXCHROMO = 5          # for initialization only

MINRULES = 4           # for initialization only
MAXRULES = 10          # for initialization only

KERNEL = 3                       # should be an odd number
HALFKERNEL = KERNEL >> 1          # center position in kernel
RESULTSPACE = 6                   # number of symbols used
BOUNDSYMBOL = 12                  # value of the end symbol

SIZEX = 5      # solution space size in x
SIZEY = 5      # solution space size in y
CELLSXY = SIZEX * SIZEY


def abs(a):
    return -a if a < 0 else a


def min(a, b):
    return a if a < b else b


GEN_POP = 1
GEN_START = 2
GEN_CHEM = 4
GEN_SOLN = 8
GEN_ALL = GEN_POP | GEN_START | GEN_CHEM | GEN_SOLN

HIST_LIMIT = 3                          # number of generations to save history
HISTORY = (1 << (HIST_LIMIT + 1)) - 1    # 15 entries

#
#       0       | 0     HIST_LIMIT = 3 implies HISTORY = 15 entries
#    1     2    | 1
#   3 4   5 6    | 2
#  78 9A BC DE   | 3
#

#--------------------------------------------------#
# Gene: a chromosome entry.
#   rule   - KERNEL x KERNEL list of context symbols, rule[x][y]
#   result - next-state symbol
#
# A "gene" is represented as a Python list [rule, result], where
# rule is a list of KERNEL lists (columns indexed by x), each of
# length KERNEL (rows indexed by y), matching the C "rule[KERNEL][KERNEL]"
# indexing convention rule[x][y].
#
# A "chromosome" (chromo struct in C) is represented as a dict:
#   {"tag": <single-char string>, "genes": [gene, ...], "split": int}
# where genes is a Python list taking the place of the linked list.
#--------------------------------------------------#


class Individual:
    __slots__ = ("genome", "bitstream", "fitness", "scaled", "stop", "history")

    def __init__(self):
        self.genome = []     # list of chromo dicts
        self.bitstream = [[0] * SIZEY for _ in range(SIZEX)]
        self.fitness = 0
        self.scaled = 0
        self.stop = 0
        self.history = [0] * HISTORY


#------------------*
# Global variables
#------------------*

cellinit = 0                            # seed value for target
chemistry = [0] * RESULTSPACE            # mapping of symbols to result bits
solution = [[0] * SIZEY for _ in range(SIZEX)]   # the actual solution

# "stree" (solution tree) is represented as nested dicts for memoization,
# replacing the C union-based binary tree:
#   node = {"zero": <node or None>, "one": <node or None>, "fitness": <int or None>}
# Internal nodes use zero/one; only depth==CELLSXY leaf nodes set fitness.
stree = None


#----------------------------------------------------------------------#
# Terminal raw-mode helpers (replace setnodelay/setnormal from termios)
#----------------------------------------------------------------------#

_saved_term_settings = None
def fitness_function(bitstream):
    sxnor = [[0] * SIZEY for _ in range(SIZEX)]
    btotal = 0

    for x in range(SIZEX):
        for y in range(SIZEY):
            sxnor[x][y] = bitstream[x][y] ^ solution[x][y]
            btotal += bitstream[x][y]

    # Discourage trivial solutions (all 1s or all 0s)
    if btotal == CELLSXY or btotal == 0:
        return 0

    # Encourage non-trivial solutions (balanced 1s and 0s)
    fittotal = CELLSXY - abs((btotal << 1) - CELLSXY)

    blockmax = min(SIZEX, SIZEY)

    for m in range(2, blockmax):
        for x in range(SIZEX + 1 - m):
            for y in range(SIZEY + 1 - m):
                all_zero = True
                for i in range(m):
                    row_ok = True
                    for j in range(m):
                        if sxnor[x + i][y + j] > 0:
                            row_ok = False
                            break
                    if not row_ok:
                        all_zero = False
                        break
                if all_zero:
                    fittotal += 1
    return fittotal


#----------------------------------------------#
# Create a random matrix to solve.
#----------------------------------------------#
def evaluate(organism, fout):
    cellnext = [[0] * SIZEY for _ in range(SIZEX)]
    cellarray = [[0] * SIZEY for _ in range(SIZEX)]
    outputbits = [[0] * SIZEY for _ in range(SIZEX)]
    key = [[0] * KERNEL for _ in range(KERNEL)]

    # Initialize cell array
    for y in range(SIZEY):
        for x in range(SIZEX):
            cellarray[x][y] = cellinit

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
    gmatch = None
    for s in range(STOPGROWTH):
        for y in range(SIZEY):
            for x in range(SIZEX):
                for j in range(KERNEL):
                    for i in range(KERNEL):
                        m = x + i - HALFKERNEL
                        n = y + j - HALFKERNEL
                        if m < 0 or m >= SIZEX or n < 0 or n >= SIZEY:
                            key[i][j] = BOUNDSYMBOL
                        else:
                            key[i][j] = cellarray[m][n]

                # Find symbolic distance between key and all table entries

                mindist = 1000
                for cp in organism.genome:
                    for gp in cp["genes"]:
                        if mindist == 0:
                            break
                        rule, result = gp
                        distance = 0
                        abort = False
                        for ll in range(KERNEL):
                            for kk in range(KERNEL):
                                sdist = key[kk][ll] - rule[kk][ll]
                                distance += abs(sdist)
                                if distance > mindist:
                                    abort = True
                                    break
                            if abort:
                                break
                        if distance < mindist:
                            mindist = distance
                            gmatch = gp

                # Apply result from best matching entry
                cellnext[x][y] = gmatch[1]
                outputbits[x][y] = chemistry[gmatch[1]]

        if fout:
            for y in range(SIZEY):
                if y == 0:
                    fout.write("%2d: " % (s + 1))
                else:
                    fout.write("    ")
                for x in range(SIZEX):
                    fout.write(chr(cellnext[x][y] + ord('a')))
                fout.write("\n")
            fout.write("\n")

        # Check whether result was the same.  If so, we can stop
        # Copy final result back into the cell

        match = 0
        for y in range(SIZEY):
            for x in range(SIZEX):
                if cellarray[x][y] == cellnext[x][y]:
                    match += 1
                else:
                    cellarray[x][y] = cellnext[x][y]
        if match == CELLSXY:
            break
    else:
        s = STOPGROWTH

    # Determine the resulting bitstream, and find the
    # fitness value for that result.  Solutions that
    # don't terminate growth are given zero fitness.

    for y in range(SIZEY):
        for x in range(SIZEX):
            organism.bitstream[x][y] = outputbits[x][y]

    if s == STOPGROWTH:
        organism.fitness = 0
    else:
        organism.fitness = find_solution(stree, organism.bitstream, 0)
    organism.stop = s

    if fout:
        for y in range(SIZEY):
            if y == 0:
                fout.write("\n      bitstream: ")
            else:
                fout.write("                 ")
            for x in range(SIZEX):
                fout.write('0' if organism.bitstream[x][y] == 0 else '1')
            if y == 0:
                fout.write("  fitness: %d\n" % organism.fitness)
            else:
                fout.write("\n")


#----------------------------------------------#
# Special function to place boundary symbols
# that satisfies their constraints.
#----------------------------------------------#

def makeboundsymbol(rule, side, extent):
    # side is the side (N/S/E/W), extent is the width of the boundary
    r = extent
    if side == 0:      # N
        for i in range(r):
            for jj in range(KERNEL):
                rule[jj][i] = BOUNDSYMBOL
    elif side == 1:    # S
        for i in range(KERNEL - 1, KERNEL - 1 - r, -1):
            for jj in range(KERNEL):
                rule[jj][i] = BOUNDSYMBOL
    elif side == 2:    # E
        for i in range(r):
            for jj in range(KERNEL):
                rule[i][jj] = BOUNDSYMBOL
    elif side == 3:    # W
        for i in range(KERNEL - 1, KERNEL - 1 - r, -1):
            for jj in range(KERNEL):
                rule[i][jj] = BOUNDSYMBOL


#----------------------------------------------#
# Randomly mutate an organism
#----------------------------------------------#

def random_mutate(organism):
    #---------------------------------------------------#
    # Mutations are:
    #  1) add a rule
    #  2) subtract a rule
    #  3) modify a rule
    #  4) modify a result
    #---------------------------------------------------#
    # To do:  move a rule, possibly between chromosomes
    #---------------------------------------------------#

    numchromo = len(organism.genome)
    c = random.randrange(numchromo)
    cp = organism.genome[c]

    numrules = len(cp["genes"])

    if numrules == 0:
        mtype = 2             # can only add a rule!
    elif numrules == 1:
        mtype = 2 + random.randrange(3)   # don't remove the last rule!
    else:
        mtype = 1 + random.randrange(4)

    if mtype == 1:    # Find a rule in the genome to remove
        rule = random.randrange(numrules)
        del cp["genes"][rule]
        if rule < cp["split"]:
            cp["split"] -= 1

    elif mtype == 2:  # Create new random rule
        newrule = [[random.randrange(RESULTSPACE) for _ in range(KERNEL)]
                   for _ in range(KERNEL)]

        # Treat BOUNDSYMBOL specially.  BOUNDSYMBOL should take up an
        # entire row or column when it appears, and other constraints
        # appropriate to a boundary condition.
        #
        # Correct:                       Incorrect:
        #   xxx   xaa   xxx                axa   xxx   axx   aaa     xaa
        #   aaa   xaa   xaa                axa   aaa   axx   xaa     aaa
        #   aaa   xaa   xaa                axa   xxx   axx   aaa     aaa

        if random.randrange(RESULTSPACE + 1) == 0:
            r = 1 + random.randrange(HALFKERNEL)    # extent
            j = random.randrange(4)                  # side
            makeboundsymbol(newrule, j, r)
            if random.randrange(2) == 0:
                r = 1 + random.randrange(HALFKERNEL)
                # NOTE: the original C reads an uninitialized variable
                # `k` here when j >= 2 (a latent bug in multi9.c). We
                # preserve the intended behavior for j < 2 (mirror to
                # the opposite side) and, for j >= 2, fall back to 0
                # (N) since that mirrors what an uninitialized
                # zero-valued stack slot would typically produce.
                if j < 2:
                    k = j + 2
                else:
                    k = 0
                makeboundsymbol(newrule, k, r)

        newresult = random.randrange(RESULTSPACE)
        newg = [newrule, newresult]

        rule = random.randrange(numrules + 1)
        cp["genes"].insert(rule, newg)

        if rule < cp["split"]:
            cp["split"] += 1

    elif mtype == 3:  # Choose a rule and modify one context unit
        rule = random.randrange(numrules)
        gp = cp["genes"][rule]
        rule_grid = gp[0]

        # Choose one of the KERNEL x KERNEL context positions
        r = random.randrange(KERNEL)
        s = random.randrange(KERNEL)

        # We really need to deal properly with BOUNDSYMBOL here!

        if rule_grid[r][s] == BOUNDSYMBOL:
            rule_grid[r][s] = random.randrange(RESULTSPACE)
        else:
            # Neighbor rules may be set to BOUNDSYMBOL
            if r != HALFKERNEL and random.randrange(RESULTSPACE) == 0:
                rule_grid[r][s] = BOUNDSYMBOL
            elif s != HALFKERNEL and random.randrange(RESULTSPACE) == 0:
                rule_grid[r][s] = BOUNDSYMBOL
            else:
                # Modify the rule by one position randomly + or -
                if rule_grid[r][s] == 0:
                    rule_grid[r][s] += 1
                elif rule_grid[r][s] == (RESULTSPACE - 1):
                    rule_grid[r][s] -= 1
                else:
                    direction = (2 * random.randrange(2)) - 1
                    rule_grid[r][s] += direction

    elif mtype == 4:  # Choose a rule and change the result
        rule = random.randrange(numrules)
        gp = cp["genes"][rule]

        if gp[1] == 0:
            gp[1] += 1
        elif gp[1] == (RESULTSPACE - 1):
            gp[1] -= 1
        else:
            direction = (2 * random.randrange(2)) - 1
            gp[1] += direction


#----------------------------------------------#
# Randomly generate a new organism
#----------------------------------------------#

def random_organism(organism, idx):
    organism.history[0] = idx
    for j in range(1, HISTORY):
        organism.history[j] = 0

    organism.fitness = 0
    organism.stop = 0
    organism.genome = []
    for j in range(SIZEY):
        for i in range(SIZEX):
            organism.bitstream[i][j] = 0

    c = MINCHROMO + random.randrange(MAXCHROMO - MINCHROMO + 1)
    for g in range(c):
        genes = []

        r = MINRULES + random.randrange(MAXRULES - MINRULES + 1)
        for _ in range(r):
            rule = [[random.randrange(RESULTSPACE) for _ in range(KERNEL)]
                    for _ in range(KERNEL)]

            # See random_mutate() for explanation of BOUNDSYMBOL positioning
            if random.randrange(RESULTSPACE + 1) == 0:
                n = 1 + random.randrange(HALFKERNEL)
                j = random.randrange(4)
                makeboundsymbol(rule, j, n)
                if random.randrange(2) == 0:
                    n = 1 + random.randrange(HALFKERNEL)
                    # See note in random_mutate() about this uninitialized
                    # `k` in the original C source.
                    if j < 2:
                        k = j + 2
                    else:
                        k = 0
                    makeboundsymbol(rule, k, n)

            result = random.randrange(RESULTSPACE)
            gp = [rule, result]
            # C code prepends to the linked list; preserve that ordering
            genes.insert(0, gp)

        split = (r // 2) + random.randrange(2)
        if split == r:
            split -= 1
        elif split == 0:
            split += 1

        cp = {"tag": chr(ord('a') + g), "genes": genes, "split": split}
        # C code prepends new chromosomes to the genome list
        organism.genome.insert(0, cp)


#----------------------------------------------#
# Print the chromosome table and growth
# pattern of an organism.
#----------------------------------------------#

def print_stuff(organism, fout):
    fout.write("   Chromosome table:\n")

    for cp in organism.genome:
        for k in range(KERNEL):
            i = 0
            if cp["split"] == 0:
                fout.write(" | ")
            for gp in cp["genes"]:
                rule, result = gp
                for j in range(KERNEL):
                    fout.write("x" if rule[j][k] == BOUNDSYMBOL else chr(rule[j][k] + ord('a')))
                i += 1
                if i == cp["split"]:
                    fout.write(" | ")
                else:
                    fout.write("   ")
            fout.write("\n")

        if cp["split"] == 0:
            fout.write("   ")
        for gp in cp["genes"]:
            result = gp[1]
            fout.write(" %c    " % chr(result + ord('a')))
            for _ in range(KERNEL - 3):
                fout.write(" ")
        fout.write("\n")

        if cp["split"] == 0:
            fout.write("   ")
        for gp in cp["genes"]:
            result = gp[1]
            fout.write("(%d)   " % chemistry[result])
            for _ in range(KERNEL - 3):
                fout.write(" ")
        fout.write("\n\n")

    fout.write("\n   Growth:\n")
    evaluate(organism, fout)


#----------------------------------------------#
# qsort sorting routine for the tag list;
# the return values ensure that the list will
# end up ordered greatest to least.
#----------------------------------------------#
def main():
    global cellinit, chemistry, stree

    fout = sys.stdout
    setrandom()

    statfile = open("stat.dat", "w")

    population = None
    fitness_bins = None
    maxfit = 0

    rtype = GEN_ALL
    mutation_rate = 200
    do_mutations = True

    fd = sys.stdin.fileno()

    while True:    # corresponds to the "rebuild:" label in the C source
        # Generate an initial solution to be used by all individuals

        if rtype & GEN_START:
            cellinit = random.randrange(RESULTSPACE)

        if rtype & GEN_SOLN:
            free_solution(stree, 0)
            stree = {"zero": None, "one": None, "fitness": None}
            make_solution(fout)
            maxfit = find_solution(stree, solution, 0)

            # Print the target solution

            fout.write("Target solution, fitness %d:\n\n" % maxfit)
            for y in range(SIZEY):
                for x in range(SIZEX):
                    fout.write(chr(solution[x][y] + ord('0')))
                fout.write("\n")
            fout.write("\n\n")

        # Generate the "chemistry", or mapping from symbols to output bits

        if rtype & GEN_CHEM:
            while True:
                btot = 0
                for x in range(RESULTSPACE):
                    chemistry[x] = random.randrange(2)
                    btot += chemistry[x]
                if 0 < btot < RESULTSPACE:
                    break

        # Generate a population of random chromosomes

        if rtype & GEN_POP:
            population = [Individual() for _ in range(POPSIZE)]
            for i in range(POPSIZE):
                random_organism(population[i], i + 1)

        fout.write("When running, key h=print command help\n")
        fout.write("Hit any key to start:")
        fout.flush()
        sys.stdin.read(1)
        fout.write("\n\n")
        setnodelay(fd)

        generation = 1
        rebuild_requested = False

        while True:
            # Evaluate fitness for each organism in the population.

            totalfit = 0
            for i in range(POPSIZE):
                organism = population[i]
                evaluate(organism, None)
                if INTFITSCALE:
                    organism.scaled = organism.fitness
                    for _ in range(int(FITSCALE) - 1):
                        organism.scaled *= organism.fitness
                else:
                    organism.scaled = int(organism.fitness ** FITSCALE)
                totalfit += organism.scaled

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
                        # Ensure diversity in the population by preventing
                        # sibling and cousin pairings back HIST_LIMIT
                        # generations.

                        if hist_compare(pairings_x[x], organism):
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
                                    elif not hist_compare(pairings_x[y], organism):
                                        x = y
                                        tries = 0
                                        found = True
                                        break
                                    y += 1
                                if not found:
                                    found2 = False
                                    for y2 in range(y, POPSIZE):
                                        if not hist_compare(pairings_x[x], pairings_x[y2]):
                                            pairings_y[x] = pairings_x[y2]
                                            found2 = True
                                            break
                                    if not found2:
                                        for y3 in range(POPSIZE):
                                            if pairings_x[y3] is not organism:
                                                while hist_compare(pairings_x[y3], organism):
                                                    random_mutate(pairings_x[y3])
                                                break

                        if not hist_compare(pairings_x[x], organism):
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

            # Done with the population.  Report results

            fout.write("\nGeneration %d: " % generation)

            fitness_bins = [0] * maxfit

            # Bin the population by fitness level and report
            for i in range(POPSIZE):
                organism = population[i]
                fitness_bins[organism.fitness] += 1

            # Which bin (other than fitness zero) represents the
            # largest portion of the population?

            maxbin = 1
            for i in range(2, maxfit):
                if fitness_bins[i] > fitness_bins[maxbin]:
                    maxbin = i

            # Print results for first organism in this bin; compare with solution.

            for i in range(POPSIZE):
                organism = population[i]
                if organism.fitness == maxbin:
                    fout.write(
                        "Typical organism fitness %d, stop %d, bits=\n"
                        % (maxbin, organism.stop)
                    )
                    for y in range(SIZEY):
                        fout.write("    ")
                        for x in range(SIZEX):
                            fout.write(chr(organism.bitstream[x][y] + ord('0')))
                        if y == (SIZEY >> 1):
                            fout.write(" vs. ")
                        else:
                            fout.write("     ")
                        for x in range(SIZEX):
                            fout.write(chr(solution[x][y] + ord('0')))
                        if y == (SIZEY >> 1):
                            fout.write("  =  ")
                        else:
                            fout.write("     ")
                        for x in range(SIZEX):
                            fout.write('+' if (solution[x][y] ^ organism.bitstream[x][y]) == 0 else ' ')
                        fout.write("\n")
                    fout.write("\n\n")
                    print_stuff(organism, fout)
                    break

            fout.write("\nGeneration %d Population statistics:\n" % generation)
            for i in range(maxfit - 1, -1, -1):
                if fitness_bins[i] == 0:
                    continue
                fout.write("%2d %4d" % (i, fitness_bins[i]))
                y = fitness_bins[i] // 100
                for _ in range(y):
                    fout.write("*")
                fout.write("\n")
            fout.write("\n\n")

            fout.write("Solutions searched: %d\n" % find_searchspace(stree, 0))

            if statfile:
                for i in range(maxfit):
                    statfile.write("%d " % fitness_bins[i])
                statfile.write("\n")

            fout.flush()

            # Check terminal input status

            c = read_nonblocking(fd)
            if c is not None:
                c = c.lower()
                if c == 'w':
                    wf = open("pop.dat", "w")
                    for i in range(POPSIZE):
                        organism = population[i]
                        wf.write("Organism %d:\n" % (i + 1))
                        print_stuff(organism, wf)
                        wf.write("\n\n")
                    wf.close()
                    fout.write("Wrote population file pop.dat.")
                elif c == 'm':
                    do_mutations = not do_mutations
                    if do_mutations:
                        sys.stdout.write("mutation enabled at %d%%\n" % mutation_rate)
                    else:
                        sys.stdout.write("mutation disabled\n")
                elif c == '-':
                    mutation_rate -= 5
                    if mutation_rate < 0:
                        mutation_rate = 0
                    sys.stdout.write("mutate at %d%%\n" % mutation_rate)
                elif c == '=':
                    mutation_rate += 5
                    sys.stdout.write("mutate at %d%%\n" % mutation_rate)
                elif c == 'o':
                    if statfile is None:
                        statfile = open("stat.dat", "w")
                        sys.stdout.write("Writing output to stat.dat\n")
                    else:
                        statfile.close()
                        statfile = None
                        sys.stdout.write("Closing stat.dat\n")
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
                        sys.stdout.write("    New solution space? ")
                        sys.stdout.flush()
                        c2 = read_blocking_char(fd)
                        sys.stdout.write("%c\n" % c2)
                        if c2 in ('y', 'Y'):
                            rtype |= GEN_SOLN
                    setnormal(fd)
                    rebuild_requested = True
                    break
                elif c in ('h', '?'):
                    fout.write("Command keystrokes:\n")
                    fout.write("r  run again\n")
                    fout.write("m  toggle mutation\n")
                    fout.write("-  decrease mutation\n")
                    fout.write("=  increase mutation\n")
                    fout.write("w  write population file pop.dat\n")
                    fout.write("o  write statistics file stat.dat\n")
                    fout.write("h  print help\n")
                    fout.write("q  quit\n")
                elif c == 'q':
                    setnormal(fd)
                    fout.write("Done!\n")
                    if statfile:
                        statfile.close()
                    free_solution(stree, 0)
                    sys.exit(0)

            # Create the next generation by crossover recombination

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
                    random_organism(organism, i + 1)
                    continue

                # update history for the new organism
                update_history(organism, parentx, parenty)
                organism.history[0] = i

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

                    # Find the corresponding chromosome in parent X (if any)
                    cpx = None
                    for cand in px.genome:
                        if cand["tag"] == tag:
                            cpx = cand
                            break

                    # Find the corresponding chromosome in parent Y (if any)
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
                            gcopy = cpx["genes"][s]
                            new_genes.append([
                                [list(col) for col in gcopy[0]], gcopy[1]
                            ])
                        new_tag = cpx["tag"]
                        new_split = cpx["split"]
                    else:
                        new_split = 0

                    if cpy is not None:
                        for s in range(cpy["split"], len(cpy["genes"])):
                            gcopy = cpy["genes"][s]
                            new_genes.append([
                                [list(col) for col in gcopy[0]], gcopy[1]
                            ])
                        new_tag = cpy["tag"]

                    # It is possible for the chromosome to be empty
                    if len(new_genes) > 0:
                        cnew = {"tag": new_tag, "genes": new_genes, "split": new_split}
                        # C code prepends new chromosomes to the genome list
                        organism.genome.insert(0, cnew)

            # (No explicit free needed in Python.)

            # Copy the new generation into the current generation
            population = nextgen

            # multi9 formula for the mutation rate:
            #
            # mutation_rate = 0,   no mutation
            # mutation_rate = 50,  50% have 1 mutation, 25% have 2, etc.
            # mutation_rate = 100, 100% have 1 mutation, 50% have 2, etc.
            # mutation_rate = 200, 100% have 1 mutation, 100% have 2, 50% 3

            if do_mutations and mutation_rate > 0:
                for i in range(POPSIZE):
                    m = mutation_rate
                    while m > 0:
                        if random.randrange(100) < m:
                            random_mutate(population[i])
                            m >>= 1
                        else:
                            break

            generation += 1

        if rebuild_requested:
            continue
        break


if __name__ == "__main__":
    main()
