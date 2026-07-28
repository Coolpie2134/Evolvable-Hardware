#!/usr/bin/env python3
#------------------------------------------------------#
# Proof-of concept indirect-mapping genetic algorithm
# January 14, 2004
# Version "linear12"
#   Like "linear9", but computes the score as a
#   binomial---individually correct bits score 1,
#   correct sequences of 2 bits score 2, etc.
#   The hope is that partially correct sequences
#   will be heavily weighted, and meaningless
#   sequences will be quickly squashed.
#
# Python translation of linear12.c
#------------------------------------------------------#

import os
import random
import sys
import select

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

POPSIZE = 8092
STOPGROWTH = 30
STOPFIT = 15

MINRULES = 6
MAXRULES = 15

KERNEL = 5                       # should be an odd number
HALFKERNEL = KERNEL >> 1          # center position in kernel
BITSTREAM = 16
RESULTSPACE = 5
ENDSYMBOL = 100


def ABS(a):
    return -a if a < 0 else a


GEN_POP = 1
GEN_START = 2
GEN_CHEM = 4
GEN_SOLN = 8
GEN_ALL = GEN_POP | GEN_START | GEN_CHEM | GEN_SOLN

#--------------------------------------------------#
# Gene: a chromosome entry.
#   rule   - list of KERNEL context symbols
#   result - next-state symbol
#
# A chromosome is a Python list of [rule_list, result]
# entries, taking the place of the C linked list.
#--------------------------------------------------#


class Individual:
    __slots__ = ("chromosome", "split", "fitness", "bitstream", "stop", "tag")

    def __init__(self):
        self.chromosome = []
        self.split = 0
        self.fitness = 0
        self.bitstream = 0
        self.stop = 0
        self.tag = 0


#------------------*
# Global variables
#------------------*

cellinit = [0] * BITSTREAM       # seed values for target, with guard cells
solution = 0                      # target result
chemistry = [0] * RESULTSPACE     # mapping of symbols to result bits


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
# Fitness function.
# Count all groups of bits larger than 1 that
# match the solution.  Multiply the result by
# the length of the correct sequence
#----------------------------------------------#

def fitness_function(bitstream):
    result = 0
    solnxnor = (~(bitstream ^ solution)) & ((1 << BITSTREAM) - 1)

    for x in range(BITSTREAM - 1):
        for k in range(2, BITSTREAM + 1 - x):
            mask = (1 << k) - 1
            if (solnxnor & mask) == mask:
                result += 1
        solnxnor >>= 1
    return result


#-------------------------------------------------#
# Growth algorithm and fitness evaluation routine
#-------------------------------------------------#

def evaluate(organism, fout):
    cellnext = [0] * BITSTREAM
    cellarray = [0] * BITSTREAM
    outputbits = [0] * BITSTREAM
    key = [0] * KERNEL

    # Initialize cell array
    for x in range(BITSTREAM):
        cellarray[x] = cellinit[x]

    if fout:
        fout.write("      ")
        for x in range(BITSTREAM):
            fout.write(chr(cellarray[x] + ord('a')))
        fout.write(" ")

    # Apply growth algorithm

    s = 0
    gmatch = None
    tag = 0
    for s in range(STOPGROWTH):
        for x in range(BITSTREAM):
            for i in range(KERNEL):
                n = x + i - HALFKERNEL
                if n < 0 or n >= BITSTREAM:
                    key[i] = ENDSYMBOL
                else:
                    key[i] = cellarray[n]

            # Find symbolic distance between key and all table entries

            mindist = 1000
            for gp in organism.chromosome:
                rule, result = gp
                distance = 0
                for i in range(KERNEL):
                    sdist = key[i] - rule[i]
                    distance += ABS(sdist)
                if distance < mindist:
                    mindist = distance
                    gmatch = gp
                if mindist == 0:
                    break

            # Apply result from best matching entry
            cellnext[x] = gmatch[1]
            outputbits[x] = chemistry[gmatch[1]]

        if fout:
            if (s % 6) == 5:
                fout.write("\n      ")
            for x in range(BITSTREAM):
                fout.write(chr(cellnext[x] + ord('a')))
            fout.write(" ")

        # Check whether result was the same.  If so, we can stop
        # Copy final result back into the cell

        match = 0
        for x in range(BITSTREAM):
            if cellarray[x] == cellnext[x]:
                match += 1
            else:
                cellarray[x] = cellnext[x]
        if match == BITSTREAM:
            break

        # Compute the tag.  This is a simple calculation meant to produce
        # the same result for individuals with the same growth pattern.

        gword = 0
        for x in range(BITSTREAM):
            gword <<= 1
            gword |= outputbits[x]
        tag <<= 1
        tag ^= gword
    else:
        s = STOPGROWTH

    # End of growth.  If we stopped early because growth was
    # stopped "naturally", add STOPFIT to the initial fitness value.

    fitness = STOPFIT if (s < STOPGROWTH) else 0

    # Determine the resulting bitstream, and find the
    # fitness value for that result.

    bitstream = 0
    for x in range(BITSTREAM):
        bitstream |= outputbits[x] << x

    fitness += fitness_function(bitstream)

    organism.fitness = fitness
    organism.bitstream = bitstream
    organism.stop = s
    organism.tag = tag

    if fout:
        fout.write("\n      bitstream: ")
        for x in range(BITSTREAM):
            fout.write("%d" % outputbits[x])
        fout.write("  fitness: %d  tag: %d\n" % (fitness, tag))


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

    numrules = len(organism.chromosome)

    if numrules == 0:
        mtype = 2             # can only add a rule!
    elif numrules == 1:
        mtype = 2 + random.randrange(3)   # don't remove the last rule!
    else:
        mtype = 1 + random.randrange(4)

    if mtype == 1:    # Find a rule in the genome to remove
        rule = random.randrange(numrules)
        del organism.chromosome[rule]
        if rule < organism.split:
            organism.split -= 1

    elif mtype == 2:  # Create new random rule
        newrule = [random.randrange(RESULTSPACE) for _ in range(KERNEL)]

        # Treat ENDSYMBOL specially.  ENDSYMBOL should only exist on one
        # side or the other, not both, never in the center position, and
        # there should be no other symbols between ENDSYMBOL and the
        # start or end.  Correct: xxabc, xxccb, xabcd.  Incorrect:
        # xabcx, axabc, xxxdc.

        if random.randrange(RESULTSPACE + 1) == 0:
            r = random.randrange(HALFKERNEL)
            if random.randrange(2) == 0:
                for i in range(r + 1):
                    newrule[i] = ENDSYMBOL
            else:
                for i in range(KERNEL - 1, (KERNEL - 1) - r - 1, -1):
                    newrule[i] = ENDSYMBOL

        newresult = random.randrange(RESULTSPACE)
        newg = [newrule, newresult]

        # Find point in the genome to insert the rule
        rule = random.randrange(numrules + 1)
        organism.chromosome.insert(rule, newg)

        if rule < organism.split:
            organism.split += 1

    elif mtype == 3:  # Choose a rule and modify one context unit
        rule = random.randrange(numrules)
        gp = organism.chromosome[rule]

        # Choose one of the KERNEL context positions
        r = random.randrange(KERNEL)

        if gp[0][r] == ENDSYMBOL:
            gp[0][r] = random.randrange(RESULTSPACE)
        else:
            # Neighbor rules may be set to ENDSYMBOL
            if r != HALFKERNEL and random.randrange(RESULTSPACE) == 0:
                gp[0][r] = ENDSYMBOL
            else:
                # Modify the rule by one position randomly + or -
                if gp[0][r] == 0:
                    gp[0][r] += 1
                elif gp[0][r] == (RESULTSPACE - 1):
                    gp[0][r] -= 1
                else:
                    direction = (2 * random.randrange(2)) - 1
                    gp[0][r] += direction

    elif mtype == 4:  # Choose a rule and change the result
        rule = random.randrange(numrules)
        gp = organism.chromosome[rule]

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

def random_organism(organism):
    organism.fitness = 0
    organism.bitstream = 0
    organism.stop = 0
    organism.chromosome = []

    r = MINRULES + random.randrange(MAXRULES - MINRULES + 1)
    for _ in range(r):
        rule = [random.randrange(RESULTSPACE) for _ in range(KERNEL)]

        # See random_mutate() for explanation of ENDSYMBOL positioning
        if random.randrange(RESULTSPACE + 1) == 0:
            n = random.randrange(HALFKERNEL)
            if random.randrange(2) == 0:
                for i in range(n + 1):
                    rule[i] = ENDSYMBOL
            else:
                for i in range(KERNEL - 1, (KERNEL - 1) - n - 1, -1):
                    rule[i] = ENDSYMBOL

        result = random.randrange(RESULTSPACE)
        gp = [rule, result]
        # C code prepends to the linked list; preserve that ordering
        organism.chromosome.insert(0, gp)

    organism.split = (r // 2) + random.randrange(2)
    if organism.split == r:
        organism.split -= 1
    elif organism.split == 0:
        organism.split += 1


#----------------------------------------------#
# Print the chromosome table and growth
# pattern of an organism.
#----------------------------------------------#

def print_stuff(organism, fout):
    fout.write("   Chromosome table:\n")
    i = 0
    for gp in organism.chromosome:
        rule, result = gp
        fout.write("     ")
        for j in range(KERNEL):
            fout.write("x" if rule[j] == ENDSYMBOL else chr(rule[j] + ord('a')))

        fout.write(" | %c (%d)\n" % (chr(result + ord('a')), chemistry[result]))

        i += 1
        if i == organism.split:
            fout.write("     ")
            for _ in range(KERNEL + 1):
                fout.write("-")
            fout.write("+------\n")

    fout.write("\n   Growth:\n")
    evaluate(organism, fout)


#----------------------------------------------#
# Here's the application program. . .
#----------------------------------------------#

def main():
    global solution, cellinit, chemistry

    fout = NullIO() if GUI is not None else sys.stdout
    setrandom()

    population = None
    fitness_bins = None
    maxfit = 0

    rtype = GEN_ALL
    do_mutate = True
    be_elitist = False

    fd = sys.stdin.fileno()

    while True:   # corresponds to the "rebuild:" label in the C source
        # Generate an initial solution to be used by all individuals

        if rtype & GEN_START:
            for x in range(BITSTREAM):
                # cellinit[x] = random.randrange(RESULTSPACE)
                cellinit[x] = 0

        # Generate the "chemistry", or mapping from symbols to output bits

        if rtype & GEN_CHEM:
            while True:
                btot = 0
                for x in range(RESULTSPACE):
                    chemistry[x] = random.randrange(2)
                    btot += chemistry[x]
                # Avoid having all zeros or all ones
                if 0 < btot < RESULTSPACE:
                    break

        # Print the initial cell

        fout.write("Initial cell:\n   ")
        for x in range(BITSTREAM):
            fout.write(chr(cellinit[x] + ord('a')))
        fout.write("\n\n")

        # Initialization routine:  Generate a target "application"

        if (rtype & GEN_SOLN) or (fitness_bins is None):
            solution = random.randrange((1 << BITSTREAM) - 1)
            maxfit = fitness_function(solution) + STOPFIT
            fitness_bins = [0] * maxfit

        fout.write("Solution 0x%x fitness %d:\n" % (solution, maxfit))

        # Generate a population of random chromosomes

        if rtype & GEN_POP:
            population = [Individual() for _ in range(POPSIZE)]
            for i in range(POPSIZE):
                random_organism(population[i])

        fout.write("When running, w=write, r=re-run, m=mutation e=elitism q=quit.\n")
        fout.write("Hit any key to start:")
        fout.flush()
        if GUI is None:
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
                totalfit += organism.fitness

            # Each individual gets to take part in a number of matings
            # according to (individual fitness / population fitness) times
            # (population size) times 2.  First-come, first-served.

            pairings_x = [None] * POPSIZE
            pairings_y = [None] * POPSIZE

            residual = 0
            binsleft = POPSIZE
            for i in range(POPSIZE):
                organism = population[i]
                fitshare = organism.fitness * POPSIZE * 2
                residual += (fitshare % totalfit)
                matings = fitshare // totalfit

                # Keep track of fractional fitness so we make sure that the
                # number of pairings comes out equal to the population size.
                if residual > 0:
                    matings += 1
                    residual -= totalfit

                tries = 0
                s = 0
                while s < matings:
                    x = random.randrange(binsleft)
                    if be_elitist:    # simple but effective elitism model
                        y = organism.fitness + 9 - (maxfit - 1)
                        if y > 0:
                            x >>= y
                    if pairings_x[x] is None:
                        pairings_x[x] = organism
                    else:
                        # This block prevents mating an organism with
                        # itself or another individual with the same
                        # genome.  The checksum-like "tag" mechanism is
                        # ad-hoc but much faster than exhaustively
                        # checking each organism's genome against all
                        # others.

                        if pairings_x[x].tag == organism.tag:
                            tries += 1
                            if tries < 10:
                                s -= 1
                            else:
                                # Okay, see if there are any bins we can use
                                y = 0
                                found = False
                                while y < binsleft:
                                    if pairings_x[y] is None:
                                        pairings_x[y] = organism
                                        tries = 0
                                        found = True
                                        break
                                    elif pairings_x[y].tag != organism.tag:
                                        x = y
                                        tries = 0
                                        found = True
                                        break
                                    y += 1
                                if not found:
                                    # There are no bins left that do not
                                    # contain the organism itself.  Some
                                    # lucky organism gets another chance.
                                    found2 = False
                                    for y2 in range(y, POPSIZE):
                                        if pairings_x[x].tag != pairings_x[y2].tag:
                                            pairings_y[x] = pairings_x[y2]
                                            found2 = True
                                            break
                                    if not found2:
                                        # Everybody has the same tag!  This
                                        # should not occur but if it does,
                                        # we start randomly mutating.
                                        for y3 in range(POPSIZE):
                                            if pairings_x[y3] is not organism:
                                                while pairings_x[y3].tag == organism.tag:
                                                    random_mutate(pairings_x[y3])
                                                break

                        if pairings_x[x].tag != organism.tag:
                            # swap the contents of this cell and cell at (binsleft - 1)
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

            # Bin the population by fitness level and report
            for i in range(maxfit):
                fitness_bins[i] = 0
            for i in range(POPSIZE):
                organism = population[i]
                fitness_bins[organism.fitness] += 1

            # Which bin represents the largest portion of the population?
            maxbin = 0
            for i in range(1, maxfit):
                if fitness_bins[i] > fitness_bins[maxbin]:
                    maxbin = i

            if GUI is not None:
                fits = [o.fitness for o in population]
                typ  = next((o for o in population if o.fitness == maxbin), population[0])
                GUI.report({
                    "generation":   generation,
                    "fitnesses":    fits,
                    "best_fitness": max(fits),
                    "mean_fitness": sum(fits) / len(fits),
                    "max_fitness":  maxfit,
                    "best_grid":    [[(typ.bitstream >> b) & 1 for b in range(BITSTREAM)]],
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
                        "Typical organism fitness %d bits 0x%x stop %d\n"
                        % (maxbin, organism.bitstream, organism.stop)
                    )
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
            fout.flush()

            # Check terminal input status

            c = None if GUI is not None else read_nonblocking(fd)
            if c is not None:
                c = c.lower()
                if c == 'w':
                    wf = open("pop.dat", "w")
                    for i in range(POPSIZE):
                        organism = population[i]
                        wf.write("Organism %d:\n" % (i + 1))
                        print_stuff(organism, wf)
                        wf.write("      tag=%d\n\n" % organism.tag)
                        wf.write("\n")
                    wf.close()
                elif c == 'e':
                    be_elitist = not be_elitist
                    if be_elitist:
                        sys.stdout.write("elitism enabled\n")
                    else:
                        sys.stdout.write("elitism disabled\n")
                elif c == 'm':
                    do_mutate = not do_mutate
                    if do_mutate:
                        sys.stdout.write("mutation enabled at 1%\n")
                    else:
                        sys.stdout.write("mutation disabled\n")
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
                elif c == 'q':
                    setnormal(fd)
                    fout.write("Done!\n")
                    sys.exit(0)

            # Create the next generation by crossover recombination

            nextgen = [Individual() for _ in range(POPSIZE)]

            for i in range(POPSIZE):
                organism = nextgen[i]
                organism.fitness = 0
                organism.chromosome = []

                # Get parents from the "pairings" table
                parentx = pairings_x[i]
                parenty = pairings_y[i]

                # If one or both parents is undeclared, then
                # randomly generate a new organism.

                if parentx is None or parenty is None:
                    random_organism(organism)
                    continue

                # Crossover combination:
                # 1st half is from x, 2nd half is from y

                new_chrom = []
                for s in range(parentx.split):
                    gcopy = parentx.chromosome[s]
                    new_chrom.append([list(gcopy[0]), gcopy[1]])

                for s in range(parenty.split, len(parenty.chromosome)):
                    gcopy = parenty.chromosome[s]
                    new_chrom.append([list(gcopy[0]), gcopy[1]])

                organism.chromosome = new_chrom
                organism.split = parentx.split

            # (No explicit free needed in Python.)

            # Copy the new generation into the current generation
            population = nextgen

            # Random mutation at 1%
            if do_mutate:
                for i in range(POPSIZE):
                    if random.randrange(100) == 0:
                        random_mutate(population[i])

            generation += 1

        if rebuild_requested:
            continue
        break


if __name__ == "__main__":
    main()
