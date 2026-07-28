#!/usr/bin/env python3
#------------------------------------------------------#
# Proof-of concept indirect-mapping genetic algorithm
# January 11, 2004
# Version "species2"
#   An expansion of the idea of chromosomes, based
#   on "linear3" but dividing the total population
#   into multiple subpopulations of "species".
#   Unlike species1, species2 does not allow the
#   species to compete for space in the population.
#
# Python translation of species2.c
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

SPECIES = 16           # number of species
POPSIZE = 512           # total population, each species

STOPGROWTH = 20
STOPFIT = 5

MINRULES = 5
MAXRULES = 20

BITSTREAM = 9
CONTEXTBITS = 3
RESULTSPACE = 1 << CONTEXTBITS
SOLUTIONSPACE = 1 << BITSTREAM


def ABS(a):
    return -a if a < 0 else a


#--------------------------------------------------#
# Gene: a chromosome entry.
#   rule   - list of 3 context symbols
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

cellinit = [0] * (BITSTREAM + 2)    # seed values for target, with guard cells
solution = [0] * SOLUTIONSPACE       # fitness "function"
chemistry = [0] * RESULTSPACE        # mapping of symbols to result bits


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
# Compute "tag" for an organism.  This is a
# simple if imperfect way to quickly tell if
# two organisms have the same genetic code.
#----------------------------------------------#

def compute_tag(organism):
    tag = 0
    for gp in organism.chromosome:
        rule, result = gp
        gword = result
        gword |= (rule[0] << 3)
        gword |= (rule[1] << 6)
        gword |= (rule[2] << 9)
        tag <<= 1
        tag ^= gword
    organism.tag = tag


#-------------------------------------------------#
# Growth algorithm and fitness evaluation routine
#-------------------------------------------------#

def evaluate(organism, fout):
    cellnext = [0] * (BITSTREAM + 2)
    cellarray = [0] * (BITSTREAM + 2)
    outputbits = [0] * BITSTREAM
    key = [0, 0, 0]

    # Initialize cell array
    for x in range(BITSTREAM + 2):
        cellarray[x] = cellinit[x]

    # Add noise to initial configuration by flipping a bit
    p = random.randrange(BITSTREAM + 2)
    s = random.randrange(CONTEXTBITS)
    # cellarray[p] ^= (1 << s)   # left disabled, as in the original C source

    if fout:
        fout.write("      ")
        for x in range(1, BITSTREAM + 1):
            fout.write(chr(cellarray[x] + ord('a')))
        fout.write(" ")

    # Apply growth algorithm

    s = 0
    gmatch = None
    for s in range(STOPGROWTH):
        for x in range(1, BITSTREAM + 1):
            key[0] = cellarray[x - 1]
            key[1] = cellarray[x]
            key[2] = cellarray[x + 1]

            # Find symbolic distance between key and all table entries

            mindist = 1000
            for gp in organism.chromosome:
                rule, result = gp
                distance = ABS(key[0] - rule[0])
                distance += ABS(key[1] - rule[1])
                distance += ABS(key[2] - rule[2])
                if distance < mindist:
                    mindist = distance
                    gmatch = gp
                if mindist == 0:
                    break

            # Apply result from best matching entry
            cellnext[x] = gmatch[1]
            outputbits[x - 1] = chemistry[gmatch[1]]

        if fout:
            if (s % 6) == 5:
                fout.write("\n      ")
            for x in range(1, BITSTREAM + 1):
                fout.write(chr(cellnext[x] + ord('a')))
            fout.write(" ")

        # Check whether result was the same.  If so, we can stop
        # Copy final result back into the cell

        match = 0
        for x in range(1, BITSTREAM + 1):
            if cellarray[x] == cellnext[x]:
                match += 1
            else:
                cellarray[x] = cellnext[x]
        if match == BITSTREAM:
            break
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

    fitness += solution[bitstream]
    organism.fitness = fitness
    organism.bitstream = bitstream
    organism.stop = s

    if fout:
        fout.write("\n      bitstream: ")
        for x in range(BITSTREAM):
            fout.write("%d" % outputbits[x])
        fout.write("  fitness: %d\n" % fitness)


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
    #  5) swap 2 rules between sides
    #  6) move the splitpoint
    #---------------------------------------------------#

    numrules = len(organism.chromosome)

    mtype = 1 + random.randrange(4)

    if mtype == 1:    # Create new random rule
        newrule = [random.randrange(RESULTSPACE) for _ in range(3)]
        newresult = random.randrange(RESULTSPACE)
        newg = [newrule, newresult]

        # Find point in the genome to insert the rule
        rule = random.randrange(numrules + 1)
        organism.chromosome.insert(rule, newg)

        if rule < organism.split:
            organism.split += 1

    elif mtype == 2:  # Find a rule in the genome to remove
        # Don't end up with zero rules!
        if numrules == 1:
            return

        rule = random.randrange(numrules)
        del organism.chromosome[rule]

        if rule < organism.split:
            organism.split -= 1

    elif mtype == 3:  # Choose a rule and modify one context unit
        rule = random.randrange(numrules)
        gp = organism.chromosome[rule]

        # Choose one of the three context rules
        r = random.randrange(3)

        # Modify the rule by one position randomly + or -
        direction = (2 * random.randrange(2)) - 1
        gp[0][r] += direction
        if gp[0][r] < 0:
            gp[0][r] += 1
        elif gp[0][r] >= RESULTSPACE:
            gp[0][r] -= 1

    elif mtype == 4:  # Choose a rule and change the result
        rule = random.randrange(numrules)
        gp = organism.chromosome[rule]

        direction = (2 * random.randrange(2)) - 1
        gp[1] += direction
        if gp[1] < 0:
            gp[1] += 1
        elif gp[1] >= RESULTSPACE:
            gp[1] -= 1

    elif mtype == 5:  # Find two rules in the genome to swap
        # Split point is on one end or the other---ignore
        if organism.split == 0 or organism.split == numrules:
            return

        rule = random.randrange(organism.split)
        rule2 = organism.split + random.randrange(numrules - organism.split)
        gp = organism.chromosome[rule]
        newg = organism.chromosome[rule2]
        for i in range(3):
            gp[0][i], newg[0][i] = newg[0][i], gp[0][i]
        gp[1], newg[1] = newg[1], gp[1]

    elif mtype == 6:  # Move the split point
        if organism.split == 0:
            organism.split += 1
        elif organism.split == numrules:
            organism.split -= 1
        else:
            direction = (2 * random.randrange(2)) - 1   # -1 or +1
            organism.split += direction

    # Recompute the tag
    compute_tag(organism)


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
        rule = [random.randrange(RESULTSPACE) for _ in range(3)]
        result = random.randrange(RESULTSPACE)
        gp = [rule, result]
        # C code prepends to the linked list; preserve that ordering
        organism.chromosome.insert(0, gp)

    organism.split = (r // 2) + random.randrange(2)
    if organism.split == r:
        organism.split -= 1
    elif organism.split == 0:
        organism.split += 1
    compute_tag(organism)


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
        for j in range(3):
            fout.write(chr(rule[j] + ord('a')))

        fout.write(" | %c (%d)\n" % (chr(result + ord('a')), chemistry[result]))

        i += 1
        if i == organism.split:
            fout.write("     ----+------\n")

    fout.write("\n   Growth:\n")
    evaluate(organism, fout)


#----------------------------------------------#
# Here's the application program. . .
#----------------------------------------------#

def main():
    global cellinit, chemistry

    fout = NullIO() if GUI is not None else sys.stdout
    setrandom()

    fd = sys.stdin.fileno()

    # Generate an initial solution to be used by all individuals

    for x in range(BITSTREAM + 2):
        cellinit[x] = random.randrange(RESULTSPACE)

    # Generate the "chemistry", or mapping from symbols to output bits

    for x in range(RESULTSPACE):
        chemistry[x] = random.randrange(2)

    # Print the initial cell

    fout.write("Initial cell:\n   ")
    for x in range(1, BITSTREAM + 1):
        fout.write(chr(cellinit[x] + ord('a')))
    fout.write("\n\n")

    # Initialization routine:  Generate a target "application"
    # With power-law solution.

    maxfit = 0
    for i in range(SOLUTIONSPACE):
        r = random.randrange(SOLUTIONSPACE - 1)
        solution[i] = 0
        while r & 0x1:
            solution[i] += 1
            r >>= 1
        if solution[i] > maxfit:
            maxfit = solution[i]

    # If maxfit is not equal to BITSTREAM - 1, then choose a random
    # solution and set it to this value.

    if maxfit < (BITSTREAM - 1):
        solution[random.randrange(SOLUTIONSPACE)] = BITSTREAM - 1
        maxfit = BITSTREAM - 1

    fout.write("Solutions with maximum fitness %d:\n" % maxfit)
    for i in range(SOLUTIONSPACE):
        if solution[i] == maxfit:
            fout.write("0x%03x\n" % i)
    fout.write("\nSolutions with penultimate fitness %d:\n" % (maxfit - 1))
    for i in range(SOLUTIONSPACE):
        if solution[i] == (maxfit - 1):
            fout.write("0x%03x\n" % i)
    fout.write("\n")

    # Generate a population of random chromosomes

    population = []
    for j in range(SPECIES):
        pop_j = [Individual() for _ in range(POPSIZE)]
        for i in range(POPSIZE):
            random_organism(pop_j[i])
        population.append(pop_j)

    fout.write("Hit any key to start:")
    fout.flush()
    if GUI is None:
        sys.stdin.read(1)
    fout.write("\n\n")
    setnodelay(fd)

    fitness_bins = [0] * (BITSTREAM + STOPFIT)
    maxfitorg = [None] * SPECIES

    generation = 0
    while True:

        # Over each species (separately)

        pairings_x = [[None] * POPSIZE for _ in range(SPECIES)]
        pairings_y = [[None] * POPSIZE for _ in range(SPECIES)]

        for j in range(SPECIES):

            # Evaluate fitness for each organism in the population.

            totalfit = 0
            for i in range(POPSIZE):
                organism = population[j][i]
                evaluate(organism, None)
                totalfit += organism.fitness

            # Each individual gets to take part in a number of matings
            # according to (individual fitness / population fitness) times
            # (population size) times 2.  First-come, first-served.

            maxfitorg[j] = population[j][0]
            residual = 0
            binsleft = POPSIZE
            for i in range(POPSIZE):
                organism = population[j][i]

                # While we're doing this, find the most fit individual
                if organism.fitness > maxfitorg[j].fitness:
                    maxfitorg[j] = organism

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
                    if pairings_x[j][x] is None:
                        pairings_x[j][x] = organism
                    else:
                        # This block is necessary to prevent mating an
                        # organism with itself or another individual with
                        # the same genome.  The checksum-like "tag"
                        # mechanism is ad-hoc but much faster than
                        # exhaustively checking each organism's genome
                        # against all others.

                        if pairings_x[j][x].tag == organism.tag:
                            tries += 1
                            if tries < 10:
                                s -= 1
                            else:
                                # Okay, see if there are any bins we can use
                                y = 0
                                found = False
                                while y < binsleft:
                                    if pairings_x[j][y] is None:
                                        pairings_x[j][y] = organism
                                        tries = 0
                                        found = True
                                        break
                                    elif pairings_x[j][y].tag != organism.tag:
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
                                        if pairings_x[j][x].tag != pairings_x[j][y2].tag:
                                            pairings_y[j][x] = pairings_x[j][y2]
                                            found2 = True
                                            break
                                    if not found2:
                                        # Everybody has the same tag!  This
                                        # should not occur but if it does,
                                        # we start randomly mutating.
                                        for y3 in range(POPSIZE):
                                            if pairings_x[j][y3] is not organism:
                                                while pairings_x[j][y3].tag == organism.tag:
                                                    random_mutate(pairings_x[j][y3])
                                                break

                        if pairings_x[j][x].tag != organism.tag:
                            # swap the contents of this cell and cell at (binsleft - 1)
                            save_x = pairings_x[j][binsleft - 1]
                            save_y = pairings_y[j][binsleft - 1]
                            pairings_x[j][binsleft - 1] = pairings_x[j][x]
                            pairings_y[j][binsleft - 1] = organism
                            pairings_x[j][x] = save_x
                            pairings_y[j][x] = save_y

                            binsleft -= 1
                            if binsleft == 0:
                                break
                    s += 1
                if binsleft == 0:
                    break

        # Done with the population.  Report results

        fout.write("\nGeneration %d: " % generation)

        if GUI is not None:
            allfits = [population[j][i].fitness
                       for j in range(SPECIES) for i in range(POPSIZE)]
            best = max((maxfitorg[j] for j in range(SPECIES)), key=lambda o: o.fitness)
            GUI.report({
                "generation":   generation,
                "fitnesses":    allfits,
                "best_fitness": best.fitness,
                "mean_fitness": sum(allfits) / len(allfits),
                "max_fitness":  BITSTREAM + STOPFIT,
                "best_grid":    [[(best.bitstream >> b) & 1 for b in range(BITSTREAM)]],
                "target_grid":  None,
                "best_stop":    best.stop,
                "extra":        {"species": SPECIES},
            })
            GUI.checkpoint()

        for j in range(SPECIES):
            fout.write(
                "Maximally fit organism fitness %d bits 0x%x stop %d\n"
                % (maxfitorg[j].fitness, maxfitorg[j].bitstream, maxfitorg[j].stop)
            )

            print_stuff(maxfitorg[j], fout)

            # Bin the population by fitness level and report
            for i in range(BITSTREAM + STOPFIT):
                fitness_bins[i] = 0
            for i in range(POPSIZE):
                organism = population[j][i]
                fitness_bins[organism.fitness] += 1
            fout.write("\nPopulation %d statistics:\n" % (j + 1))
            for i in range(BITSTREAM + STOPFIT - 1, -1, -1):
                fout.write("%d->%d " % (i, fitness_bins[i]))
            fout.write("\n\n")

        fout.flush()

        # Check terminal input status

        c = None if GUI is not None else read_nonblocking(fd)
        if c is not None:
            c = c.lower()
            if c == 'w':
                wf = open("pop.dat", "w")
                for j in range(SPECIES):
                    for i in range(POPSIZE):
                        organism = population[j][i]
                        wf.write("Organism %d:\n" % (i + 1))
                        print_stuff(organism, wf)
                        wf.write("\t   species=%d\n" % (j + 1))
                        wf.write("      tag=%d\n\n" % organism.tag)
                        wf.write("\n")
                wf.close()
            elif c == 'q':
                setnormal(fd)
                fout.write("Done!\n")
                sys.exit(0)

        # Create the next generation by crossover recombination

        for j in range(SPECIES):
            nextgen_j = [Individual() for _ in range(POPSIZE)]

            for i in range(POPSIZE):
                organism = nextgen_j[i]
                organism.fitness = 0
                organism.chromosome = []

                # Get parents from the "pairings" table
                parentx = pairings_x[j][i]
                parenty = pairings_y[j][i]

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
                compute_tag(organism)

            # (No explicit free needed in Python.)

            # Copy the new generation into the current generation
            population[j] = nextgen_j

            # Random mutation at 1%
            for i in range(POPSIZE):
                if random.randrange(100) == 0:
                    random_mutate(population[j][i])

        generation += 1


if __name__ == "__main__":
    main()
