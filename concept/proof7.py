#!/usr/bin/env python3
#------------------------------------------------------#
# Proof-of concept indirect-mapping genetic algorithm
# December 31, 2003
# Version 7
#   Expanded version of #6
#
# Python translation of proof7.c
#------------------------------------------------------#

import random
import sys

#--------------------------------------------------#
# Constants
#--------------------------------------------------#

POPSIZE = 1024
MAXGEN = 10000
STOPGROWTH = 20
STOPFIT = 5

SIZE_X = 5
SIZE_Y = 5

SOLUTIONSPACE = 65536
FITBITS = 15

#--------------------------------------------------#
# Gene: one associative-memory entry
#   rule   - 9-bit context key
#   result - output bit (0 or 1)
#
# A chromosome is represented as a Python list of
# [rule, result] genes (in place of the C linked list).
# List order corresponds to the original linked-list
# order (each new gene was pushed onto the head of the
# list in C, i.e. prepended); we preserve that ordering
# convention by always inserting/removing at index 0
# where the C code did the same.
#--------------------------------------------------#


class Individual:
    __slots__ = ("chromosome", "split", "fitness", "bitstream", "stop")

    def __init__(self):
        self.chromosome = []   # list of [rule, result]
        self.split = 0
        self.fitness = 0
        self.bitstream = 0
        self.stop = 0


#--------------------------------------------------#
# Global variables
#--------------------------------------------------#

cellinit = [[0] * SIZE_Y for _ in range(SIZE_X)]   # seed values for target
solution_bitstream = [0] * SOLUTIONSPACE            # map for determining fitness
solution_fitness = [0] * SOLUTIONSPACE


#----------------------------------------------------------------------#
# Routine to set the random seed from the current time in microseconds
#----------------------------------------------------------------------#

def solution(bitstream):
    maxspace = 1 << (SIZE_X * SIZE_Y)

    if SOLUTIONSPACE > maxspace:
        return solution_fitness[bitstream]

    maxmatch = 0
    matchfit = 0
    for i in range(SOLUTIONSPACE):
        match = bitstream ^ solution_bitstream[i]
        matchbits = 0
        while match:
            if not (match & 0x1):
                matchbits += 1
            match >>= 1
        if matchbits > maxmatch:
            maxmatch = matchbits
            matchfit = solution_fitness[i]
        if maxmatch == (SIZE_X * SIZE_Y):
            break
    return matchfit


#----------------------------------------------#
# Solution-space initial generation
#----------------------------------------------#

def init_solution():
    maxspace = 1 << (SIZE_X * SIZE_Y)
    fitspace = 1 << FITBITS
    solutionspace = SOLUTIONSPACE

    if solutionspace > maxspace:
        solutionspace = maxspace

    # Initialization routine: Generate a target "application"
    # With power-law solution.

    maxfit = 0
    for i in range(solutionspace):
        if solutionspace == maxspace:
            solution_bitstream[i] = i
        else:
            rl = random.randrange(maxspace)
            solution_bitstream[i] = rl

        r = random.randrange(fitspace)
        fit = 0
        while r & 0x1:
            fit += 1
            r >>= 1
        solution_fitness[i] = fit
        if fit > maxfit:
            maxfit = fit

    for i in range(solutionspace, SOLUTIONSPACE):
        solution_bitstream[i] = i
        solution_fitness[i] = 0

    sys.stdout.write("Solutions with maximum fitness %d:\n" % maxfit)
    for i in range(solutionspace):
        if solution_fitness[i] == maxfit:
            sys.stdout.write("0x%04x\n" % i)
    sys.stdout.write("\n")


#-------------------------------------------------#
# Growth algorithm and fitness evaluation routine
#-------------------------------------------------#

def evaluate(organism, doprint):
    cellnext = [[0] * SIZE_Y for _ in range(SIZE_X)]
    cellarray = [[0] * SIZE_Y for _ in range(SIZE_X)]

    # Initialize cell array
    for y in range(SIZE_Y):
        for x in range(SIZE_X):
            cellarray[x][y] = cellinit[x][y]

    # Randomize start point by flipping a bit from the original
    # (version 5 test)
    x = random.randrange(SIZE_X)
    y = random.randrange(SIZE_Y)
    cellarray[x][y] ^= 1

    if doprint:
        for y in range(SIZE_Y):
            for x in range(SIZE_X):
                sys.stdout.write("%d" % cellarray[x][y])
            sys.stdout.write("\n")
        sys.stdout.write("\n")

    # Apply growth algorithm

    s = 0
    gmatch = None
    for s in range(STOPGROWTH):
        for y in range(SIZE_Y):
            for x in range(SIZE_X):
                xp = (x + 1) % SIZE_X     # cell at x + 1, with wraparound
                yp = (y + 1) % SIZE_Y     # cell at y + 1, with wraparound
                xm = (x + SIZE_X - 1) % SIZE_X   # cell at x - 1, with wraparound
                ym = (y + SIZE_Y - 1) % SIZE_Y   # cell at y - 1, with wraparound

                key = cellarray[x][y]                  # self
                key |= cellarray[xm][y] << 1           # east
                key |= cellarray[xp][y] << 2           # west
                key |= cellarray[x][yp] << 3           # south
                key |= cellarray[x][ym] << 4           # north
                key |= cellarray[xm][yp] << 5           # southeast
                key |= cellarray[xm][ym] << 6           # northeast
                key |= cellarray[xp][yp] << 7           # southwest
                key |= cellarray[xp][ym] << 8           # northwest

                # Find hamming distance between key and all table entries
                matchmax = 0
                for gp in organism.chromosome:
                    match = gp[0] ^ key
                    matchcount = 0
                    for _ in range(9):
                        if not (match & 0x1):
                            matchcount += 1
                        match >>= 1
                    if matchcount > matchmax:
                        matchmax = matchcount
                        gmatch = gp

                if doprint > 1:
                    sys.stdout.write(
                        "Key = 0x%03x, Rule = 0x%03x, Result = %d\n"
                        % (key, gmatch[0], gmatch[1])
                    )

                # Apply result from best matching entry
                cellnext[x][y] = gmatch[1]

        if doprint:
            for y in range(SIZE_Y):
                for x in range(SIZE_X):
                    sys.stdout.write("%d" % cellnext[x][y])
                sys.stdout.write("\n")
            sys.stdout.write("\n")

        # Check whether result was the same. If so, we can stop
        match_count = 0
        for y in range(SIZE_Y):
            for x in range(SIZE_X):
                if cellarray[x][y] == cellnext[x][y]:
                    match_count += 1
        if match_count == (SIZE_X * SIZE_Y):
            break

        # Copy final result back into the cell
        for y in range(SIZE_Y):
            for x in range(SIZE_X):
                cellarray[x][y] = cellnext[x][y]
    else:
        # Loop completed without break: s ends at STOPGROWTH - 1, but
        # the C for-loop leaves s == STOPGROWTH in that case. Match that.
        s = STOPGROWTH

    # End of growth. If we stopped early because growth was
    # stopped "naturally", add STOPFIT to the initial fitness value.

    fitness = STOPFIT if (s < STOPGROWTH) else 0

    # Collect the array bits into an integer, and find the
    # fitness value for that result.

    bitstream = 0
    p = 0
    for x in range(SIZE_X):
        for y in range(SIZE_Y):
            bitstream |= (cellarray[x][y] << p)
            p += 1

    fitness += solution(bitstream)
    organism.fitness = fitness
    organism.bitstream = bitstream
    organism.stop = s


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
    #  5) swap 2 rules between sides [disabled]
    #  6) move the splitpoint [disabled]
    #---------------------------------------------------#

    numrules = len(organism.chromosome)

    mtype = 1 + random.randrange(4)

    if mtype == 1:    # Create new random rule
        newg = [random.randrange(512), random.randrange(2)]

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

    elif mtype == 3:  # Choose a rule and modify (invert) 1 bit
        rule = random.randrange(numrules)
        organism.chromosome[rule][0] ^= (1 << random.randrange(9))

    elif mtype == 4:  # Choose a rule and modify (invert) the result
        rule = random.randrange(numrules)
        organism.chromosome[rule][1] ^= 1

    elif mtype == 5:  # Find two rules in the genome to swap
        # Split point is on one end or the other---ignore
        if organism.split == 0 or organism.split == numrules:
            return

        rule = random.randrange(organism.split)
        rule2 = organism.split + random.randrange(numrules - organism.split)
        gp = organism.chromosome[rule]
        newg = organism.chromosome[rule2]
        gp[0], newg[0] = newg[0], gp[0]
        gp[1], newg[1] = newg[1], gp[1]

    elif mtype == 6:  # Move the split point
        if organism.split == 0:
            organism.split += 1
        elif organism.split == numrules:
            organism.split -= 1
        else:
            direction = (2 * random.randrange(2)) - 1   # -1 or +1
            organism.split += direction


#----------------------------------------------#
# Randomly generate a new organism
#----------------------------------------------#

def random_organism(organism):
    organism.fitness = 0
    organism.bitstream = 0
    organism.stop = 0
    organism.chromosome = []

    r = 2 + random.randrange(15)    # 2 to 16 rules each
    for _ in range(r):
        gp = [random.randrange(512), random.randrange(2)]
        # C code prepends to the linked list; preserve that ordering
        organism.chromosome.insert(0, gp)

    organism.split = (r // 2) + random.randrange(2)
    if organism.split == r:
        organism.split -= 1
    elif organism.split == 0:
        organism.split += 1


#----------------------------------------------#
# Here's the application program. . .
#----------------------------------------------#

def main():
    random.seed()

    # Generate a random initial seed to be used by all individuals
    for y in range(SIZE_Y):
        for x in range(SIZE_X):
            cellinit[x][y] = random.randrange(2)

    # A not-so-random initial state which preserves orientation
    # (left disabled, as in the original C source)
    # for y in range(SIZE_Y):
    #     for x in range(SIZE_X):
    #         cellinit[x][y] = 0
    # cellinit[0][SIZE_Y - 1] = 1
    # cellinit[0][SIZE_Y - 2] = 1
    # cellinit[1][SIZE_Y - 1] = 1

    # Print the initial cell and the corresponding bitstream

    sys.stdout.write("Initial cell:\n   ")
    for y in range(SIZE_Y):
        for x in range(SIZE_X):
            sys.stdout.write("%d" % cellinit[x][y])
        sys.stdout.write("\n   ")

    bitstream = 0
    i = 0
    for x in range(SIZE_X):
        for y in range(SIZE_Y):
            bitstream |= (cellinit[x][y] << i)
            i += 1
    sys.stdout.write("\nInitial cell corresponds to bitstream 0x%03x\n\n" % bitstream)

    init_solution()

    # Generate a population of random chromosomes

    population = [Individual() for _ in range(POPSIZE)]
    for i in range(POPSIZE):
        random_organism(population[i])

    for generation in range(MAXGEN):

        # Evaluate fitness for each organism in the population.

        totalfit = 0
        for i in range(POPSIZE):
            organism = population[i]
            evaluate(organism, 0)
            totalfit += organism.fitness

        # Each individual gets to take part in a number of matings
        # according to (individual fitness / population fitness) times
        # (population size) times 2. First-come, first-served.

        pairings_x = [None] * POPSIZE
        pairings_y = [None] * POPSIZE

        maxfitorg = population[0]
        residual = 0
        binsleft = POPSIZE
        i = 0
        for i in range(POPSIZE):
            organism = population[i]

            # While we're doing this, find the most fit individual
            if organism.fitness > maxfitorg.fitness:
                maxfitorg = organism

            fitshare = organism.fitness * POPSIZE * 2
            residual += (fitshare % totalfit)
            matings = fitshare // totalfit

            # Keep track of fractional fitness so we make sure that the
            # number of pairings comes out equal to the population size.
            if residual > 0:
                matings += 1
                residual -= totalfit

            for _s in range(matings):
                if binsleft == 0:
                    break
                x = random.randrange(binsleft)
                y = -1
                while x >= 0:
                    y += 1
                    while pairings_x[y] is not None and pairings_y[y] is not None:
                        y += 1
                        if y == POPSIZE:
                            sys.stdout.write("ERROR:  Overrun!\n")
                            binsleft = 0
                            break
                    x -= 1
                if pairings_x[y] is None:
                    pairings_x[y] = organism
                elif pairings_y[y] is None:
                    pairings_y[y] = organism
                    binsleft -= 1
            if binsleft == 0:
                break

        # Diagnostic

        if binsleft == 0:
            sys.stdout.write("Population filled at i = %d\n" % i)
        else:
            sys.stdout.write("Population done with %d bins unfilled\n" % binsleft)

        # Done with the population. Report results

        sys.stdout.write("Generation %d: " % generation)

        sys.stdout.write(
            "Maximally fit organism fitness %d bits 0x%x stop %d\n"
            % (maxfitorg.fitness, maxfitorg.bitstream, maxfitorg.stop)
        )

        sys.stdout.write("   Chromosome table:\n")
        for gp in maxfitorg.chromosome:
            sys.stdout.write("  0x%03x | %d\n" % (gp[0], gp[1]))
        sys.stdout.write("\n")

        sys.stdout.write("   Growth:\n")
        evaluate(maxfitorg, 1)

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
            # Note that these are added in reverse, due to the nature
            # of the linked list in the original C code. The order of
            # gene entries in the chromosome is unimportant. However,
            # the correct split-point needs to be maintained.

            for s in range(parenty.split, len(parenty.chromosome)):
                gcopy = parenty.chromosome[s]
                gp = [gcopy[0], gcopy[1]]
                organism.chromosome.insert(0, gp)

            for s in range(parentx.split):
                gcopy = parentx.chromosome[s]
                gp = [gcopy[0], gcopy[1]]
                organism.chromosome.insert(0, gp)

            organism.split = parentx.split

        # (No explicit free needed in Python; previous generation is
        # garbage-collected.)

        # Copy the new generation into the current generation
        population = nextgen

        # Random mutation at 1%
        for i in range(POPSIZE):
            if random.randrange(100) == 0:
                random_mutate(population[i])

    sys.stdout.write("Done!\n")
    sys.exit(0)


if __name__ == "__main__":
    main()
