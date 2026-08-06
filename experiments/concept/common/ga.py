"""
experiments/concept/common/ga.py
Fitness-proportional (roulette-wheel) GA selection loop shared by all
2-D concept simulations (multi9, mhier1, coderack7, hierarch4).

Each simulation calls do_one_generation() and make_next_generation(),
supplying its own evaluate() and random_mutate() callables so that the
selection mechanics stay completely net-type-agnostic.

GEN_* rebuild flags are also kept here since every simulation that uses
this module shares them.
"""
import random
import functools

# -- Rebuild flags (which parts of state to regenerate between runs) --
GEN_POP   = 1    # regenerate population
GEN_START = 2    # regenerate starting (seed) state
GEN_CHEM  = 4    # regenerate chemistry map
GEN_SOLN  = 8    # regenerate target solution
GEN_ALL   = GEN_POP | GEN_START | GEN_CHEM | GEN_SOLN


def _tagcomp(a, b):
    """Descending comparator for tag-sorted ranking."""
    return -1 if a > b else 1


def do_one_generation(population, evaluate_fn, fitness_scale,
                      ancestor_fn=None, mutate_fn=None):
    """
    One generation of fitness-proportional selection.

    Parameters
    ----------
    population    : list of Individual
    evaluate_fn   : callable(organism) - sets organism.fitness in place
    fitness_scale : int  - fitness is raised to this power for selection
    ancestor_fn   : optional callable(org1, org2) -> bool
                    if provided, used to avoid incestuous pairings
    mutate_fn     : optional callable(org) - used as fallback when
                    ancestor avoidance cannot find a valid pair

    Returns
    -------
    (pairings_x, pairings_y) : two lists of length POPSIZE; each
    paired index (x, y) is a mating pair for make_next_generation.
    """
    popsize = len(population)

    # -- evaluate --
    totalfit = 0
    for org in population:
        evaluate_fn(org)
        org.scaled = org.fitness
        for _ in range(fitness_scale - 1):
            org.scaled *= org.fitness
        totalfit += org.scaled

    if totalfit == 0:
        # Degenerate case: all zero fitness; pair randomly
        pairs = list(range(popsize))
        random.shuffle(pairs)
        px = [population[i] for i in pairs]
        py = [population[(i + 1) % popsize] for i in pairs]
        return px, py

    # -- fitness-proportional mating slot allocation --
    pairings_x = [None] * popsize
    pairings_y = [None] * popsize
    residual   = 0
    binsleft   = popsize

    def _no_incest(a, b):
        return ancestor_fn is None or not ancestor_fn(a, b)

    for org in population:
        fitshare = org.scaled * popsize * 2
        residual += fitshare % totalfit
        matings   = fitshare // totalfit
        if residual > 0:
            matings  += 1
            residual -= totalfit

        tries = 0
        s     = 0
        while s < matings:
            x = random.randrange(binsleft)
            if pairings_x[x] is None:
                pairings_x[x] = org
            else:
                if not _no_incest(pairings_x[x], org):
                    tries += 1
                    if tries < 10:
                        s -= 1
                    else:
                        # Fall back: scan for a valid slot
                        found = False
                        for y in range(binsleft):
                            if pairings_x[y] is None:
                                pairings_x[y] = org
                                tries = 0
                                found = True
                                x = y
                                break
                            elif _no_incest(pairings_x[y], org):
                                x = y
                                tries = 0
                                found = True
                                break
                        if not found and mutate_fn is not None:
                            # Last resort: mutate to break ancestry
                            for y3 in range(popsize):
                                if pairings_x[y3] is not org:
                                    while not _no_incest(pairings_x[y3], org):
                                        mutate_fn(pairings_x[y3])
                                    break

                if _no_incest(pairings_x[x], org):
                    save_x = pairings_x[binsleft - 1]
                    save_y = pairings_y[binsleft - 1]
                    pairings_x[binsleft - 1] = pairings_x[x]
                    pairings_y[binsleft - 1] = org
                    pairings_x[x] = save_x
                    pairings_y[x] = save_y
                    binsleft -= 1
                    if binsleft == 0:
                        break
            s += 1
        if binsleft == 0:
            break

    return pairings_x, pairings_y


def sort_by_tag(population):
    """Sort population descending by tag[0] (generation tag) in place."""
    population.sort(key=functools.cmp_to_key(
        lambda a, b: _tagcomp(a.tag[0] if hasattr(a, 'tag') else 0,
                               b.tag[0] if hasattr(b, 'tag') else 0)))
