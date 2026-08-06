"""
experiments/concept/common/solution_tree.py
Binary-trie fitness cache used by multi9 and mhier1.

The trie memoises fitness evaluations so that repeated encounters of
the same output bitstream skip the full fitness_function() call.

Usage
-----
    from experiments.concept.common.solution_tree import (
        make_solution, find_solution, free_solution, find_searchspace
    )

    # Caller must supply:
    #   solution  - 2-D list [SIZEX][SIZEY] written to by make_solution()
    #   fitness_function(bitstream) - returns a numeric score
    #   SIZEX, SIZEY, CELLSXY      - grid dimensions
"""
import random


def make_solution(solution, sizex, sizey):
    """Fill *solution* with random binary values (0 or 1)."""
    for x in range(sizex):
        for y in range(sizey):
            solution[x][y] = random.randrange(2)


def find_solution(node, bitstream, bitidx, fitness_function, sizex, sizey, cellsxy):
    """
    Walk/build the trie for *bitstream*, caching fitness at leaves.

    Parameters
    ----------
    node            : dict with keys 'zero', 'one', 'fitness'
    bitstream       : 2-D list [sizex][sizey]
    bitidx          : current flat index (0 ... cellsxy-1)
    fitness_function: callable(bitstream) -> numeric score
    sizex, sizey, cellsxy : grid dimensions

    Returns the cached or freshly computed fitness.
    """
    if bitidx == cellsxy:
        return node["fitness"]

    x   = bitidx // sizey
    y   = bitidx %  sizex
    bit = bitstream[x][y]
    key = "one" if bit else "zero"

    child = node[key]
    if child is None:
        child = {"zero": None, "one": None, "fitness": None}
        node[key] = child
        if bitidx == cellsxy - 1:
            child["fitness"] = fitness_function(bitstream)

    return find_solution(child, bitstream, bitidx + 1,
                         fitness_function, sizex, sizey, cellsxy)


def free_solution(node, bitidx):
    """
    No-op in Python (garbage collection handles it).
    Kept for structural parity with the original C source.
    """
    pass


def find_searchspace(node, bitidx, cellsxy):
    """Count how many distinct bitstreams have been evaluated so far."""
    if node is None:
        return 0
    if bitidx >= cellsxy:
        return 1
    return (find_searchspace(node["zero"], bitidx + 1, cellsxy) +
            find_searchspace(node["one"],  bitidx + 1, cellsxy))


def new_tree():
    """Return a fresh root node for a solution trie."""
    return {"zero": None, "one": None, "fitness": None}
