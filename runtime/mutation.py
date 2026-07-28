"""Shared mutation scheduling for temporal evolution backends."""
from __future__ import annotations


STRESS_PATIENCE = 12
DEFAULT_MUTATION_LIMIT = 8.0
# Back-compatible public name used by the backend modules.
STRESS_MAX_MULT = DEFAULT_MUTATION_LIMIT
DEFAULT_STAGNATION_BETA = 1.0


def adaptive_mutation_rate(annealed_rate, stagnation, solved=False,
                           beta=DEFAULT_STAGNATION_BETA,
                           mutation_limit=DEFAULT_MUTATION_LIMIT):
    """Return the effective mutation rate after plateau reheating.

    ``beta`` is the user-facing plateau response. Zero disables reheating, one
    preserves the original schedule, and larger values increase the rate more
    quickly after ``STRESS_PATIENCE`` flat generations. ``mutation_limit`` is
    the hard upper bound on the resulting mutations-per-child rate. Solved runs
    stay on the capped annealed rate because no improvement remains to find.
    """
    beta = float(beta)
    mutation_limit = float(mutation_limit)
    if beta < 0:
        raise ValueError('stagnation beta must be non-negative')
    if mutation_limit < 1:
        raise ValueError('mutation limit must be at least 1')
    # Mutations are constructive and always perform at least one event, so 1 is
    # the meaningful floor. The user limit is a true cap on both the initial
    # rate and plateau reheating, not merely on the latter.
    base = min(mutation_limit, max(1.0, float(annealed_rate)))
    if solved or beta == 0.0 or stagnation < STRESS_PATIENCE:
        return base
    plateau_age = ((stagnation - STRESS_PATIENCE)
                   / max(1.0, STRESS_PATIENCE / 2.0))
    ramp = 1.0 + beta * plateau_age
    return min(mutation_limit, max(base, ramp))
