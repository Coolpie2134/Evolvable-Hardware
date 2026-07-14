"""Structural limits shared by every evolution backend."""
from __future__ import annotations

# Keep a finite ceiling because chromosome count multiplies population memory,
# crossover work, and genome visualisation cost.  The old ceiling of six was
# unnecessarily restrictive for experiments, while 32 leaves ample room for
# larger genomes without making a typo such as 1000 silently allocate them.
MAX_CHROMOSOME_COUNT = 32
