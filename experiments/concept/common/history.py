"""
concept/common/history.py
Generational history tracking for ancestry-based incest prevention.
Used by multi9 and mhier1.

Each organism carries a history[] list of length (2^(HIST_LIMIT+1) - 1).
Slot 0 holds the organism's own tag; slots 1..N hold parent/ancestor tags
in a binary-tree layout.

hist_compare() returns True if two organisms share a recent ancestor
within HIST_LIMIT generations.  update_history() fills the child's
history from its two parents after mating.
"""

HIST_LIMIT = 3                           # generations of ancestry to track
HISTORY    = (1 << (HIST_LIMIT + 1)) - 1  # = 15 slots


def hist_compare(org1, org2):
    """
    Return True if org1 and org2 share any direct ancestor within
    HIST_LIMIT generations.  history[0] == 0 signals the first
    generation and is never counted as a match.
    """
    for k in range(1, HIST_LIMIT):
        l = (1 << k) - 1
        m = (1 << (k + 1)) - 1
        for j in range(l, m):
            if org1.history[k] == org2.history[j] and org1.history[k] != 0:
                return True
    return False


def update_history(org, px, py):
    """
    Build the child's history from parent px and parent py.
    Call this immediately after creating a new offspring.
    """
    org.history[1] = px.history[0]
    org.history[2] = py.history[0]

    for k in range(1, HIST_LIMIT):
        n = (1 << (k - 1)) - 1   # (0, 1, 3)
        l = (1 << k) - 1          # (1, 3, 7)
        for j in range(n + 1):
            org.history[l + j]         = px.history[n + j]
            org.history[l + n + 1 + j] = py.history[n + j]
