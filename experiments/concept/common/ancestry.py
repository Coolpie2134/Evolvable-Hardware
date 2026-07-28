"""
experiments/concept/common/ancestry.py
Two-generation ancestry check used by coderack7 and hierarch4.

Organisms carry a tag[] list where tag[1..6] hold parent/grandparent IDs.
quadcheck() tests whether two (id, id) pairs are the same unordered pair.
ancestors() uses quadcheck to decide whether two organisms share recent ancestry.
"""


def quadcheck(a1, b1, a2, b2):
    """Return True if (a1,b1) and (a2,b2) are the same unordered pair."""
    if a1 == a2 and b1 == b2:
        return True
    if a1 == b2 and b1 == a2:
        return True
    return False


def ancestors(org1, org2):
    """
    Return True if org1 and org2 share any direct ancestor within
    two generations (checked via their tag[] arrays).
    """
    if quadcheck(org1.tag[1], org1.tag[2], org2.tag[1], org2.tag[2]):
        return True
    if quadcheck(org1.tag[3], org1.tag[4], org2.tag[3], org2.tag[4]):
        return True
    if quadcheck(org1.tag[5], org1.tag[6], org2.tag[5], org2.tag[6]):
        return True
    if quadcheck(org1.tag[3], org1.tag[4], org2.tag[5], org2.tag[6]):
        return True
    if quadcheck(org1.tag[5], org1.tag[6], org2.tag[3], org2.tag[4]):
        return True
    return False
