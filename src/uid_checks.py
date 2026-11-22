"""Explicit uid checks so that a merge can never drop, repeat or invent rows silently."""
from collections import Counter


def _duplicates(uids):
    return sorted(u for u, n in Counter(uids).items() if n > 1)


def check_uid_alignment(uids, reference_uids, name="file", reference_name="reference"):
    """Raise ValueError unless `uids` holds each reference uid exactly once and nothing else.

    Checks, in this order: duplicates in `uids`, duplicates in the reference, reference uids that are
    missing from `uids`, and uids that are not in the reference.
    """
    uids, reference_uids = list(uids), list(reference_uids)
    dup = _duplicates(uids)
    if dup:
        raise ValueError(f"{name}: {len(dup)} duplicate uids, e.g. {dup[:5]}")
    dup_ref = _duplicates(reference_uids)
    if dup_ref:
        raise ValueError(f"{reference_name}: {len(dup_ref)} duplicate uids, e.g. {dup_ref[:5]}")
    missing = set(reference_uids) - set(uids)
    if missing:
        raise ValueError(f"{name}: {len(missing)} uids from {reference_name} are missing, e.g. {sorted(missing)[:5]}")
    unexpected = set(uids) - set(reference_uids)
    if unexpected:
        raise ValueError(f"{name}: {len(unexpected)} uids are not in {reference_name}, e.g. {sorted(unexpected)[:5]}")
