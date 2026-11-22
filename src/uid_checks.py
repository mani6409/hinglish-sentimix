"""Explicit uid checks so that a merge can never drop, repeat or invent rows silently.

Purpose:
    ``check_uid_alignment`` verifies that a list of tweet ids (uids) contains every reference uid exactly
    once and nothing else.

Why this file exists:
    Predictions are joined to labels, CMI values and tweet text by uid. A plain pandas inner join would drop
    unmatched rows without a word, so a short or misaligned prediction file could give a score computed on
    fewer tweets than we believe. Calling this check first turns such problems into an error.

Place in the pipeline:
    prediction file / results table  ->  check_uid_alignment(...)  ->  merge  ->  metrics

Inputs:
    Two sequences of uids and two names used only in error messages.

Output:
    None. It returns normally when the uids match, and raises ``ValueError`` otherwise.

Used by:
    src/cmi_evaluation.py, src/error_analysis_models.py and tests/test_uid_checks.py.

Depends on:
    The standard library only.

Design decisions:
    * Order does not matter: the check compares sets and counts, not positions, because the project joins by
      uid and never by row position.
    * The checks run in a fixed order (duplicates in the data, duplicates in the reference, missing,
      unexpected) so that the first error message names the most basic problem.
"""
from collections import Counter


def _duplicates(uids):
    """Return the uids that occur more than once, sorted (so error messages are stable).

    Args:
        uids: Iterable of uids (integers in this project; they must be sortable).

    Returns:
        A sorted list of the repeated uids, empty if there are none.
    """
    return sorted(u for u, n in Counter(uids).items() if n > 1)


def check_uid_alignment(uids, reference_uids, name="file", reference_name="reference"):
    """Raise ValueError unless ``uids`` holds each reference uid exactly once and nothing else.

    Purpose:
        Guard a merge: after this call returns, an inner join between the two sources keeps every row.

    How it works:
        Checks, in this order: duplicates in ``uids``, duplicates in the reference, reference uids that are
        missing from ``uids``, and uids that are not in the reference. The first failing check raises, and
        the message shows the number of offending uids and up to five examples.

    Args:
        uids: The uids to check (for example the ``uid`` column of a prediction file).
        reference_uids: The uids that must be present (for example the dev split).
        name: Label for ``uids`` in error messages (for example the file name).
        reference_name: Label for the reference in error messages.

    Returns:
        None.

    Raises:
        ValueError: Any of the four problems above.

    Example:
        check_uid_alignment([3, 1, 2], [1, 2, 3])        # fine: same uids, any order
        check_uid_alignment([1, 1, 2], [1, 2])           # ValueError: file: 1 duplicate uids, e.g. [1]
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
