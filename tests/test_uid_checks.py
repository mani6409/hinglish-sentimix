"""Tests for src/uid_checks.py: the guard that makes uid merges safe.

What is covered:
    ``check_uid_alignment`` accepts matching uids in any order and raises ``ValueError`` for duplicates (in
    either list), missing uids and unexpected uids.

Why it matters:
    Predictions are joined to labels and CMI values by uid. Without the guard a short or misaligned prediction
    file would be scored on fewer tweets than we believe.

How to run:
    ``python -m pytest tests/test_uid_checks.py -q``. No data files are needed.
"""
import pytest

from src.uid_checks import check_uid_alignment


def test_matching_uids_pass_in_any_order():
    """The same uids in a different order are accepted (no error).

    Input: [3, 1, 2] against the reference [1, 2, 3]. The project joins by uid and never by position, so order
    must not matter. Prevents a regression where a harmless re-ordering would raise.
    """
    check_uid_alignment([3, 1, 2], [1, 2, 3])


@pytest.mark.parametrize("uids, ref, text", [
    ([1, 1, 2], [1, 2], "duplicate"),
    ([1, 2], [1, 2, 2], "duplicate"),
    ([1], [1, 2], "missing"),
    ([1, 2, 3], [1, 2], "not in"),
])
def test_bad_alignment_raises(uids, ref, text):
    """Each kind of mismatch raises ValueError with a message that names the problem.

    Cases: a repeated uid in the data, a repeated uid in the reference, a reference uid that the data lacks
    ("missing"), and a data uid that the reference lacks ("not in"). The ``match`` text checks that the
    message tells the reader which problem occurred. Prevents silent row loss or duplication in merges.
    """
    with pytest.raises(ValueError, match=text):
        check_uid_alignment(uids, ref, name="a", reference_name="b")
