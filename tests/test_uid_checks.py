import pytest

from src.uid_checks import check_uid_alignment


def test_matching_uids_pass_in_any_order():
    check_uid_alignment([3, 1, 2], [1, 2, 3])


@pytest.mark.parametrize("uids, ref, text", [
    ([1, 1, 2], [1, 2], "duplicate"),
    ([1, 2], [1, 2, 2], "duplicate"),
    ([1], [1, 2], "missing"),
    ([1, 2, 3], [1, 2], "not in"),
])
def test_bad_alignment_raises(uids, ref, text):
    with pytest.raises(ValueError, match=text):
        check_uid_alignment(uids, ref, name="a", reference_name="b")
