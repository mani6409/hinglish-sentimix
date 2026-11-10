import pytest

from src.cmi import cmi


def test_single_language_is_zero():
    assert cmi(["Hin", "Hin", "Hin"]) == 0.0


def test_worked_example_from_notes():
    # 10 tokens, 2 independent, 5 Hin, 3 Eng -> 100 * (1 - 5/8) = 37.5
    langs = ["Hin"] * 5 + ["Eng"] * 3 + ["O", "EMT"]
    assert cmi(langs) == pytest.approx(37.5)


def test_even_split_is_fifty():
    assert cmi(["Hin", "Eng"]) == pytest.approx(50.0)


def test_only_independent_tokens_is_zero():
    assert cmi(["O", "O", "EMT"]) == 0.0


def test_empty_is_zero():
    assert cmi([]) == 0.0


def test_emt_can_be_counted_as_a_language_token():
    # with EMT not independent it counts as its own "language"
    assert cmi(["Hin", "EMT"], independent=("O",)) == pytest.approx(50.0)
