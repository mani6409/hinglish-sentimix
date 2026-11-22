"""Tests for src/cmi.py: the Code-Mixing Index of one tweet.

What is covered:
    The formula ``CMI = 100 * (1 - max(w_i) / (n - u))`` on hand-calculated inputs and on the edge cases
    (one language, no language tokens, empty tweet, a different set of language-independent tags).

Why it matters:
    Every CMI bucket in the project is built from this function. A silent change in the formula would move
    tweets between buckets and change the conclusions about code-mixing.

How to run:
    ``python -m pytest tests/test_cmi.py -q``. No data files are needed.
"""
import pytest

from src.cmi import cmi


def test_single_language_is_zero():
    """A tweet whose language tokens are all Hindi has CMI 0.

    Input: three ``Hin`` tags. Expected: 0.0, because max(w) equals n - u (all language tokens belong to the
    most frequent language), so ``1 - 1`` is 0. Prevents a regression where a one-language tweet would get a
    positive mixing score.
    """
    assert cmi(["Hin", "Hin", "Hin"]) == 0.0


def test_worked_example_from_notes():
    """The hand-calculated example also used in the docstring of ``cmi``.

    Input: 10 tags: 5 Hin, 3 Eng, 1 ``O`` and 1 ``EMT``. By hand: n = 10, u = 2 (``O`` and ``EMT``),
    max(w) = 5, so ``100 * (1 - 5/8) = 37.5``. Prevents errors in how the independent tags are removed from
    the denominator.
    """
    # 10 tokens, 2 independent, 5 Hin, 3 Eng -> 100 * (1 - 5/8) = 37.5
    langs = ["Hin"] * 5 + ["Eng"] * 3 + ["O", "EMT"]
    assert cmi(langs) == pytest.approx(37.5)


def test_even_split_is_fifty():
    """One Hindi and one English token give the maximum value for two languages, 50.

    Expected: ``100 * (1 - 1/2) = 50``. Documents the upper bound used when reading CMI values: with the
    project's tags nothing can exceed 50.
    """
    assert cmi(["Hin", "Eng"]) == pytest.approx(50.0)


def test_only_independent_tokens_is_zero():
    """A tweet with only language-independent tags (``O``, ``EMT``) has CMI 0.

    Here n equals u, so there is nothing to compare; the function must return 0.0 and must not divide by
    zero. Prevents a ZeroDivisionError on tweets that consist only of mentions, URLs and symbols.
    """
    assert cmi(["O", "O", "EMT"]) == 0.0


def test_empty_is_zero():
    """An empty tag list gives 0.0 (and no error).

    Prevents a crash on an empty tweet, which would otherwise stop the whole CMI computation.
    """
    assert cmi([]) == 0.0


def test_emt_can_be_counted_as_a_language_token():
    """The set of language-independent tags can be changed with the ``independent`` argument.

    Input: ``["Hin", "EMT"]`` with ``independent=("O",)``. Now ``EMT`` is not independent, so it counts as a
    second "language": n - u = 2, max(w) = 1, CMI = 50. Shows that the argument really changes the result,
    which the experiments on how the paper might have defined CMI relied on.
    """
    # with EMT not independent it counts as its own "language"
    assert cmi(["Hin", "EMT"], independent=("O",)) == pytest.approx(50.0)
