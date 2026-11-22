"""Tests for src/data.py (the CoNLL parser and ``load_split``) and src/seed.py (``set_seed``).

What is covered:
    * ``parse_conll`` on small synthetic files: normal tweets, a file with no blank line at the end, headers
      without a label, and a malformed token line.
    * ``set_seed`` makes ``random`` and NumPy repeat their numbers.
    * One test on the real data files: split sizes, class counts, label completeness of the test split,
      unique uids and aligned token/tag lists. It is skipped when the data folder is missing.

Why it matters:
    All experiments start from ``load_split``. If the parser dropped or merged a tweet, or the seed did not
    fix the randomness, every later result would be affected.

How to run:
    ``python -m pytest tests/test_data_seed.py -q``. The real-data test needs data/raw/... (it is part of
    the repository). Note that this test loads the test split together with its labels to check counts; this
    descriptive access is disclosed in docs/bilstm_final_evaluation_protocol.md.
"""
import random

import numpy as np
import pytest

from src.data import DEFAULT_ROOT, load_split, parse_conll
from src.seed import set_seed

# Two tweets. The second one deliberately has no blank line after it, to check that the last tweet is still
# stored when the file ends.
SAMPLE = (
    "meta\t1\tpositive\nyaar\tHin\nso\tEng\ngood\tEng\n\n"
    "meta\t2\tnegative\n@\tO\nbura\tHin\n"  # no trailing blank line
)


def test_parse_small_file(tmp_path):
    """A small file is parsed into the right tweets, tokens, tags and labels.

    Input: ``SAMPLE`` (two tweets, the last without a trailing blank line). Expected: uids [1, 2]; the first
    tweet has tokens yaar/so/good with tags Hin/Eng/Eng; the second has the label ``negative``. Prevents the
    last tweet from being lost and the tag lists from getting out of step with the tokens.
    """
    p = tmp_path / "s.txt"
    p.write_text(SAMPLE, encoding="utf-8")
    tw = parse_conll(p)
    assert [t["uid"] for t in tw] == [1, 2]
    assert tw[0]["tokens"] == ["yaar", "so", "good"]
    assert tw[0]["langs"] == ["Hin", "Eng", "Eng"]
    assert tw[1]["label"] == "negative"


def test_unlabelled_header_gives_none(tmp_path):
    """A header without a sentiment field (as in the test file) gives ``label = None``.

    Input: ``meta<TAB>7`` followed by one token. Expected: label None. Prevents the empty field from
    becoming an empty string or an error; ``load_split`` relies on None for the unlabelled test file.
    """
    p = tmp_path / "u.txt"
    p.write_text("meta\t7\nword\tEng\n", encoding="utf-8")
    assert parse_conll(p)[0]["label"] is None


def test_malformed_line_raises(tmp_path):
    """A token line without a TAB raises ValueError.

    Input: a valid header followed by the line ``notabhere``. Expected: ValueError. Prevents a damaged data
    file from being read silently with a wrong tag.
    """
    p = tmp_path / "bad.txt"
    p.write_text("meta\t1\tpositive\nnotabhere\n", encoding="utf-8")
    with pytest.raises(ValueError):
        parse_conll(p)


def test_seed_repeatable():
    """The same seed gives the same random numbers from Python's ``random`` and from NumPy.

    Input: seed 123 set twice, one draw from each generator after each call. Expected: identical pairs.
    Prevents ``set_seed`` from forgetting one of the two generators, which would make reruns differ.
    """
    set_seed(123)
    a = (random.random(), np.random.rand())
    set_seed(123)
    b = (random.random(), np.random.rand())
    assert a == b


# Marker for tests that need the real data files; they are skipped when the data folder is absent.
real = pytest.mark.skipif(not DEFAULT_ROOT.exists(), reason="data not downloaded")


@real
def test_real_split_sizes_and_labels():
    """The real data files have the expected sizes, class counts and label completeness.

    Expected values: 14,000 / 3,000 / 3,000 tweets; train labels neutral 5,264, positive 4,634, negative
    4,102; dev labels neutral 1,128, positive 982, negative 890; every test tweet has a label from the
    separate labels file and only valid label names; uids unique in each split; every tweet has as many
    language tags as tokens (checked for the train split only). They correspond to the percentages in docs/eda_part1.md. Prevents a changed or corrupted data
    file, or a parser change, from going unnoticed.
    """
    tr, dv, te = load_split("train"), load_split("dev"), load_split("test")
    assert (len(tr), len(dv), len(te)) == (14000, 3000, 3000)
    assert tr["label"].value_counts().to_dict() == {"neutral": 5264, "positive": 4634, "negative": 4102}
    assert dv["label"].value_counts().to_dict() == {"neutral": 1128, "positive": 982, "negative": 890}
    assert te["label"].notna().all()  # filled from the separate labels file
    assert set(te["label"]) <= {"positive", "negative", "neutral"}
    assert tr["uid"].is_unique and dv["uid"].is_unique and te["uid"].is_unique
    assert (tr["tokens"].apply(len) == tr["langs"].apply(len)).all()
