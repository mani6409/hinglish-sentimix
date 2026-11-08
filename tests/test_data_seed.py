import random

import numpy as np
import pytest

from src.data import DEFAULT_ROOT, load_split, parse_conll
from src.seed import set_seed

SAMPLE = (
    "meta\t1\tpositive\nyaar\tHin\nso\tEng\ngood\tEng\n\n"
    "meta\t2\tnegative\n@\tO\nbura\tHin\n"  # no trailing blank line
)


def test_parse_small_file(tmp_path):
    p = tmp_path / "s.txt"
    p.write_text(SAMPLE, encoding="utf-8")
    tw = parse_conll(p)
    assert [t["uid"] for t in tw] == [1, 2]
    assert tw[0]["tokens"] == ["yaar", "so", "good"]
    assert tw[0]["langs"] == ["Hin", "Eng", "Eng"]
    assert tw[1]["label"] == "negative"


def test_unlabelled_header_gives_none(tmp_path):
    p = tmp_path / "u.txt"
    p.write_text("meta\t7\nword\tEng\n", encoding="utf-8")
    assert parse_conll(p)[0]["label"] is None


def test_malformed_line_raises(tmp_path):
    p = tmp_path / "bad.txt"
    p.write_text("meta\t1\tpositive\nnotabhere\n", encoding="utf-8")
    with pytest.raises(ValueError):
        parse_conll(p)


def test_seed_repeatable():
    set_seed(123)
    a = (random.random(), np.random.rand())
    set_seed(123)
    b = (random.random(), np.random.rand())
    assert a == b


real = pytest.mark.skipif(not DEFAULT_ROOT.exists(), reason="data not downloaded")


@real
def test_real_split_sizes_and_labels():
    tr, dv, te = load_split("train"), load_split("dev"), load_split("test")
    assert (len(tr), len(dv), len(te)) == (14000, 3000, 3000)
    assert tr["label"].value_counts().to_dict() == {"neutral": 5264, "positive": 4634, "negative": 4102}
    assert dv["label"].value_counts().to_dict() == {"neutral": 1128, "positive": 982, "negative": 890}
    assert te["label"].notna().all()  # filled from the separate labels file
    assert set(te["label"]) <= {"positive", "negative", "neutral"}
    assert tr["uid"].is_unique and dv["uid"].is_unique and te["uid"].is_unique
    assert (tr["tokens"].apply(len) == tr["langs"].apply(len)).all()
