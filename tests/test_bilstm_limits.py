"""Review item I3: length limits must be identical for raw, clean and normalized, and derived from train only."""
import json

import pytest

pytest.importorskip("torch")

import pandas as pd  # noqa: E402

from src.preprocess import PRESETS, load_processed_split, preprocess_frame  # noqa: E402
from src.train_bilstm import (  # noqa: E402
    BiLSTMConfig,
    derive_shared_limits,
    main,
    resolve_limits,
    train_one_seed,
)
from tests.synthetic_conll import SMALL  # noqa: E402

# Hand-made train tweets where the three modes disagree about their OWN limits:
#  - raw keeps the split-up URL (7 extra tokens), so it has the most words (12);
#  - clean/normalized rejoin "wouldn ' t" into "wouldn't" (8 chars) so their words are longer than raw (6).
RAW = ["He wouldn ’ t go https // t . co / abc", "a b c d e f g h i j", "ok"]


def _frame(mode, texts=RAW):
    df = pd.DataFrame({"uid": range(len(texts)), "label": ["positive", "negative", "neutral"][: len(texts)] or [],
                       "text": texts})
    return preprocess_frame(df, mode)


def test_modes_disagree_about_their_own_limits_but_share_one_limit():
    d = derive_shared_limits(RAW)
    per = d["per_mode"]
    assert per["raw"] == {"max_words": 12, "max_chars": 6}
    assert per["clean"] == {"max_words": 10, "max_chars": 8}
    assert per["normalized"] == {"max_words": 10, "max_chars": 8}
    assert (d["max_words"], d["max_chars"]) == (12, 8)  # max over modes, for each limit separately
    assert set(per) == set(PRESETS)


def test_resolved_limits_are_identical_for_every_mode():
    resolved = {m: resolve_limits(_frame(m), BiLSTMConfig(mode=m)) for m in PRESETS}
    vals = {(r["max_words"], r["max_chars"]) for r in resolved.values()}
    assert vals == {(12, 8)}  # one pair for all three modes


def test_limits_do_not_depend_on_row_order():
    a = resolve_limits(_frame("raw", RAW), BiLSTMConfig())
    b = resolve_limits(_frame("raw", RAW[::-1]), BiLSTMConfig())
    assert (a["max_words"], a["max_chars"]) == (b["max_words"], b["max_chars"])


def test_explicit_config_limits_win_and_partial_override_derives_the_other():
    both = resolve_limits(_frame("raw"), BiLSTMConfig(max_words=5, max_chars=4))
    assert (both["max_words"], both["max_chars"], both["source"]) == (5, 4, "config")
    partial = resolve_limits(_frame("raw"), BiLSTMConfig(max_words=5))
    assert (partial["max_words"], partial["max_chars"]) == (5, 8)
    with pytest.raises(ValueError):
        BiLSTMConfig().with_overrides({"max_words": 0})


def test_original_text_column_is_required_to_derive_limits():
    df = _frame("raw").drop(columns=["text"])
    with pytest.raises(ValueError, match="text"):
        resolve_limits(df, BiLSTMConfig())


def test_limits_come_from_train_only_dev_cannot_change_them(data_root):
    tr = load_processed_split("train", mode="normalized", root=data_root)
    dv = load_processed_split("dev", mode="normalized", root=data_root)
    long_dev = dv.copy()
    long_dev.loc[long_dev.index[0], "text_proc"] = "x" * 60 + " " + " ".join(["word"] * 200)  # far longer than any train tweet
    cfg = BiLSTMConfig(**{**SMALL, "epochs": 1})
    a = train_one_seed(tr, dv, cfg, 1, verbose=False)
    b = train_one_seed(tr, long_dev, cfg, 1, verbose=False)
    assert (a["max_words"], a["max_chars"]) == (b["max_words"], b["max_chars"])  # unchanged by dev
    assert a["truncation"]["dev"] == {"tweets": 0, "words": a["truncation"]["dev"]["words"]}
    assert b["truncation"]["dev"]["tweets"] == 1 and b["truncation"]["dev"]["words"] > a["truncation"]["dev"]["words"]
    assert a["truncation"]["train"]["tweets"] == 0  # the limit never cuts a train tweet


def test_training_uses_identical_limits_in_all_modes_while_vocabularies_differ(data_root):
    cfg = {m: BiLSTMConfig(**{**SMALL, "epochs": 1, "mode": m}) for m in PRESETS}
    res = {}
    for m in PRESETS:
        tr = load_processed_split("train", mode=m, root=data_root)
        dv = load_processed_split("dev", mode=m, root=data_root)
        res[m] = train_one_seed(tr, dv, cfg[m], 42, verbose=False)
    assert len({(r["max_words"], r["max_chars"]) for r in res.values()}) == 1  # one limit pair
    assert all(r["truncation"]["train"]["tweets"] == 0 for r in res.values())
    # the vocabularies are mode-specific by design (that is what is being compared)
    assert res["raw"]["word_vocab_size"] > res["normalized"]["word_vocab_size"]  # raw keeps the URL fragments
    # config objects differ only in `mode`
    assert {json.dumps({**vars(c), "mode": None}, sort_keys=True) for c in cfg.values()} == {
        json.dumps({**vars(cfg["raw"]), "mode": None}, sort_keys=True)}


def test_cli_runs_in_separate_processes_per_mode_record_the_same_limits(data_root, tmp_path):
    limits = {}
    for m in PRESETS:  # one CLI call per mode, like the real experiments
        out = tmp_path / f"out_{m}"
        main(["--mode", m, "--seeds", "1", "--out-dir", str(out), "--data-root", str(data_root), "--no-checkpoints",
              *sum((["--set", f"{k}={json.dumps(v)}"] for k, v in {**SMALL, "epochs": 1}.items()), [])])
        saved = json.loads((out / f"bilstm_bilstm_{m}_dev_scores.json").read_text(encoding="utf-8"))
        limits[m] = (saved["limits"]["max_words"], saved["limits"]["max_chars"],
                     saved["per_seed"][0]["max_words"], saved["per_seed"][0]["max_chars"])
    assert len(set(limits.values())) == 1
