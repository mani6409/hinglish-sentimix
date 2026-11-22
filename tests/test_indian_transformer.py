"""Tests for src/indian_transformer.py with a tiny random BERT and a hand-made vocabulary (no download)."""
import json

import pandas as pd
import pytest
import torch
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

from src import indian_transformer as it
from src.metrics import LABELS

WORDS = "good bad ok movie bakwas accha nahi hai yeh film".split()


@pytest.fixture()
def tiny(tmp_path):
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + WORDS
    vf = tmp_path / "vocab.txt"
    vf.write_text("\n".join(vocab), encoding="utf-8")
    tok = BertTokenizerFast(str(vf))
    cfg = BertConfig(vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                     intermediate_size=32, num_labels=3, max_position_embeddings=32)
    torch.manual_seed(0)
    return tok, BertForSequenceClassification(cfg)


def frame(n):
    rows = [("good movie", "positive"), ("bad bakwas film", "negative"), ("yeh film hai", "neutral")]
    return pd.DataFrame({"uid": range(n), "label": [rows[i % 3][1] for i in range(n)],
                         "text_proc": [rows[i % 3][0] for i in range(n)]})


def test_test_split_is_refused():
    with pytest.raises(ValueError, match="test set"):
        it.load_texts("test")


def test_label_order_matches_project_metrics():
    assert [it.ID2LABEL[i] for i in range(3)] == LABELS


def test_token_length_report_counts_unknown_words(tiny):
    tok, _ = tiny
    r = it.token_length_report(tok, ["good movie", "zzzz qqqq"])
    assert r["n_texts"] == 2 and r["length_percentiles"]["p100"] == 4  # [CLS] w w [SEP]
    assert r["unk_token_share"] == pytest.approx(2 / 8)


def test_fine_tune_runs_and_returns_valid_predictions(tiny, tmp_path):
    tok, model = tiny
    train, dev = frame(30), frame(9)
    res = it.fine_tune("tiny", train, dev, seed=1, max_len=8, batch_size=8, lr=5e-3, epochs=2, device="cpu",
                       tokenizer=tok, model=model, checkpoint_dir=tmp_path / "ck", log=lambda *_: None)
    assert len(res["pred"]) == len(dev) and set(res["pred"]) <= set(LABELS)
    assert len(res["history"]) == 2 and res["best_epoch"] in (1, 2)
    assert (tmp_path / "ck" / "config.json").exists()          # checkpoint saved
    assert 0.0 <= res["metrics"]["weighted_f1"] <= 1.0


def test_fine_tune_is_reproducible_for_same_seed(tiny):
    tok, _ = tiny

    def run():
        cfg = BertConfig(vocab_size=len(tok), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                         intermediate_size=32, num_labels=3, max_position_embeddings=32)
        torch.manual_seed(0)
        return it.fine_tune("tiny", frame(30), frame(9), seed=5, max_len=8, batch_size=8, lr=5e-3, epochs=1,
                            device="cpu", tokenizer=tok, model=BertForSequenceClassification(cfg), log=lambda *_: None)

    a, b = run(), run()
    assert a["history"] == b["history"] and a["pred"] == b["pred"]


def test_prediction_paths_and_saved_csv_format(tmp_path, monkeypatch):
    monkeypatch.setattr(it, "RESULTS", tmp_path)
    assert it.prediction_path("muril", 42) == tmp_path / "muril_dev_predictions.csv"
    assert it.prediction_path("muril", 43) == tmp_path / "predictions" / "muril_s43_seed43_dev.csv"
    dev = frame(3)
    result = {"pred": ["positive", "negative", "neutral"], "best_epoch": 1, "metrics": {"weighted_f1": 1.0},
              "history": [], "device": "cpu", "train_seconds": 0.1}
    path, scores = it.save_run("muril", "google/muril-base-cased", dev, result, 42, {"lr": 2e-5})
    df = pd.read_csv(path)
    assert list(df.columns) == ["uid", "label", "pred"] and len(df) == 3
    assert json.loads(scores.read_text())["model"] == "google/muril-base-cased"
