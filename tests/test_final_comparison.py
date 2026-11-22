"""Tests for src.final_comparison: rows come from the result files, missing test evaluations are marked, never invented."""
import json

import numpy as np
import pandas as pd
import pytest

from src.evaluate_bilstm_test import SCALARS
from src.final_comparison import build
from src.metrics import LABELS, compute_metrics


def _pred(path, n=30, shift=0):
    rng = np.random.default_rng(shift)
    labels = [LABELS[i % 3] for i in range(n)]
    preds = [lab if rng.random() < 0.7 else LABELS[(LABELS.index(lab) + 1) % 3] for lab in labels]
    pd.DataFrame({"uid": range(n), "label": labels, "pred": preds}).to_csv(path, index=False)
    return compute_metrics(labels, preds)["weighted_f1"]


def _summary(values):
    return {k: {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)), "values": values} for k in SCALARS}


@pytest.fixture()
def results(tmp_path):
    r = tmp_path / "results"
    (r / "predictions").mkdir(parents=True)
    json.dump({"majority_baseline": {"label": "neutral", "weighted_f1": 0.2, "accuracy": 0.37}, "char_2-5": {"best_dev": {}}},
              open(r / "tfidf_dev_scores.json", "w"))
    _pred(r / "predictions" / "tfidf_char_2-5_dev.csv")
    wf1 = [_pred(r / "predictions" / f"bilstm_full_clean_s{s}_seed{s}_dev.csv", shift=s) for s in (42, 43, 44)]
    (r / "bilstm_final_test").mkdir()
    json.dump({"clean": _summary([0.66, 0.67, 0.68])}, open(r / "bilstm_final_test" / "metrics_summary.json", "w"))
    rows = []
    for s, run in ((42, "xlmr_clean_ep6_lr1e-5"), (43, "xlmr_clean_ep6_lr1e-5_final"), (44, "xlmr_clean_ep6_lr1e-5_final")):
        _pred(r / "predictions" / f"transformer_{run}_seed{s}_dev.csv", shift=s + 10)
        rows.append({"row_type": "seed", "model": "xlmr", "lr": 1e-5, "seeds": s, "notes": f"run={run}; best_epoch=3"})
    pd.DataFrame(rows).to_csv(r / "transformer_comparison.csv", index=False)
    return r, wf1


def test_rows_come_from_files_and_missing_test_runs_are_pending(results):
    r, wf1 = results
    df = build(r)
    assert (r / "final_model_comparison.csv").exists()
    bil = df[(df.family == "BiLSTM") & (df.variant == "clean")].set_index("split")
    assert bil.loc["dev", "weighted_f1_mean"] == pytest.approx(np.mean(wf1), abs=1e-4)
    assert bil.loc["dev", "weighted_f1_sd"] == pytest.approx(np.std(wf1, ddof=1), abs=1e-4)
    assert bil.loc["test", "weighted_f1_mean"] == pytest.approx(0.67) and bil.loc["test", "n_runs"] == 3
    x = df[df.system.str.startswith("XLM-R")].set_index("split")
    assert x.loc["dev", "status"] == "done" and x.loc["dev", "n_runs"] == 3
    assert x.loc["test", "status"] == "pending" and np.isnan(x.loc["test", "weighted_f1_mean"])  # no number is invented
    assert df[df.system == "TF-IDF + logistic regression"].status.item() == "not run"


def test_transformer_test_row_is_read_from_the_frozen_evaluation(results):
    r, _ = results
    (r / "transformer_final_test").mkdir()
    json.dump({"xlmr": _summary([0.70, 0.71, 0.72])}, open(r / "transformer_final_test" / "metrics_summary.json", "w"))
    x = build(r, write=False).query("system.str.startswith('XLM-R')", engine="python").set_index("split")
    assert x.loc["test", "status"] == "done" and x.loc["test", "weighted_f1_mean"] == pytest.approx(0.71)
