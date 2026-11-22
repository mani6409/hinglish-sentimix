"""Tests for the issue #6 driver (src.run_transformer_experiments) and comparison (src.transformer_comparison).

The whole protocol (lr search, selection, final seeds, comparison table, representative predictions,
paired bootstrap) runs end to end on the synthetic data with two copies of the tiny local BERT.
"""
import csv
import json
import shutil

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from src.metrics import LABELS, compute_metrics  # noqa: E402
from src.run_transformer_experiments import (  # noqa: E402
    final_run_name,
    lr_tag,
    main as run_main,
    search_run_name,
)
from src.train_transformer import checkpoint_dir, prediction_path, scores_path  # noqa: E402
from src.transformer_comparison import (  # noqa: E402
    build,
    full_metrics,
    median_run,
    paired_bootstrap,
)
from tests.test_transformer_pipeline import tiny_model  # noqa: E402,F401  (fixture)

LRS = ["1e-3", "3e-3"]


def test_lr_tag_and_run_names():
    assert [lr_tag(x) for x in (1e-5, 2e-5, 3e-5, 5e-5, 1e-3)] == ["1e-5", "2e-5", "3e-5", "5e-5", "1e-3"]
    assert search_run_name("xlmr", "clean", 2e-5) == "xlmr_clean_lr2e-5"
    assert final_run_name("mbert", "clean", 3e-5) == "mbert_clean_lr3e-5_final"
    assert search_run_name("xlmr", "clean", 2e-5, "_ep6") == "xlmr_clean_ep6_lr2e-5"
    assert final_run_name("mbert", "clean", 3e-5, "_ep6") == "mbert_clean_ep6_lr3e-5_final"


def test_full_metrics_adds_precision_and_recall():
    y = ["negative", "neutral", "positive", "neutral"]
    p = ["negative", "positive", "positive", "neutral"]
    m = full_metrics(y, p)
    assert m["weighted_f1"] == compute_metrics(y, p)["weighted_f1"]
    assert m["recall_weighted"] == pytest.approx(m["accuracy"])  # weighted recall equals accuracy
    assert m["precision_macro"] == pytest.approx((1 + 1 + 0.5) / 3)
    assert m["recall_macro"] == pytest.approx((1 + 0.5 + 1) / 3)


def test_median_run_picks_middle_seed():
    runs = [{"seed": s, "metrics": {"weighted_f1": f}} for s, f in ((42, 0.70), (43, 0.60), (44, 0.65))]
    assert median_run(runs)["seed"] == 44


def test_paired_bootstrap_is_zero_for_identical_predictions_and_reproducible():
    rng = np.random.default_rng(0)
    y = list(rng.choice(LABELS, 200))
    p = list(rng.choice(LABELS, 200))
    assert np.all(paired_bootstrap(y, p, p, np.random.default_rng(1), n=50) == 0)
    a = paired_bootstrap(y, y, p, np.random.default_rng(1), n=50)
    b = paired_bootstrap(y, y, p, np.random.default_rng(1), n=50)
    assert np.array_equal(a, b) and np.all(a > 0)  # perfect predictions always beat random ones


@pytest.fixture()
def two_models(tiny_model, tmp_path):
    a, b = tmp_path / "tinya", tmp_path / "tinyb"
    shutil.copytree(tiny_model, a)
    shutil.copytree(tiny_model, b)
    return str(a), str(b)


@pytest.fixture()
def small_config(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"epochs": 4, "patience": 4, "batch_size": 16, "warmup_ratio": 0.0,
                                "device": "cpu", "fp16": False}), encoding="utf-8")
    return str(path)


def test_protocol_end_to_end(two_models, small_config, data_root, tmp_path):
    out = tmp_path / "results"
    argv = ["--models", *two_models, "--lrs", *LRS, "--search-seed", "1", "--final-seeds", "2", "3",
            "--config", small_config, "--out-dir", str(out), "--prune-checkpoints", "--", "--data-root", str(data_root)]
    best = run_main(argv)

    # search table: one row per model and lr, exactly one selected per model, the best one
    rows = list(csv.DictReader(open(out / "transformer_lr_search.csv", encoding="utf-8", newline="")))
    assert len(rows) == 4
    for slug in ("tinya", "tinyb"):
        own = [r for r in rows if r["model"] == slug]
        sel = [r for r in own if r["selected"] == "True"]
        assert len(sel) == 1
        assert float(sel[0]["dev_weighted_f1"]) == max(float(r["dev_weighted_f1"]) for r in own)

    # final seeds ran at the selected lr; unselected search checkpoints were pruned, selected ones kept
    for m in two_models:
        lr = best[m]["lr"]
        assert scores_path(out, final_run_name(m, "clean", lr)).exists()
        assert checkpoint_dir(out, search_run_name(m, "clean", lr), 1).is_dir()
        for other in LRS:
            if float(other) != lr:
                assert not checkpoint_dir(out, search_run_name(m, "clean", float(other)), 1).exists()

    # restart skips finished runs (nothing retrained, nothing overwritten)
    before = {p: p.stat().st_mtime_ns for p in out.rglob("*.csv") if "predictions" in p.parts}
    run_main(argv)
    assert before == {p: p.stat().st_mtime_ns for p in out.rglob("*.csv") if "predictions" in p.parts}

    res = build(list(two_models), "clean", out)
    comp = pd.read_csv(out / "transformer_comparison.csv")
    means = comp[comp.row_type == "mean"]
    assert list(means.model) == ["tinya", "tinyb"]
    assert (means.n_seeds == 3).all() and (means.seeds == "1 2 3").all()
    assert (comp[comp.row_type == "seed"].groupby("model").size() == 3).all()
    for slug, runs in res["runs"].items():
        rep = pd.read_csv(out / f"{slug}_dev_predictions.csv")
        assert list(rep.columns) == ["uid", "label", "pred"]
        assert rep.equals(pd.read_csv(median_run(runs)["pred_file"]))
        wf1 = [r["metrics"]["weighted_f1"] for r in runs]
        row = means[means.model == slug].iloc[0]
        assert row.weighted_f1_mean == pytest.approx(np.mean(wf1), abs=1e-4)
        assert row.weighted_f1_sd == pytest.approx(np.std(wf1, ddof=1), abs=1e-4)
        assert row.weighted_f1_mean_unselected_seeds == pytest.approx(np.mean(wf1[1:]), abs=1e-4)
    pairs = pd.read_csv(out / "transformer_pairwise.csv")
    assert list(pairs.basis[:3]) == ["seed 1", "seed 2", "seed 3"]
    boot = pairs.iloc[3]
    assert boot.ci_low <= boot.difference <= boot.ci_high


def test_suffix_and_global_set_keep_protocols_apart(two_models, small_config, data_root, tmp_path):
    out = tmp_path / "results"
    common = ["--models", *two_models, "--lrs", "1e-3", "--search-seed", "1", "--final-seeds", "2",
              "--config", small_config, "--out-dir", str(out)]
    run_main(common + ["--", "--data-root", str(data_root)])
    run_main(common + ["--suffix", "_ep2", "--set", "epochs=2", "--set", "patience=2", "--",
                       "--data-root", str(data_root)])
    assert (out / "transformer_lr_search.csv").exists() and (out / "transformer_lr_search_ep2.csv").exists()
    for m in two_models:
        base = json.loads(scores_path(out, final_run_name(m, "clean", 1e-3)).read_text(encoding="utf-8"))
        ep2 = json.loads(scores_path(out, final_run_name(m, "clean", 1e-3, "_ep2")).read_text(encoding="utf-8"))
        assert base["config"]["epochs"] == 4 and ep2["config"]["epochs"] == 2
        assert ep2["per_seed"][0]["epochs_run"] <= 2
    res = build(list(two_models), "clean", out, suffix="_ep2")
    assert all(r["run_name"].endswith(("_ep2_lr1e-3", "_ep2_lr1e-3_final")) for rs in res["runs"].values() for r in rs)


def test_comparison_rejects_tampered_predictions(two_models, small_config, data_root, tmp_path):
    out = tmp_path / "results"
    run_main(["--models", *two_models, "--lrs", "1e-3", "--search-seed", "1", "--final-seeds", "2",
              "--config", small_config, "--out-dir", str(out), "--", "--data-root", str(data_root)])
    path = prediction_path(out, final_run_name(two_models[0], "clean", 1e-3), 2)
    df = pd.read_csv(path)
    df.loc[0, "pred"] = next(x for x in LABELS if x != df.loc[0, "pred"])
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="differs"):
        build(list(two_models), "clean", out, write=False)
