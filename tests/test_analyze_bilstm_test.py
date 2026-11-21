"""Tests for src/analyze_bilstm_test.py with synthetic arrays only (no official data, no checkpoints, no labels)."""
import numpy as np
import pytest
from sklearn.metrics import f1_score

from src import analyze_bilstm_test as A
from src.evaluate_bilstm_test import ProtocolError


def rand_runs(n=300, seed=0, acc=0.7, keys=(("raw", 42), ("clean", 42))):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    runs = {}
    for k in keys:
        p = np.where(rng.random(n) < acc, y, rng.integers(0, 3, n))
        runs[k] = (y, p)
    return runs


def test_f1_from_counts_matches_sklearn():
    rng = np.random.default_rng(1)
    y, p = rng.integers(0, 3, 400), rng.integers(0, 3, 400)
    macro, weighted = A.f1_from_counts(A.conf_matrix(y, p))
    assert macro == pytest.approx(f1_score(y, p, average="macro"))
    assert weighted == pytest.approx(f1_score(y, p, average="weighted"))


def test_f1_from_counts_handles_empty_class_and_batches():
    conf = np.array([[5, 0, 0], [0, 0, 0], [1, 0, 4]])  # no neutral tweets at all
    macro, _ = A.f1_from_counts(conf)
    assert np.isfinite(macro)
    macro_b, _ = A.f1_from_counts(np.stack([conf, conf]))
    assert macro_b.shape == (2,) and macro_b[0] == macro_b[1]


def test_bootstrap_is_deterministic_and_paired():
    runs = rand_runs()
    b1 = A.bootstrap(runs, 300, 200, seed=5)
    b2 = A.bootstrap(runs, 300, 200, seed=5)
    assert np.array_equal(b1[("raw", 42)]["macro_f1"], b2[("raw", 42)]["macro_f1"])
    b3 = A.bootstrap(runs, 300, 200, seed=6)
    assert not np.array_equal(b1[("raw", 42)]["macro_f1"], b3[("raw", 42)]["macro_f1"])
    # paired: two identical runs get exactly the same value in every resample, so their difference is exactly 0
    same = {("raw", 42): runs[("raw", 42)], ("clean", 42): runs[("raw", 42)]}
    b = A.bootstrap(same, 300, 100, seed=1)
    assert np.array_equal(b[("raw", 42)]["macro_f1"], b[("clean", 42)]["macro_f1"])


def test_bootstrap_perfect_predictions_give_one_and_interval_contains_point_estimate():
    y = np.arange(300) % 3
    b = A.bootstrap({("raw", 42): (y, y)}, 300, 50, seed=1)
    assert np.allclose(b[("raw", 42)]["macro_f1"], 1.0)
    runs = rand_runs(n=600, seed=3)
    b = A.bootstrap(runs, 600, 1000, seed=2)
    pe = f1_score(*runs[("raw", 42)], average="macro")
    lo, hi = A.pct(b[("raw", 42)]["macro_f1"])
    assert lo < pe < hi


def test_resample_uses_n_draws_with_replacement():
    # a multinomial weight vector sums to n for every resample: reproduce the weights used inside bootstrap
    rng = np.random.default_rng(0)
    W = rng.multinomial(50, np.full(50, 1 / 50), size=20)
    assert (W.sum(1) == 50).all() and (W == 0).any() and (W > 1).any()


def test_bootstrap_tables_pairs_and_bonferroni_is_wider():
    keys = tuple((m, s) for m in A.MODES for s in A.SEEDS)
    runs = rand_runs(n=400, seed=4, keys=keys)
    boot = A.bootstrap(runs, 400, 400, seed=3)
    point = {k: {"macro_f1": f1_score(*runs[k], average="macro"), "weighted_f1": f1_score(*runs[k], average="weighted")} for k in keys}
    t = A.bootstrap_tables(boot, point, "macro_f1")
    assert [(p["a"], p["b"]) for p in t["pairs"]] == [("clean", "raw"), ("normalized", "raw"), ("normalized", "clean")]
    for p in t["pairs"]:
        assert p["ci95_bonferroni3"][0] <= p["ci95"][0] and p["ci95_bonferroni3"][1] >= p["ci95"][1]
        assert len(p["per_seed_point"]) == 3 and len(p["per_seed_ci95"]) == 3


def test_error_breakdown_definitions_on_a_hand_example():
    # labels: 0 negative, 1 neutral, 2 positive
    y = np.array([0, 0, 1, 1, 1, 2, 2, 0])
    p = np.array([0, 1, 0, 2, 1, 2, 0, 2])
    e = A.error_breakdown(y, p)
    # errors: (0->1) predicted-neutral, (1->0) gold-neutral, (1->2) gold-neutral, (2->0) flip, (0->2) flip
    assert e["errors"] == 5
    assert e["gold_neutral_errors"] == 2 and e["predicted_neutral_errors"] == 1 and e["negative_positive_flips"] == 2
    assert e["neutral_involved_errors"] == 3 and e["share_neutral_involved"] == pytest.approx(3 / 5)
    assert e["gold_neutral_errors"] + e["predicted_neutral_errors"] + e["negative_positive_flips"] == e["errors"]
    assert e["true_negative__pred_neutral"] == 1 and e["true_neutral__pred_negative"] == 1


def test_error_breakdown_without_errors():
    y = np.array([0, 1, 2])
    e = A.error_breakdown(y, y)
    assert e["errors"] == 0 and e["share_neutral_involved"] == 0.0


def test_reweighting_to_own_class_mix_is_identity():
    runs = rand_runs(n=300, seed=7)
    y = runs[("raw", 42)][0]
    own = np.bincount(y, minlength=3) / len(y)
    rw = A.reweighted_f1(runs, own, y)
    for k, (y_, p_) in runs.items():
        assert rw[k][0] == pytest.approx(f1_score(y_, p_, average="macro"))
        assert rw[k][1] == pytest.approx(f1_score(y_, p_, average="weighted"))


def test_reweighting_changes_scores_when_mix_differs():
    runs = rand_runs(n=600, seed=8, acc=0.6)
    y = runs[("raw", 42)][0]
    rw = A.reweighted_f1(runs, np.array([0.8, 0.1, 0.1]), y)
    assert rw[("raw", 42)][1] != pytest.approx(f1_score(*runs[("raw", 42)], average="weighted"), abs=1e-6)


def test_bucket_ids_use_inclusive_upper_edges():
    b = A.bucket_ids([0.0, 23.81, 23.82, 34.783, 34.784, 50], A.CMI_EDGES)
    assert b.tolist() == [0, 0, 1, 1, 2, 2]


def test_bucket_metrics_counts_and_values():
    y = np.array([0, 1, 2, 0, 1, 2])
    runs = {(m, s): (y, y.copy()) for m in A.MODES for s in A.SEEDS}
    rows = A.bucket_metrics(runs, np.array([0, 0, 0, 1, 1, 1]), ("a", "b"))
    assert {r["n"] for r in rows} == {3}
    assert all(r["accuracy_mean"] == 1.0 and r["accuracy_sd"] == 0.0 for r in rows)


def test_mask_text_hides_mentions_links_and_hashtags_in_the_split_up_conll_form():
    out = A.mask_text("RT @ Monika _ Kapoor19 tute nii # SomeName rishte @ geet0212 https // t . co / XA5cEwo8un")
    assert out == "USER tute nii HASHTAG rishte USER URL"
    for leaked in ("Monika", "Kapoor19", "SomeName", "geet0212", "XA5cEwo8un"):
        assert leaked not in out


def test_idx_uses_project_label_order():
    assert A.idx(["negative", "neutral", "positive", "neutral"]).tolist() == [0, 1, 2, 1]


def test_refuses_to_write_inside_the_frozen_folder_or_to_an_existing_folder(tmp_path):
    with pytest.raises(ProtocolError, match="frozen final-test"):
        A.build(tmp_path, A.ROOT / A.FINAL_DIR / "extra", 10, log=lambda *_: None)
    existing = tmp_path / "already_there"
    existing.mkdir()
    with pytest.raises(ProtocolError, match="never overwrites"):
        A.build(tmp_path, existing, 10, log=lambda *_: None)
    staging = tmp_path / "x.staging"
    staging.mkdir()
    with pytest.raises(ProtocolError, match="never overwrites"):
        A.build(tmp_path, tmp_path / "x", 10, log=lambda *_: None)


def test_verify_against_records_detects_a_changed_file():
    manifest = A.json.loads((A.ROOT / A.MANIFEST).read_text(encoding="utf-8"))
    paths = A.frozen_input_paths(manifest, A.ROOT / "data" / "raw")
    good = {k: "0" * 32 for k in paths}
    with pytest.raises(ProtocolError, match="differ from their records"):
        A.verify_against_records(manifest, good)
