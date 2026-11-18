"""Tests of the final-evaluation infrastructure (src/evaluate_bilstm_test.py).

Everything here uses SYNTHETIC data: tiny checkpoints trained on a keyword-labelled toy dataset, a synthetic "test" file and a
synthetic label file in a temporary folder. The official test file and the official test labels are never used.
"""
import builtins
import hashlib
import json
import random
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import src.evaluate_bilstm_test as ev
from src.evaluate_bilstm_test import (
    LABELS,
    MODES,
    RUNS,
    SEEDS,
    ConfirmationRequired,
    HashMismatchError,
    ProtocolError,
    aggregate_runs,
    atomic_write,
    evaluate,
    full_metrics,
    join_by_uid,
    main,
    md5_file,
    read_labels_strict,
    validate_manifest,
    validate_predictions,
    verify_files,
)
from tests.synthetic_conll import FILLER, KEYWORDS, SMALL, make_data_root

UNL_NAME, LAB_NAME = "Hinglish_test_unalbelled_conll_updated.txt", "Hinglish_test_labels.txt"
N_TEST = 60


# ====================================================== pure-function tests ====
def test_join_by_uid_is_independent_of_row_order():
    labels = {u: LABELS[u % 3] for u in range(100, 130)}
    uids = list(labels)
    shuffled = uids[:]
    random.Random(1).shuffle(shuffled)
    assert join_by_uid(shuffled, labels) == [labels[u] for u in shuffled]  # follows the uids, not the dict/row order
    assert join_by_uid(uids, labels) != join_by_uid(shuffled, labels)  # so the order matters for the output ...
    pred = {u: LABELS[(u + (u % 2)) % 3] for u in uids}  # partly right, partly wrong
    m1 = full_metrics(join_by_uid(uids, labels), [pred[u] for u in uids])
    m2 = full_metrics(join_by_uid(shuffled, labels), [pred[u] for u in shuffled])
    assert m1 == m2  # ... but the metrics are the same for any row order


def test_join_rejects_missing_extra_and_duplicate_uids():
    labels = {1: "negative", 2: "neutral", 3: "positive"}
    with pytest.raises(ProtocolError, match="no label"):
        join_by_uid([1, 2, 3, 4], labels)  # a predicted uid without a label
    with pytest.raises(ProtocolError, match="no prediction"):
        join_by_uid([1, 2], labels)  # a label nobody predicted
    with pytest.raises(ProtocolError, match="duplicate"):
        join_by_uid([1, 2, 2, 3], labels)


def test_validate_predictions_coverage():
    uids = [5, 6, 7]
    validate_predictions(uids, [7, 5, 6], ["negative", "neutral", "positive"])  # any order is fine
    with pytest.raises(ProtocolError, match="missing"):
        validate_predictions(uids, [5, 6], ["negative", "neutral"])
    with pytest.raises(ProtocolError, match="unexpected"):
        validate_predictions(uids, [5, 6, 8], ["negative", "neutral", "positive"])
    with pytest.raises(ProtocolError, match="duplicate"):
        validate_predictions(uids, [5, 6, 6], ["negative", "neutral", "positive"])
    with pytest.raises(ProtocolError, match="invalid"):
        validate_predictions(uids, [5, 6, 7], ["negative", "neutral", "happy"])
    with pytest.raises(ProtocolError, match="uids but"):
        validate_predictions(uids, [5, 6, 7], ["negative", "neutral"])


def test_read_labels_strict_rejects_bad_files(tmp_path):
    def write(text):
        p = tmp_path / "labels.csv"
        p.write_text(text, encoding="utf-8")
        return p

    assert read_labels_strict(write("Uid,Sentiment\r\n1,negative\r\n2,neutral\r\n")) == {1: "negative", 2: "neutral"}
    for bad, msg in [("Id,Label\n1,negative\n", "header"), ("Uid,Sentiment\n1,negative\n1,neutral\n", "duplicate"),
                     ("Uid,Sentiment\n1,happy\n", "invalid label"), ("Uid,Sentiment\n1,negative,extra\n", "2 columns")]:
        with pytest.raises(ProtocolError, match=msg):
            read_labels_strict(write(bad))


def test_metrics_match_hand_calculation():
    y_true = ["negative", "negative", "neutral", "neutral", "positive", "positive", "neutral"]
    y_pred = ["negative", "neutral", "neutral", "neutral", "positive", "negative", "neutral"]
    m = full_metrics(y_true, y_pred)
    assert m["accuracy"] == pytest.approx(5 / 7)
    assert m["macro_precision"] == pytest.approx((0.5 + 0.75 + 1.0) / 3)
    assert m["macro_recall"] == pytest.approx((0.5 + 1.0 + 0.5) / 3)
    assert m["macro_f1"] == pytest.approx((0.5 + 6 / 7 + 2 / 3) / 3)
    assert m["weighted_precision"] == pytest.approx((2 * 0.5 + 3 * 0.75 + 2 * 1.0) / 7)
    assert m["weighted_recall"] == pytest.approx(5 / 7) and m["weighted_recall"] == pytest.approx(m["accuracy"])
    assert m["weighted_f1"] == pytest.approx((2 * 0.5 + 3 * 6 / 7 + 2 * 2 / 3) / 7)
    assert m["per_class"]["neutral"] == pytest.approx({"precision": 0.75, "recall": 1.0, "f1": 6 / 7, "support": 3})
    assert m["confusion"] == [[1, 1, 0], [0, 3, 0], [1, 0, 1]]  # rows = true, columns = predicted, order negative/neutral/positive


def test_class_never_predicted_gives_zero_not_nan():
    m = full_metrics(["negative", "positive"], ["negative", "negative"])
    assert m["per_class"]["positive"]["recall"] == 0.0 and m["per_class"]["neutral"]["f1"] == 0.0 and not np.isnan(m["macro_f1"])


def _fake_run(macro_f1, weighted_f1, offset=0.0):
    r = {k: 0.5 + offset for k in ev.SCALARS}
    r["macro_f1"], r["weighted_f1"] = macro_f1, weighted_f1
    r["per_class"] = {c: {"precision": 0.5, "recall": 0.5, "f1": 0.5 + offset, "support": 10} for c in LABELS}
    r["confusion"] = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    return r


def test_aggregation_uses_sample_standard_deviation_per_mode():
    per_run = {}
    for mode, base in zip(MODES, (0.60, 0.50, 0.40)):
        for k, seed in enumerate(SEEDS):
            per_run[(mode, seed)] = _fake_run(base + 0.02 * k, base + 0.05 * k, offset=0.01 * k)
    agg = aggregate_runs(per_run)
    assert agg["raw"]["macro_f1"]["mean"] == pytest.approx(0.62)
    assert agg["raw"]["macro_f1"]["std"] == pytest.approx(0.02)  # sample std (ddof=1); the population std would be 0.0163
    assert agg["raw"]["macro_f1"]["std"] == pytest.approx(np.std([0.60, 0.62, 0.64], ddof=1))
    assert agg["clean"]["weighted_f1"]["std"] == pytest.approx(0.05) and agg["normalized"]["macro_f1"]["values"] == pytest.approx([0.40, 0.42, 0.44])
    assert agg["raw"]["per_class"]["neutral"]["f1"]["mean"] == pytest.approx(np.mean([0.5, 0.51, 0.52]))
    assert agg["raw"]["confusion_summed"] == [[3, 6, 9], [12, 15, 18], [21, 24, 27]]  # summed over the three seeds
    assert set(agg) == set(MODES)


def test_aggregation_requires_all_nine_runs():
    per_run = {r: _fake_run(0.5, 0.5) for r in RUNS}
    del per_run[("clean", 43)]
    with pytest.raises(ProtocolError, match="nine runs"):
        aggregate_runs(per_run)


def test_md5_and_hash_mismatch_detection(tmp_path):
    f = tmp_path / "frozen.bin"
    f.write_bytes(b"frozen model bytes")
    entry = {"label": "ckpt", "path": f, "md5": hashlib.md5(b"frozen model bytes").hexdigest(), "size": 18}
    assert verify_files([entry]) == {"ckpt": entry["md5"]} and md5_file(f) == entry["md5"]
    f.write_bytes(b"frozen model bytez")  # same size, one byte changed
    with pytest.raises(HashMismatchError, match="MD5"):
        verify_files([entry])
    f.write_bytes(b"short")
    with pytest.raises(HashMismatchError, match="size"):
        verify_files([entry])
    assert verify_files([{**entry, "md5": "0" * 32, "size": 5}], check_md5=False) == {}  # size-only mode does not hash
    f.unlink()
    with pytest.raises(HashMismatchError, match="missing"):
        verify_files([entry])


def _good_manifest():
    return {"labels": LABELS, "expected": {k: 1 for k in ("n_test", "max_words", "max_chars", "train_size", "dev_size", "format_version", "selected_on", "variants_sha256")},
            "checkpoints": [{"mode": m, "seed": s, "path": f"c_{m}_{s}.pt", "md5": "a" * 32, "size": 10} for m, s in RUNS],
            "test_files": {"unlabeled": {"file": "u", "md5": "b" * 32, "size": 1}, "labels": {"file": "l", "md5": "c" * 32, "size": 1}}}


def test_manifest_must_list_exactly_the_nine_runs():
    validate_manifest(_good_manifest())
    m = _good_manifest()
    del m["checkpoints"][4]
    with pytest.raises(ProtocolError, match="exactly the nine"):
        validate_manifest(m)
    m = _good_manifest()
    m["checkpoints"][1] = dict(m["checkpoints"][0])  # duplicate raw/42
    with pytest.raises(ProtocolError, match="duplicate"):
        validate_manifest(m)
    m = _good_manifest()
    m["checkpoints"][0]["mode"] = "lowercased"
    with pytest.raises(ProtocolError):
        validate_manifest(m)
    m = _good_manifest()
    m["checkpoints"][0]["md5"] = "short"
    with pytest.raises(ProtocolError, match="malformed"):
        validate_manifest(m)
    m = _good_manifest()
    m["labels"] = ["positive", "neutral", "negative"]
    with pytest.raises(ProtocolError, match="order"):
        validate_manifest(m)


def test_atomic_write_leaves_no_temporary_file(tmp_path):
    atomic_write(tmp_path / "sub" / "a.txt", "héllo\n")
    atomic_write(tmp_path / "b.bin", b"\x00\x01")
    assert (tmp_path / "sub" / "a.txt").read_text(encoding="utf-8") == "héllo\n" and (tmp_path / "b.bin").read_bytes() == b"\x00\x01"
    assert not list(tmp_path.rglob("*.tmp"))


def test_confirmation_flag_is_checked_before_anything_else(tmp_path):
    with pytest.raises(ConfirmationRequired, match="--confirm-final-test"):
        evaluate(tmp_path / "does_not_exist.json", repo_root=tmp_path, confirm=False, validate_only=False)  # no file is even looked at
    with pytest.raises(SystemExit) as e:  # the CLI needs exactly one of the two flags
        main(["--manifest", str(tmp_path / "x.json")])
    assert e.value.code == 2
    with pytest.raises(SystemExit):
        main(["--validate-only", "--confirm-final-test"])
    assert not list(tmp_path.iterdir())


# ====================================================== synthetic environment ====
def _write_synthetic_test_files(root: Path, n=N_TEST, seed=7):
    rng = random.Random(seed)
    lab = ["Uid,Sentiment"]
    lines = []
    for i in range(n):
        uid, label = 90000 + i, LABELS[i % 3]
        toks = rng.sample(FILLER, 3) + rng.sample(KEYWORDS[label], 2) + rng.sample(FILLER, 2)
        if i % 4 == 0:
            toks = ["@", "user", "bohot", "acha"] + toks + ["https", "//", "t", ".", "co", "/", "abc123"]
        lines.append(f"meta\t{uid}")  # NO label in the unlabeled file
        lines += [f"{t}\tHin" for t in toks]
        lines.append("")
        lab.append(f"{uid},{label}")
    (root / UNL_NAME).write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    (root / LAB_NAME).write_bytes(("\r\n".join(lab) + "\r\n").encode("utf-8"))  # bytes: no newline translation on Windows


def _write_manifest(base: Path):
    from src.bilstm_checkpoint import load_checkpoint, variants_fingerprint

    entries = []
    for mode, seed in RUNS:
        rel = f"results/checkpoints/bilstm_full_{mode}_s{seed}_seed{seed}.pt"
        entries.append({"mode": mode, "seed": seed, "path": rel, "md5": md5_file(base / rel), "size": (base / rel).stat().st_size})
    ck = load_checkpoint(base / entries[0]["path"])
    root = base / "Hinglish"
    m = {"labels": LABELS,
         "expected": {"n_test": N_TEST, "max_words": ck.encoder.max_words, "max_chars": ck.encoder.max_chars, "train_size": 240, "dev_size": 60,
                      "format_version": 2, "selected_on": "dev weighted_f1", "variants_sha256": variants_fingerprint()},
         "checkpoints": entries,
         "test_files": {"unlabeled": {"file": UNL_NAME, "md5": md5_file(root / UNL_NAME), "size": (root / UNL_NAME).stat().st_size},
                        "labels": {"file": LAB_NAME, "md5": md5_file(root / LAB_NAME), "size": (root / LAB_NAME).stat().st_size}}}
    (base / "configs").mkdir(exist_ok=True)
    (base / "configs" / "bilstm_frozen_checkpoints.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    return m


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    pytest.importorskip("torch")
    from src.preprocess import load_processed_split
    from src.train_bilstm import BiLSTMConfig, run_experiment

    base = tmp_path_factory.mktemp("frozen_env")
    root = make_data_root(base)
    _write_synthetic_test_files(root)
    for mode in MODES:
        tr, dv = load_processed_split("train", mode=mode, root=root), load_processed_split("dev", mode=mode, root=root)
        for seed in SEEDS:
            cfg = BiLSTMConfig(**{**SMALL, "epochs": 1, "mode": mode})
            run_experiment(tr, dv, cfg, [seed], out_dir=base / "results", run_name=f"full_{mode}_s{seed}", verbose=False)
    _write_manifest(base)
    return SimpleNamespace(base=base)


class Env:
    """A private copy of the synthetic environment, so a test may modify files freely."""

    def __init__(self, src_base: Path, dst: Path):
        shutil.copytree(src_base, dst)
        self.base, self.data_root = dst, dst / "Hinglish"
        self.manifest_path = dst / "configs" / "bilstm_frozen_checkpoints.json"
        self.unl, self.lab = self.data_root / UNL_NAME, self.data_root / LAB_NAME

    def manifest(self):
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def save_manifest(self, m):
        self.manifest_path.write_text(json.dumps(m, indent=2), encoding="utf-8")

    def refresh_hashes(self):
        """Re-record the hashes of the (possibly edited) synthetic test files in the manifest."""
        m = self.manifest()
        for key, p in (("unlabeled", self.unl), ("labels", self.lab)):
            m["test_files"][key].update(md5=md5_file(p), size=p.stat().st_size)
        self.save_manifest(m)

    def run(self, out, **kw):
        kw.setdefault("confirm", True)
        kw.setdefault("make_figs", False)
        kw.setdefault("log", lambda *_: None)
        return evaluate(self.manifest_path, repo_root=self.base, data_root=self.data_root, out_dir=out, **kw)


@pytest.fixture()
def env(synth, tmp_path):
    return Env(synth.base, tmp_path / "env")


def label_lines(env_obj):
    return env_obj.lab.read_bytes().decode("utf-8").splitlines()


def write_label_lines(env_obj, lines):
    env_obj.lab.write_bytes(("\r\n".join(lines) + "\r\n").encode("utf-8"))


def tree_hash(root: Path):
    return {str(p.relative_to(root)): md5_file(p) for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.fixture()
def no_inference(monkeypatch):
    """Any call to predict_texts is counted (and must be zero in the tests that use this)."""
    from src.bilstm_checkpoint import LoadedCheckpoint

    calls = []

    def boom(self, *a, **k):
        calls.append(1)
        raise AssertionError("inference must not run here")

    monkeypatch.setattr(LoadedCheckpoint, "predict_texts", boom)
    return calls


def spy_open(monkeypatch, names, out_root):
    """Record every open() of the watched files together with the number of prediction files that already exist on disk."""
    real, log = builtins.open, []

    def fake(file, *a, **k):
        try:
            name = Path(file).name
        except TypeError:
            name = None
        if name in names:
            log.append((name, len(list(Path(out_root).rglob("*_predictions.csv")))))
        return real(file, *a, **k)

    monkeypatch.setattr(builtins, "open", fake)
    return log


# ===================================================== validation-only mode ====
def test_validate_only_reads_no_labels_runs_no_inference_and_writes_nothing(env, tmp_path, monkeypatch, no_inference):
    log = spy_open(monkeypatch, {UNL_NAME, LAB_NAME}, tmp_path)
    out = tmp_path / "out"
    res = env.run(out, validate_only=True, confirm=False)
    assert res["validated"] is True and not no_inference
    assert log == []  # neither the labels file nor even the unlabeled test file was opened (size check only)
    assert not out.exists() and not out.with_name("out.staging").exists()
    assert len(list(tmp_path.glob("out*"))) == 0


def test_validate_only_still_detects_a_changed_checkpoint(env, tmp_path):
    ck = env.base / env.manifest()["checkpoints"][3]["path"]
    data = bytearray(ck.read_bytes())
    data[-1] ^= 0xFF
    ck.write_bytes(bytes(data))
    with pytest.raises(HashMismatchError, match="MD5"):
        env.run(tmp_path / "out", validate_only=True, confirm=False)


# ===================================================== the full synthetic run ====
def test_full_run_produces_every_output_and_correct_numbers(env, tmp_path):
    out = tmp_path / "out"
    before = tree_hash(env.base)
    res = env.run(out, make_figs=True)
    assert tree_hash(env.base) == before  # checkpoints, test files, manifest: all unchanged
    names = {str(p.relative_to(out)).replace("\\", "/") for p in out.rglob("*") if p.is_file()}
    expected = {"predictions_manifest.json", "metrics_per_run.json", "metrics_summary.json", "confusion_matrices.json", "bilstm_final_test_report.md", "integrity_manifest.json"}
    expected |= {f"predictions/bilstm_final_test_{m}_seed{s}_predictions.csv" for m, s in RUNS}
    assert expected <= names and not list(out.rglob("*.tmp")) and not out.with_name("out.staging").exists()
    try:
        import matplotlib  # noqa: F401

        assert {"figures/bilstm_final_test_confusion_all_modes.png", *{f"figures/bilstm_final_test_confusion_{m}.png" for m in MODES}} <= names
    except ImportError:
        pass

    gold = dict(read_labels_strict(env.lab))  # synthetic labels
    per_run = json.loads((out / "metrics_per_run.json").read_text(encoding="utf-8"))
    summary = json.loads((out / "metrics_summary.json").read_text(encoding="utf-8"))
    for mode, seed in RUNS:
        df = pd.read_csv(out / "predictions" / f"bilstm_final_test_{mode}_seed{seed}_predictions.csv")
        assert list(df.columns) == ["uid", "pred"] and len(df) == N_TEST and df["uid"].is_unique and set(df["uid"]) == set(gold)
        independent = full_metrics([gold[u] for u in df["uid"]], list(df["pred"]))  # independent recomputation from the saved files
        stored = per_run[mode][str(seed)]
        assert stored["macro_f1"] == pytest.approx(independent["macro_f1"]) and stored["accuracy"] == pytest.approx(independent["accuracy"])
        assert json.loads((out / "confusion_matrices.json").read_text(encoding="utf-8"))["per_run"][mode][str(seed)] == independent["confusion"]
    for mode in MODES:
        vals = [per_run[mode][str(s)]["macro_f1"] for s in SEEDS]
        assert summary[mode]["macro_f1"]["mean"] == pytest.approx(np.mean(vals)) and summary[mode]["macro_f1"]["std"] == pytest.approx(np.std(vals, ddof=1))
    report = (out / "bilstm_final_test_report.md").read_text(encoding="utf-8")
    assert "Macro F1" in report and "Weighted F1" in report and "sample standard deviation" in report and "no preprocessing mode is declared the winner" in report
    manifest = json.loads((out / "integrity_manifest.json").read_text(encoding="utf-8"))
    names_in_events = [e["event"] for e in manifest["events"]]
    assert names_in_events.index("predictions_saved_and_validated") < names_in_events.index("labels_read_after_predictions_were_saved") < names_in_events.index("hashes_verified_after_evaluation")
    assert manifest["inputs"]["md5_before_and_after_identical"] and len(manifest["inputs"]["md5"]) == 9 + 2  # nine checkpoints + both test files
    for rel, h in manifest["outputs_md5"].items():
        assert md5_file(out / rel) == h
    assert res["out_dir"] == out


def test_predictions_are_saved_before_the_labels_file_is_first_read(env, tmp_path, monkeypatch):
    log = spy_open(monkeypatch, {LAB_NAME}, tmp_path)
    env.run(tmp_path / "out")
    assert log, "the labels file must be read in stage B"
    assert all(n_pred == 9 for _, n_pred in log)  # at the first and every later read, all nine prediction files already existed


def test_labels_cannot_influence_the_predictions(env, tmp_path):
    env.run(tmp_path / "out_a")
    # a second copy of the world with completely different (permuted) synthetic labels
    env2 = Env(Path(env.base), tmp_path / "env2")
    lines = label_lines(env2)
    head, rows = lines[0], [r.split(",") for r in lines[1:] if r]
    labs = [r[1] for r in rows]
    random.Random(3).shuffle(labs)
    write_label_lines(env2, [head] + [f"{r[0]},{l}" for r, l in zip(rows, labs)])
    env2.refresh_hashes()
    env2.run(tmp_path / "out_b")
    for m, s in RUNS:
        f = f"predictions/bilstm_final_test_{m}_seed{s}_predictions.csv"
        assert md5_file(tmp_path / "out_a" / f) == md5_file(tmp_path / "out_b" / f)  # identical predictions
    assert json.loads((tmp_path / "out_a" / "metrics_summary.json").read_text()) != json.loads((tmp_path / "out_b" / "metrics_summary.json").read_text())


def test_join_follows_uids_even_if_the_label_file_is_shuffled(env, tmp_path):
    env.run(tmp_path / "out_a")
    env2 = Env(Path(env.base), tmp_path / "env2")
    lines = label_lines(env2)
    rows = [r for r in lines[1:] if r]
    assert len(rows) == N_TEST
    random.Random(5).shuffle(rows)
    write_label_lines(env2, [lines[0]] + rows)
    env2.refresh_hashes()
    env2.run(tmp_path / "out_b")
    assert (tmp_path / "out_a" / "metrics_per_run.json").read_text() == (tmp_path / "out_b" / "metrics_per_run.json").read_text()


# ======================================================== integrity failures ====
def test_hash_mismatch_of_a_checkpoint_stops_everything_before_inference(env, tmp_path, no_inference):
    ck = env.base / env.manifest()["checkpoints"][0]["path"]
    data = bytearray(ck.read_bytes())
    data[len(data) // 2] ^= 0x01  # one flipped bit, same size
    ck.write_bytes(bytes(data))
    with pytest.raises(HashMismatchError, match="MD5"):
        env.run(tmp_path / "out")
    assert not no_inference
    assert not list(tmp_path.glob("out*"))  # not even a staging folder


def test_hash_mismatch_of_the_unlabeled_test_file_stops_before_inference(env, tmp_path, no_inference):
    env.unl.write_bytes(env.unl.read_bytes().replace(b"meta\t90001", b"meta\t90002", 1))
    with pytest.raises(HashMismatchError, match="unlabeled"):
        env.run(tmp_path / "out")
    assert not no_inference and not list(tmp_path.glob("out*"))


def test_labels_file_hash_is_checked_after_predictions_and_predictions_survive(env, tmp_path):
    data = bytearray(env.lab.read_bytes())
    data[len(data) // 2] ^= 0x01  # same size, different bytes
    env.lab.write_bytes(bytes(data))
    with pytest.raises(HashMismatchError, match="labels"):
        env.run(tmp_path / "out")
    failed = list(tmp_path.glob("out.failed_*"))
    assert len(failed) == 1 and not (tmp_path / "out").exists()  # nothing published
    assert len(list((failed[0] / "predictions").glob("*_predictions.csv"))) == 9  # the nine saved prediction files are preserved


def test_a_checkpoint_changing_during_the_run_is_detected(env, tmp_path, monkeypatch):
    from src.bilstm_checkpoint import LoadedCheckpoint

    victim = env.base / env.manifest()["checkpoints"][2]["path"]
    real, state = LoadedCheckpoint.predict_texts, {"done": False}

    def tamper(self, *a, **k):
        out = real(self, *a, **k)
        if not state["done"]:
            state["done"] = True
            with open(victim, "ab") as f:
                f.write(b"\x00")  # appended while the evaluation is running
        return out

    monkeypatch.setattr(LoadedCheckpoint, "predict_texts", tamper)
    with pytest.raises(HashMismatchError):
        env.run(tmp_path / "out")
    assert not (tmp_path / "out").exists() and len(list(tmp_path.glob("out.failed_*"))) == 1  # never published


def test_wrong_checkpoint_to_run_mapping_is_detected(env, tmp_path, no_inference):
    m = env.manifest()
    a = next(c for c in m["checkpoints"] if (c["mode"], c["seed"]) == ("raw", 42))
    b = next(c for c in m["checkpoints"] if (c["mode"], c["seed"]) == ("normalized", 42))
    a["path"], a["md5"], a["size"] = b["path"], b["md5"], b["size"]  # the 'raw seed 42' slot now points at the normalized checkpoint
    env.save_manifest(m)
    with pytest.raises(ProtocolError, match="raw/seed42: checkpoint was trained in mode 'normalized'"):
        env.run(tmp_path / "out")
    assert not no_inference
    m = env.manifest()
    c = next(c for c in m["checkpoints"] if (c["mode"], c["seed"]) == ("clean", 44))
    other = next(c2 for c2 in m["checkpoints"] if (c2["mode"], c2["seed"]) == ("clean", 42))
    c["path"], c["md5"], c["size"] = other["path"], other["md5"], other["size"]  # right mode, wrong seed
    env.save_manifest(m)
    with pytest.raises(ProtocolError, match="clean/seed44: checkpoint seed is 42"):
        env.run(tmp_path / "out")


def test_all_nine_checkpoints_map_to_their_own_mode_and_seed(env):
    from src.evaluate_bilstm_test import load_frozen_checkpoints

    cks = load_frozen_checkpoints(env.manifest(), env.base)
    assert set(cks) == set(RUNS)
    for (mode, seed), ck in cks.items():
        assert ck.config.mode == mode == ck.meta["mode"] and ck.meta["seed"] == seed and ck.meta["preprocess_options"] == ev.PRESETS[mode]


# ==================================================== coverage and uid problems ====
def test_duplicate_uid_in_the_unlabeled_test_file_stops_before_inference(env, tmp_path, no_inference):
    env.unl.write_bytes(env.unl.read_bytes().replace(b"meta\t90001", b"meta\t90000", 1))
    env.refresh_hashes()
    with pytest.raises(ProtocolError, match="duplicate uids"):
        env.run(tmp_path / "out")
    assert not no_inference and not list(tmp_path.glob("out*"))


def test_wrong_number_of_test_tweets_is_rejected(env, tmp_path, no_inference):
    m = env.manifest()
    m["expected"]["n_test"] = N_TEST + 1
    env.save_manifest(m)
    with pytest.raises(ProtocolError, match="expected 61"):
        env.run(tmp_path / "out")


def test_incomplete_prediction_coverage_is_rejected(env, tmp_path, monkeypatch):
    from src.bilstm_checkpoint import LoadedCheckpoint

    real = LoadedCheckpoint.predict_texts
    monkeypatch.setattr(LoadedCheckpoint, "predict_texts", lambda self, texts, **k: real(self, texts, **k)[:-1])  # one prediction short
    with pytest.raises(ProtocolError, match="uids but"):
        env.run(tmp_path / "out")
    assert not list(tmp_path.glob("out*"))  # failed during stage A: nothing published, staging discarded, a corrected rerun is possible
    monkeypatch.undo()
    env.run(tmp_path / "out")  # ... and the rerun indeed works


def test_missing_label_uid_is_rejected_after_predictions_are_saved(env, tmp_path):
    lines = label_lines(env)
    assert len(lines) == N_TEST + 1  # header + one row per test tweet
    write_label_lines(env, lines[:-1])  # drop the last uid
    env.refresh_hashes()
    with pytest.raises(ProtocolError, match="label coverage mismatch"):
        env.run(tmp_path / "out")
    failed = list(tmp_path.glob("out.failed_*"))
    assert len(failed) == 1 and len(list((failed[0] / "predictions").glob("*.csv"))) == 9 and not (tmp_path / "out").exists()


def test_extra_label_uid_is_rejected(env, tmp_path):
    env.lab.write_bytes(env.lab.read_bytes() + b"99999,neutral\r\n")
    env.refresh_hashes()
    with pytest.raises(ProtocolError, match="label coverage mismatch"):
        env.run(tmp_path / "out")


def test_duplicate_label_uid_is_rejected(env, tmp_path):
    env.lab.write_bytes(env.lab.read_bytes() + b"90000,neutral\r\n")
    env.refresh_hashes()
    with pytest.raises(ProtocolError, match="duplicate uid"):
        env.run(tmp_path / "out")


# ============================================== overwrite protection and the CLI ====
def test_existing_outputs_are_never_overwritten(env, tmp_path):
    out = tmp_path / "out"
    env.run(out)
    before = tree_hash(out)
    with pytest.raises(FileExistsError, match="never overwritten"):
        env.run(out)
    assert tree_hash(out) == before  # every byte untouched
    (tmp_path / "empty_out").mkdir()
    with pytest.raises(FileExistsError):
        env.run(tmp_path / "empty_out")  # even an existing empty folder blocks the run
    (tmp_path / "third.failed_20260101-000000-000000").mkdir()
    with pytest.raises(FileExistsError):
        env.run(tmp_path / "third")  # a failed earlier attempt must be dealt with deliberately
    (tmp_path / "fourth.staging").mkdir()
    with pytest.raises(FileExistsError):
        env.run(tmp_path / "fourth")
    assert not (tmp_path / "third").exists() and not (tmp_path / "fourth").exists()


def test_validate_only_also_reports_an_existing_output_folder(env, tmp_path):
    (tmp_path / "out").mkdir()
    with pytest.raises(FileExistsError):
        env.run(tmp_path / "out", validate_only=True, confirm=False)


def test_cli_requires_the_confirmation_flag_and_returns_exit_codes(env, tmp_path):
    common = ["--manifest", str(env.manifest_path), "--repo-root", str(env.base), "--data-root", str(env.data_root), "--no-figures"]
    with pytest.raises(SystemExit) as e:
        main(common + ["--out-dir", str(tmp_path / "x")])
    assert e.value.code == 2 and not (tmp_path / "x").exists()  # neither flag: refuses
    assert main(common + ["--validate-only", "--out-dir", str(tmp_path / "v")]) == 0 and not (tmp_path / "v").exists()
    assert main(common + ["--confirm-final-test", "--out-dir", str(tmp_path / "final")]) == 0 and (tmp_path / "final" / "bilstm_final_test_report.md").is_file()
    assert main(common + ["--confirm-final-test", "--out-dir", str(tmp_path / "final")]) == 1  # second run is refused, not overwritten
