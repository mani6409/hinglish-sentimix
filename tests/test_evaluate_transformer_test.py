"""Tests of the transformer final-evaluation program (src/evaluate_transformer_test.py).

Everything uses SYNTHETIC data: two copies of a tiny local BERT trained briefly on the keyword-labelled toy dataset, a synthetic
"test" file and a synthetic label file in a temporary folder. The official test files are never used.
"""
import builtins
import json
import random
import shutil
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from src.eval_protocol_utils import md5_file  # noqa: E402
from src.evaluate_transformer_test import (  # noqa: E402
    CHECKPOINT_FILES,
    MODELS,
    RUNS,
    SEEDS,
    ConfirmationRequired,
    HashMismatchError,
    ProtocolError,
    aggregate_runs,
    evaluate,
    main,
    pred_name,
    seed_differences,
    validate_manifest,
)
from src.metrics import LABELS, compute_metrics  # noqa: E402
from src.preprocess import load_processed_split  # noqa: E402
from src.train_transformer import checkpoint_dir, prediction_path, run_experiment  # noqa: E402
from src.transformer_utils import TransformerConfig  # noqa: E402
from tests.synthetic_conll import FILLER, KEYWORDS, make_data_root  # noqa: E402
from tests.test_transformer_pipeline import tiny_model  # noqa: E402,F401  (fixture)

LRS = {"mbert": 1e-3, "xlmr": 3e-3}
UNL_NAME, LAB_NAME = "Hinglish_test_unalbelled_conll_updated.txt", "Hinglish_test_labels.txt"
N_TEST = 60


def _write_synthetic_test_files(root: Path, n=N_TEST, seed=7):
    """A synthetic unlabeled "test" file (CoNLL, no labels) and a separate Uid,Sentiment label file, as in the official data."""
    rng = random.Random(seed)
    lab, lines = ["Uid,Sentiment"], []
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


def _write_manifest(base: Path, model_dirs: dict) -> None:
    root = base / "Hinglish"
    cks = []
    for model, seed in RUNS:
        run = f"{model}_s{seed}"
        d, p = checkpoint_dir(base / "results", run, seed), prediction_path(base / "results", run, seed)
        cks.append({"model": model, "seed": seed, "path": d.relative_to(base).as_posix(),
                    "files": {f: {"md5": md5_file(d / f), "size": (d / f).stat().st_size} for f in sorted(CHECKPOINT_FILES)},
                    "dev_predictions": {"path": p.relative_to(base).as_posix(), "md5": md5_file(p), "size": p.stat().st_size}})
    info = json.loads((base / cks[0]["path"] / "run_info.json").read_text(encoding="utf-8"))
    m = {"labels": LABELS,
         "expected": {"n_test": N_TEST, "train_size": 240, "dev_size": 60, "mode": "clean", "epochs": 2, "selected_on": "dev weighted_f1"},
         "inference": {"device": "cpu", "fp16": False, "eval_batch_size": 128},
         "dev_reproduction": {"min_agreement": 0.99, "max_weighted_f1_diff": 0.005},
         "models": {m_: {"model_name": model_dirs[m_], "checkpoint": model_dirs[m_], "lr": LRS[m_], "max_length": info["max_length"]} for m_ in MODELS},
         "checkpoints": cks,
         "test_files": {"unlabeled": {"file": UNL_NAME, "md5": md5_file(root / UNL_NAME), "size": (root / UNL_NAME).stat().st_size},
                        "labels": {"file": LAB_NAME, "md5": md5_file(root / LAB_NAME), "size": (root / LAB_NAME).stat().st_size}}}
    (base / "configs").mkdir(exist_ok=True)
    (base / "configs" / "frozen.json").write_text(json.dumps(m, indent=2), encoding="utf-8")


@pytest.fixture(scope="module")
def synth(tiny_model, tmp_path_factory):  # noqa: F811
    base = tmp_path_factory.mktemp("frozen_tf_env")
    root = make_data_root(base)
    _write_synthetic_test_files(root)
    tr, dv = load_processed_split("train", mode="clean", root=root), load_processed_split("dev", mode="clean", root=root)
    dirs = {}
    for model in MODELS:
        dirs[model] = str(base / f"tiny_{model}")
        shutil.copytree(tiny_model, dirs[model])
        cfg = TransformerConfig(model_name=dirs[model], epochs=2, patience=2, batch_size=16, lr=LRS[model], warmup_ratio=0.0,
                                device="cpu", fp16=False).validate()
        for seed in SEEDS:
            run_experiment(tr, dv, cfg, [seed], out_dir=base / "results", run_name=f"{model}_s{seed}", verbose=False)
    _write_manifest(base, dirs)
    return base


class Env:
    def __init__(self, src: Path, dst: Path):
        shutil.copytree(src, dst)
        self.base, self.root, self.manifest_path = dst, dst / "Hinglish", dst / "configs" / "frozen.json"

    def manifest(self):
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def save(self, m):
        self.manifest_path.write_text(json.dumps(m, indent=2), encoding="utf-8")

    def run(self, out=None, **kw):
        kw.setdefault("confirm", True)
        kw.setdefault("make_figs", False)
        kw.setdefault("log", lambda *_: None)
        return evaluate(self.manifest_path, repo_root=self.base, data_root=self.root, out_dir=out or self.base / "out", **kw)


@pytest.fixture()
def env(synth, tmp_path):
    return Env(synth, tmp_path / "env")


def watch_opens(monkeypatch, names, out_dir):
    """Record each open() of the watched files with the number of test prediction files already on disk at that moment."""
    real, seen = builtins.open, []

    def spy(file, *a, **k):
        if Path(str(file)).name in names:
            staged = out_dir.with_name(out_dir.name + ".staging") / "predictions"
            seen.append((Path(str(file)).name, len(list(staged.glob("*.csv"))) if staged.exists() else 0))
        return real(file, *a, **k)

    monkeypatch.setattr(builtins, "open", spy)
    return seen


# ====================================================================== tests ====
def test_manifest_needs_exactly_the_six_runs(env):
    m = env.manifest()
    validate_manifest(m)
    bad = json.loads(json.dumps(m))
    bad["checkpoints"].pop()
    with pytest.raises(ProtocolError, match="exactly the six"):
        validate_manifest(bad)
    bad = json.loads(json.dumps(m))
    bad["checkpoints"][1] = bad["checkpoints"][0]
    with pytest.raises(ProtocolError, match="duplicate"):
        validate_manifest(bad)
    bad = json.loads(json.dumps(m))
    del bad["checkpoints"][0]["files"]["model.safetensors"]
    with pytest.raises(ProtocolError, match="malformed"):
        validate_manifest(bad)


def test_confirmation_flag_is_required(tmp_path):
    with pytest.raises(ConfirmationRequired):
        evaluate(tmp_path / "missing.json", repo_root=tmp_path)
    with pytest.raises(SystemExit):
        main(["--manifest", str(tmp_path / "x.json")])
    assert not list(tmp_path.iterdir())


def test_validate_only_reproduces_dev_and_never_opens_test_files(env, monkeypatch):
    out = env.base / "out"
    seen = watch_opens(monkeypatch, {UNL_NAME, LAB_NAME}, out)
    res = env.run(out, validate_only=True, confirm=False)
    assert res["validated"] and not seen and not out.exists()
    assert all(d["agreement"] == 1.0 for d in res["dev_reproduction"].values())


def test_full_run_reads_labels_only_after_all_predictions_are_saved(env, monkeypatch):
    out = env.base / "out"
    seen = watch_opens(monkeypatch, {LAB_NAME}, out)
    res = env.run(out)
    assert seen and all(n == len(RUNS) for _, n in seen)  # every read of the labels file came after the six prediction files
    labels = dict(pd.read_csv(env.root / LAB_NAME).values)
    for model, seed in RUNS:
        p = pd.read_csv(out / "predictions" / pred_name(model, seed))
        assert list(p.columns) == ["uid", "pred"] and len(p) == N_TEST
        lab = pd.read_csv(out / "predictions_labelled" / pred_name(model, seed).replace("_predictions", ""))
        assert list(lab.columns) == ["uid", "label", "pred"] and (lab["label"] == lab["uid"].map(labels)).all()
        ref = compute_metrics(lab["label"], lab["pred"])
        assert res["per_run"][(model, seed)]["weighted_f1"] == pytest.approx(ref["weighted_f1"])
    summary = json.loads((out / "metrics_summary.json").read_text(encoding="utf-8"))
    assert set(MODELS) <= set(summary) and len(summary["seed_differences"]) == 3
    assert (out / "transformer_final_test_report.md").exists() and (out / "integrity_manifest.json").exists()
    with pytest.raises(FileExistsError):  # never a second run into the same place
        env.run(out)


def test_changed_checkpoint_file_or_extra_file_is_refused(env):
    m = env.manifest()
    cfg = env.base / m["checkpoints"][3]["path"] / "config.json"
    cfg.write_text(cfg.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(HashMismatchError, match="config.json"):
        env.run(validate_only=True, confirm=False)
    cfg.write_text(cfg.read_text(encoding="utf-8")[:-1], encoding="utf-8")
    (env.base / m["checkpoints"][0]["path"] / "adapter_model.bin").write_bytes(b"x")
    with pytest.raises(HashMismatchError, match="unexpected files"):
        env.run(validate_only=True, confirm=False)


def test_run_info_must_match_the_manifest(env):
    m = env.manifest()
    m["models"]["xlmr"]["lr"] = 5e-5
    env.save(m)
    with pytest.raises(ProtocolError, match="lr"):
        env.run(validate_only=True, confirm=False)


def test_failed_dev_reproduction_stops_before_any_test_file_is_touched(env, monkeypatch):
    out = env.base / "out"
    seen = watch_opens(monkeypatch, {UNL_NAME, LAB_NAME}, out)

    def wrong(ckpt_dir, texts, max_length, inference):
        return [LABELS[i % 3] for i in range(len(texts))]

    with pytest.raises(ProtocolError, match="do not reproduce"):
        env.run(out, predict=wrong)
    assert not out.exists() and not out.with_name("out.staging").exists()
    assert not [n for n, _ in seen if n == LAB_NAME]  # the labels file was never opened


def test_failure_after_labels_were_read_keeps_a_blocking_failed_folder(env):
    m = env.manifest()
    m["test_files"]["labels"]["md5"] = "0" * 32  # right size, wrong hash: detected in Stage B
    env.save(m)
    with pytest.raises(HashMismatchError, match="labels file"):
        env.run()
    failed = list(env.base.glob("out.failed_*"))
    assert len(failed) == 1 and len(list((failed[0] / "predictions").glob("*.csv"))) == len(RUNS)
    with pytest.raises(FileExistsError):
        env.run()


def test_aggregation_uses_sample_sd_and_seed_differences():
    def run(f):
        r = {k: f for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_precision", "weighted_recall", "weighted_f1")}
        r["per_class"] = {c: {"precision": f, "recall": f, "f1": f, "support": 1} for c in LABELS}
        r["confusion"] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        return r

    per_run = {("mbert", s): run(0.60 + 0.01 * i) for i, s in enumerate(SEEDS)} | {("xlmr", s): run(0.62 + 0.02 * i) for i, s in enumerate(SEEDS)}
    agg = aggregate_runs(per_run)
    assert agg["mbert"]["weighted_f1"]["std"] == pytest.approx(0.01) and agg["xlmr"]["weighted_f1"]["mean"] == pytest.approx(0.64)
    assert agg["xlmr"]["confusion_summed"] == [[3, 0, 0], [0, 3, 0], [0, 0, 3]]
    d = seed_differences(per_run)
    assert [round(x["weighted_f1_mbert_minus_xlmr"], 4) for x in d] == [-0.02, -0.03, -0.04]
    del per_run[("xlmr", 44)]
    with pytest.raises(ProtocolError, match="six runs"):
        aggregate_runs(per_run)
