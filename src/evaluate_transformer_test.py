"""Final, single evaluation of the six frozen transformer checkpoints (mBERT and XLM-R x seeds 42, 43, 44) on the official
SentiMix Hinglish TEST set.

READ docs/transformer_final_evaluation_protocol.md FIRST. This script is meant to be run ONCE.

    python -m src.evaluate_transformer_test --validate-only          # checks only: no test file is opened, nothing is written
    python -m src.evaluate_transformer_test --confirm-final-test     # the real, one-time evaluation

It follows the BiLSTM protocol (docs/bilstm_final_evaluation_protocol.md) and uses the same helpers, now in
src/eval_protocol_utils.py:
- Every file of the six frozen checkpoint folders, and the six frozen DEV prediction files, are MD5-verified before anything runs;
  each checkpoint's run_info.json must describe the run the manifest names (model, seed, lr, mode, epochs, max_length, ...).
- Dev reproduction (both modes): every checkpoint predicts the DEV split again and must reproduce its frozen dev predictions
  (agreement and weighted F1 within the tolerances fixed in the manifest). This proves that the inference path used for the test
  set is the one the models were selected with. Only train/dev data are involved.
- Stage A: the unlabeled test file is hash-verified and parsed (no labels); all six checkpoints predict; the six prediction files
  are written, read back and validated.
- Stage B: only then is the labels file hash-verified and read for the FIRST time; labels are joined to the saved predictions by
  UID (never by row order) and all metrics are computed from the saved files.
- All hashed inputs are verified again at the end. Outputs go to a staging folder and are published with one atomic rename; an
  existing output folder is never overwritten. A failure after the labels were read leaves "<out-dir>.failed_<time>", which
  blocks a silent rerun.
- Nothing is selected, tuned or changed from test results.

--validate-only never opens, parses or hashes either test file (it checks only that they exist with the expected size), runs no
test inference and writes nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import DEFAULT_ROOT
from src.eval_protocol_utils import (
    CLASS_KEYS,
    PRIMARY,
    SCALARS,
    SECONDARY,
    ConfirmationRequired,
    HashMismatchError,
    ProtocolError,
    atomic_write,
    atomic_write_json,
    csv_text,
    full_metrics,
    join_by_uid,
    md5_file,
    read_labels_strict,
    read_predictions_csv,
    read_unlabeled_test,
    utc_now,
    validate_predictions,
    verify_files,
)
from src.metrics import LABELS, compute_metrics
from src.preprocess import PRESETS, preprocess_frame

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("mbert", "xlmr")
SEEDS = (42, 43, 44)
RUNS = tuple(product(MODELS, SEEDS))  # the six frozen runs
DEFAULT_MANIFEST = "configs/transformer_frozen_checkpoints.json"
DEFAULT_OUT_DIR = "results/transformer_final_test"
PROTOCOL_DOC = "docs/transformer_final_evaluation_protocol.md"
CHECKPOINT_FILES = {"config.json", "model.safetensors", "run_info.json", "tokenizer.json", "tokenizer_config.json"}
# run_info.config keys that may differ between the frozen runs of ONE model: only the seed (stored outside config)
PER_MODEL_KEYS = ("model_name", "lr")  # keys that may differ BETWEEN the two models


# ================================================================ the manifest ===
def validate_manifest(m: dict) -> dict:
    """Structure checks: exactly the six (model, seed) runs, well-formed hashes, every required setting present."""
    for key in ("labels", "expected", "inference", "dev_reproduction", "models", "checkpoints", "test_files"):
        if key not in m:
            raise ProtocolError(f"manifest is missing '{key}'")
    if list(m["labels"]) != LABELS:
        raise ProtocolError(f"manifest labels {m['labels']} differ from the project order {LABELS}")
    need = {"n_test", "train_size", "dev_size", "mode", "epochs", "selected_on"}
    if not need <= set(m["expected"]):
        raise ProtocolError(f"manifest 'expected' lacks {sorted(need - set(m['expected']))}")
    if m["expected"]["mode"] not in PRESETS:
        raise ProtocolError(f"unknown text mode {m['expected']['mode']!r}")
    if not {"device", "fp16", "eval_batch_size"} <= set(m["inference"]):
        raise ProtocolError("manifest 'inference' needs device, fp16 and eval_batch_size")
    if not {"min_agreement", "max_weighted_f1_diff"} <= set(m["dev_reproduction"]):
        raise ProtocolError("manifest 'dev_reproduction' needs min_agreement and max_weighted_f1_diff")
    if set(m["models"]) != set(MODELS):
        raise ProtocolError(f"manifest 'models' must be exactly {list(MODELS)}")
    for name, spec in m["models"].items():
        if not {"model_name", "checkpoint", "lr", "max_length"} <= set(spec):
            raise ProtocolError(f"manifest models.{name} needs model_name, checkpoint, lr and max_length")
    pairs = [(c.get("model"), c.get("seed")) for c in m["checkpoints"]]
    dup = sorted({p for p in pairs if pairs.count(p) > 1}, key=str)
    if dup:
        raise ProtocolError(f"duplicate checkpoint entries for {dup}")
    if set(pairs) != set(RUNS) or len(pairs) != len(RUNS):
        raise ProtocolError(f"the manifest must list exactly the six runs {list(RUNS)}; found {pairs}")

    def ok_hash(e):
        return isinstance(e, dict) and isinstance(e.get("md5"), str) and len(e["md5"]) == 32 and isinstance(e.get("size"), int)

    for c in m["checkpoints"]:
        tag = f"{c.get('model')}/{c.get('seed')}"
        if not isinstance(c.get("path"), str) or not isinstance(c.get("files"), dict) or set(c["files"]) != CHECKPOINT_FILES:
            raise ProtocolError(f"malformed checkpoint entry {tag} (needs path and exactly the files {sorted(CHECKPOINT_FILES)})")
        if not all(ok_hash(e) and e["size"] > 0 for e in c["files"].values()):
            raise ProtocolError(f"malformed file hash in checkpoint entry {tag}")
        dp = c.get("dev_predictions")
        if not (ok_hash(dp) and isinstance(dp.get("path"), str)):
            raise ProtocolError(f"malformed dev_predictions entry for {tag}")
    for key in ("unlabeled", "labels"):
        t = m["test_files"].get(key, {})
        if not (ok_hash(t) and isinstance(t.get("file"), str)):
            raise ProtocolError(f"malformed test_files.{key} entry")
    return m


def load_manifest(path) -> dict:
    return validate_manifest(json.loads(Path(path).read_text(encoding="utf-8")))


def frozen_entries(manifest: dict, repo_root: Path) -> list:
    """verify_files entries for every checkpoint file and every frozen dev prediction file. A checkpoint folder must hold
    exactly the listed files (an extra file could change what from_pretrained loads)."""
    entries, extra = [], []
    for c in manifest["checkpoints"]:
        d = repo_root / c["path"]
        tag = f"{c['model']} seed {c['seed']}"
        if d.is_dir():
            found = {p.name for p in d.iterdir()}
            if found - CHECKPOINT_FILES:
                extra.append(f"{tag}: unexpected files {sorted(found - CHECKPOINT_FILES)} in {d}")
        for name, h in sorted(c["files"].items()):
            entries.append({"label": f"checkpoint {tag}: {name}", "path": d / name, "md5": h["md5"], "size": h["size"]})
        dp = c["dev_predictions"]
        entries.append({"label": f"dev predictions {tag}", "path": repo_root / dp["path"], "md5": dp["md5"], "size": dp["size"]})
    if extra:
        raise HashMismatchError("integrity check failed:\n  - " + "\n  - ".join(extra))
    return entries


def check_run_info(manifest: dict, repo_root: Path) -> dict:
    """Read every run_info.json (no model is loaded) and verify that each checkpoint IS the run the manifest names, and that the
    six runs share every setting except the seed, and the model and its learning rate. Returns {(model, seed): run_info}."""
    exp, problems, infos = manifest["expected"], [], {}
    shared = None
    for c in manifest["checkpoints"]:
        model, seed = c["model"], c["seed"]
        spec, tag = manifest["models"][model], f"{model}/seed{seed}"
        info = json.loads((repo_root / c["path"] / "run_info.json").read_text(encoding="utf-8"))
        cfg = info.get("config", {})
        checks = [
            (cfg.get("model_name"), spec["model_name"], "config.model_name"), (info.get("checkpoint"), spec["checkpoint"], "checkpoint"),
            (info.get("seed"), seed, "seed"), (cfg.get("lr"), spec["lr"], "lr"), (cfg.get("mode"), exp["mode"], "mode"),
            (cfg.get("epochs"), exp["epochs"], "epochs"), (info.get("max_length"), spec["max_length"], "max_length"),
            (info.get("selected_on"), exp["selected_on"], "selected_on"), (info.get("preprocess_options"), dict(PRESETS[exp["mode"]]), "preprocess_options"),
            (info.get("train_size"), exp["train_size"], "train_size"), (info.get("dev_size"), exp["dev_size"], "dev_size"),
        ]
        problems += [f"{tag}: {what} is {got!r}, expected {want!r}" for got, want, what in checks if got != want]
        rest = {k: v for k, v in cfg.items() if k not in PER_MODEL_KEYS}
        if shared is None:
            shared = rest
        elif rest != shared:
            diff = sorted(k for k in set(rest) | set(shared) if rest.get(k) != shared.get(k))
            problems.append(f"{tag}: training settings differ from the other runs in {diff}")
        infos[(model, seed)] = info
    if problems:
        raise ProtocolError("checkpoint verification failed:\n  - " + "\n  - ".join(problems))
    return infos


# ================================================================= inference ===
def resolve_inference_device(inference: dict):
    import torch

    name = inference["device"]
    if name == "cuda" and not torch.cuda.is_available():
        raise ProtocolError("the manifest fixes inference on CUDA, but no CUDA device is available")
    return torch.device(name)


def predict_with_checkpoint(ckpt_dir: Path, texts, max_length: int, inference: dict) -> list:
    """Labels for `texts` from one frozen checkpoint: same tokenisation, truncation, padding, batch size and precision as the dev
    evaluation during training (src.train_transformer). Eval mode, no gradients. The model is freed afterwards."""
    import torch
    from torch.utils.data import DataLoader

    from src.transformer_utils import EncodedTweets, ids_to_labels, load_checkpoint_dir, make_collate, predict_ids

    device = resolve_inference_device(inference)
    model, tokenizer, _ = load_checkpoint_dir(ckpt_dir)
    model.to(device)
    loader = DataLoader(EncodedTweets(tokenizer, list(texts), None, max_length), batch_size=inference["eval_batch_size"],
                        shuffle=False, collate_fn=make_collate(tokenizer))
    preds = ids_to_labels(predict_ids(model, loader, device, fp16=bool(inference["fp16"])))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return preds


def read_frozen_dev_predictions(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if list(df.columns) != ["uid", "label", "pred"]:
        raise ProtocolError(f"{path}: expected columns uid,label,pred")
    return df


def check_dev_reproduction(manifest: dict, repo_root: Path, dev_df: pd.DataFrame, infos: dict, predict=predict_with_checkpoint,
                           log=print) -> dict:
    """Every checkpoint predicts DEV again; agreement with its frozen dev predictions and the weighted F1 must be within the
    manifest's tolerances (GPU fp16 kernels are not bit-exact, so a few changed predictions are allowed)."""
    tol, out, problems = manifest["dev_reproduction"], {}, []
    texts, uids, gold = list(dev_df["text_proc"]), list(dev_df["uid"]), list(dev_df["label"])
    for c in manifest["checkpoints"]:
        model, seed = c["model"], c["seed"]
        tag = f"{model}/seed{seed}"
        frozen = read_frozen_dev_predictions(repo_root / c["dev_predictions"]["path"])
        if sorted(frozen["uid"]) != sorted(uids):
            raise ProtocolError(f"{tag}: frozen dev predictions do not cover the dev split")
        preds = predict(repo_root / c["path"], texts, manifest["models"][model]["max_length"], manifest["inference"])
        frozen_by_uid = dict(zip(frozen["uid"], frozen["pred"]))
        agree = float(np.mean([p == frozen_by_uid[u] for u, p in zip(uids, preds)]))
        wf1 = compute_metrics(gold, preds)["weighted_f1"]
        reported = infos[(model, seed)]["dev_metrics"]["weighted_f1"]
        out[tag] = {"agreement": agree, "weighted_f1": wf1, "weighted_f1_reported_in_training": reported,
                    "weighted_f1_diff": abs(wf1 - reported), "changed_predictions": int(round((1 - agree) * len(uids)))}
        log(f"  dev reproduction {tag}: agreement {agree:.4f}, wF1 {wf1:.4f} (training reported {reported:.4f})")
        if agree < tol["min_agreement"] or abs(wf1 - reported) > tol["max_weighted_f1_diff"]:
            problems.append(f"{tag}: agreement {agree:.4f} (min {tol['min_agreement']}), wF1 difference {abs(wf1 - reported):.4f} "
                            f"(max {tol['max_weighted_f1_diff']})")
    if problems:
        raise ProtocolError("the frozen checkpoints do not reproduce their dev predictions:\n  - " + "\n  - ".join(problems))
    return out


# ===================================================================== metrics ===
def aggregate_runs(per_run: dict) -> dict:
    """Per model: mean and SAMPLE standard deviation (ddof = 1) over seeds 42, 43, 44, for every scalar and class-wise metric."""
    missing = [r for r in RUNS if r not in per_run]
    if missing:
        raise ProtocolError(f"aggregation needs all six runs; missing {missing}")

    def stat(vals):
        v = np.asarray(vals, float)
        return {"mean": float(v.mean()), "std": float(v.std(ddof=1)), "values": [float(x) for x in v]}

    out = {}
    for model in MODELS:
        runs = [per_run[(model, s)] for s in SEEDS]
        out[model] = {k: stat([r[k] for r in runs]) for k in SCALARS}
        out[model]["per_class"] = {c: {k: stat([r["per_class"][c][k] for r in runs]) for k in CLASS_KEYS} for c in LABELS}
        out[model]["confusion_summed"] = np.sum([np.asarray(r["confusion"]) for r in runs], axis=0).tolist()
    return out


def seed_differences(per_run: dict) -> list:
    """Pre-specified descriptive comparison: mBERT minus XLM-R for each seed and each primary metric."""
    return [{"seed": s, **{f"{k}_mbert_minus_xlmr": per_run[("mbert", s)][k] - per_run[("xlmr", s)][k] for k in PRIMARY}} for s in SEEDS]


# ===================================================================== report ===
def _ms(stat, d=4):
    return f"{stat['mean']:.{d}f} ± {stat['std']:.{d}f}"


NAMES = {"mbert": "mBERT", "xlmr": "XLM-R"}


def render_report(ctx: dict) -> str:
    agg, per_run, n, man = ctx["aggregate"], ctx["per_run"], ctx["n_test"], ctx["manifest"]
    L = []
    add = L.append
    add("# Transformer final test-set evaluation (single, frozen-model run)")
    add("")
    add(f"*Evaluates the six frozen checkpoints (mBERT and XLM-R x seeds 42, 43, 44) on the official SentiMix Hinglish test set ({n:,} tweets). "
        f"Protocol: `{PROTOCOL_DOC}`. Produced by `python -m src.evaluate_transformer_test --confirm-final-test` on {ctx['finished_utc']}.*")
    add("")
    add("**Nothing was selected, tuned or changed based on these results.** Checkpoints, learning rates, text preprocessing, inference settings and metrics "
        "were frozen before the test labels were read. No model is declared the winner by this evaluation.")
    add("")
    add("## 1. Primary results: mean ± sample standard deviation over seeds 42, 43, 44")
    add("")
    add("| Model | Checkpoint | lr | Macro F1 | Weighted F1 |")
    add("|---|---|---|---|---|")
    for m in MODELS:
        spec = man["models"][m]
        add(f"| {NAMES[m]} | `{spec['checkpoint']}` | {spec['lr']:g} | **{_ms(agg[m]['macro_f1'])}** | **{_ms(agg[m]['weighted_f1'])}** |")
    add("")
    add("Per seed (seed 42 / 43 / 44):")
    add("")
    add("| Model | Macro F1 | Weighted F1 |")
    add("|---|---|---|")
    for m in MODELS:
        add(f"| {NAMES[m]} | " + " / ".join(f"{v:.4f}" for v in agg[m]["macro_f1"]["values"]) + " | "
            + " / ".join(f"{v:.4f}" for v in agg[m]["weighted_f1"]["values"]) + " |")
    add("")
    add("Pre-specified descriptive comparison, mBERT minus XLM-R per seed (same seed, same test tweets):")
    add("")
    add("| Seed | Macro F1 difference | Weighted F1 difference |")
    add("|---|---|---|")
    for d in ctx["differences"]:
        add(f"| {d['seed']} | {d['macro_f1_mbert_minus_xlmr']:+.4f} | {d['weighted_f1_mbert_minus_xlmr']:+.4f} |")
    add("")
    add("Standard deviations are sample standard deviations (n = 3, ddof = 1) and are rough with three seeds. "
        "The test set is one sample of 3,000 tweets, so F1 carries a sampling error of roughly ±0.01 that is not estimated here.")
    add("")
    add("## 2. Secondary metrics (mean ± sample std over seeds)")
    add("")
    add("| Model | Accuracy | Macro precision | Macro recall | Weighted precision | Weighted recall |")
    add("|---|---|---|---|---|---|")
    for m in MODELS:
        add(f"| {NAMES[m]} | " + " | ".join(_ms(agg[m][k]) for k in SECONDARY) + " |")
    add("")
    add("Weighted recall equals accuracy by definition.")
    add("")
    add("## 3. Class-wise precision, recall and F1 (mean ± sample std over seeds)")
    add("")
    add("| Model | Class | Support | Precision | Recall | F1 |")
    add("|---|---|---|---|---|---|")
    for m in MODELS:
        for c in LABELS:
            sup = per_run[(m, SEEDS[0])]["per_class"][c]["support"]
            add(f"| {NAMES[m]} | {c} | {sup:,} | " + " | ".join(_ms(agg[m]["per_class"][c][k]) for k in CLASS_KEYS) + " |")
    add("")
    add("## 4. Every run")
    add("")
    add("| Run | Accuracy | Macro P | Macro R | Macro F1 | Weighted P | Weighted R | Weighted F1 |")
    add("|---|---|---|---|---|---|---|---|")
    for (m, s) in RUNS:
        r = per_run[(m, s)]
        add(f"| {NAMES[m]} seed {s} | " + " | ".join(f"{r[k]:.4f}" for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1",
                                                                             "weighted_precision", "weighted_recall", "weighted_f1")) + " |")
    add("")
    add("## 5. Confusion matrices")
    add("")
    add(f"Rows = true class, columns = predicted class. Each matrix is the **sum over the three seeds** ({3 * n:,} predictions per model); "
        "percentages are row-normalised. Per-run matrices are in `confusion_matrices.json`.")
    add("")
    for m in MODELS:
        cm = np.asarray(agg[m]["confusion_summed"])
        add(f"**{NAMES[m]}**")
        add("")
        add("| | pred negative | pred neutral | pred positive | Row total |")
        add("|---|---|---|---|---|")
        for i, c in enumerate(LABELS):
            tot = cm[i].sum()
            add(f"| true {c} | " + " | ".join(f"{cm[i, j]:,} ({100 * cm[i, j] / tot:.1f}%)" if tot else "0" for j in range(3)) + f" | {tot:,} |")
        add("")
    if ctx.get("figures"):
        add("Figure: " + ", ".join(f"`figures/{f}`" for f in ctx["figures"]) + ".")
        add("")
    elif ctx.get("figures_error"):
        add(f"*The figure was not generated: {ctx['figures_error']}*")
        add("")
    add("## 6. Integrity and procedure")
    add("")
    add("- Dev reproduction before the test files were touched (agreement with the frozen dev predictions; weighted F1 recomputed vs. reported in training):")
    add("")
    add("| Run | Agreement | Changed predictions | Dev wF1 now | Dev wF1 in training |")
    add("|---|---|---|---|---|")
    for tag, d in ctx["dev_reproduction"].items():
        add(f"| {tag} | {d['agreement']:.4f} | {d['changed_predictions']} | {d['weighted_f1']:.4f} | {d['weighted_f1_reported_in_training']:.4f} |")
    add("")
    add("- All checkpoint files, the frozen dev predictions and the unlabeled test file were MD5-verified **before** inference. The labels file was verified and read "
        "for the **first time only after** all six test prediction files had been written and validated. Every hashed input was verified **again after** evaluation and was unchanged.")
    add("- Predictions were joined to the gold labels by **UID**, never by row order. Metrics were computed from the saved prediction files.")
    add("- Hashes, the time of every stage, the environment and the output hashes are in `integrity_manifest.json`.")
    add("")
    add("## 7. Limitations")
    add("")
    add("- Three seeds per model: the standard deviation is rough.")
    add("- One test sample of 3,000 tweets; differences of about 0.01 F1 are within sampling error.")
    add("- The learning rate and the best epoch were chosen on dev; seed 42 also took part in choosing the learning rate. The test scores are the unbiased estimate for these frozen models.")
    add("- The team's EDA computed descriptive statistics on the test split earlier (label counts, overlaps, CMI); no transformer decision used them (see the protocol).")
    add("")
    return "\n".join(L) + "\n"


def make_figure(fig_dir: Path, aggregate: dict) -> list:
    """One figure, two row-normalised confusion matrices (counts summed over seeds), single-hue ramp, shared scale."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    surface, ink, ink2 = "#fcfcfb", "#0b0b0b", "#52514e"
    cmap = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"])
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
                         "figure.facecolor": surface, "savefig.facecolor": surface, "axes.facecolor": surface, "text.color": ink})
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.0), dpi=300)
    fig.subplots_adjust(left=0.09, right=0.88, top=0.78, bottom=0.14, wspace=0.35)
    im = None
    for ax, m in zip(axes, MODELS):
        conf = np.asarray(aggregate[m]["confusion_summed"])
        pct = 100 * conf / conf.sum(1, keepdims=True)
        im = ax.imshow(pct, cmap=cmap, vmin=0, vmax=100, aspect="equal", interpolation="nearest")
        for i in range(3):
            for j in range(3):
                dark = pct[i, j] > 48
                ax.text(j, i - 0.12, f"{pct[i, j]:.1f}%", ha="center", va="center", fontsize=11,
                        fontweight="bold" if i == j else "normal", color="#ffffff" if dark else ink)
                ax.text(j, i + 0.22, f"n = {conf[i, j]:,}", ha="center", va="center", fontsize=7.5, color="#e6eefb" if dark else ink2)
        ax.set_xticks(range(3), [c.capitalize() for c in LABELS])
        ax.set_yticks(range(3), [c.capitalize() for c in LABELS])
        ax.set_xlabel("Predicted label", color=ink2)
        ax.set_ylabel("True label", color=ink2)
        ax.tick_params(length=0, colors=ink2)
        for sp in ax.spines.values():
            sp.set_visible(False)
        for k in (0.5, 1.5):
            ax.axhline(k, color=surface, linewidth=3)
            ax.axvline(k, color=surface, linewidth=3)
        ax.set_title(NAMES[m], loc="left", fontsize=12, fontweight="bold", pad=10)
    cax = fig.add_axes([0.905, 0.25, 0.013, 0.42])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("% of the true class", color=ink2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0)
    fig.text(0.09, 0.94, "Transformer confusion matrices on the test set", fontsize=13, fontweight="bold")
    fig.text(0.09, 0.895, "Row-normalised; counts summed over seeds 42, 43, 44; one shared color scale", fontsize=9, color=ink2)
    name = "transformer_final_test_confusion.png"
    fig.savefig(fig_dir / name)
    plt.close(fig)
    return [name]


# ================================================================ orchestration ===
def environment_info(infos: dict) -> dict:
    info = {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__}
    for mod in ("sklearn", "torch", "transformers", "tokenizers"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            pass
    try:
        import torch

        info["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001
        pass
    info["torch_version_recorded_in_checkpoints"] = sorted({str(i.get("torch_version")) for i in infos.values()})
    return info


def pred_name(model: str, seed: int) -> str:
    return f"transformer_final_test_{model}_seed{seed}_predictions.csv"


def evaluate(manifest_path, *, repo_root=ROOT, data_root=None, out_dir=None, validate_only: bool = False, confirm: bool = False,
             make_figs: bool = True, log=print, predict=predict_with_checkpoint) -> dict:
    """Run the protocol. validate_only=True: pre-flight and dev reproduction only. Otherwise `confirm=True` is mandatory."""
    if not validate_only and not confirm:
        raise ConfirmationRequired("refusing to run: the final test evaluation needs the explicit --confirm-final-test flag "
                                   "(use --validate-only for the pre-flight checks).")
    repo_root = Path(repo_root)
    data_root = Path(data_root) if data_root else DEFAULT_ROOT
    out_dir = Path(out_dir) if out_dir else repo_root / DEFAULT_OUT_DIR
    staging = out_dir.with_name(out_dir.name + ".staging")
    blockers = [p for p in (out_dir, staging, *out_dir.parent.glob(out_dir.name + ".failed_*")) if p.exists()] if out_dir.parent.exists() else []
    if blockers:
        raise FileExistsError("an earlier output or attempt exists and is never overwritten; remove it deliberately or choose another --out-dir: "
                              + ", ".join(str(b) for b in blockers))
    manifest = load_manifest(manifest_path)
    exp, tf = manifest["expected"], manifest["test_files"]
    unl_path, lab_path = data_root / tf["unlabeled"]["file"], data_root / tf["labels"]["file"]
    events = []

    def ev(name, **kw):
        events.append({"event": name, "utc": utc_now(), **kw})
        log(f"[{events[-1]['utc']}] {name}" + (f" {kw}" if kw else ""))

    ev("start", mode="validate-only" if validate_only else "final-evaluation", manifest=str(manifest_path))

    # ---- pre-flight: hashes of all frozen files; test files size-only (validate-only) or unlabeled hashed (final)
    fz = frozen_entries(manifest, repo_root)
    unl_entry = {"label": "test file (unlabeled)", "path": unl_path, "md5": tf["unlabeled"]["md5"], "size": tf["unlabeled"]["size"]}
    lab_entry = {"label": "test file (labels)", "path": lab_path, "md5": tf["labels"]["md5"], "size": tf["labels"]["size"]}
    if validate_only:
        hashes_before = verify_files(fz, check_md5=True)
        verify_files([unl_entry, lab_entry], check_md5=False)
    else:
        hashes_before = verify_files(fz + [unl_entry], check_md5=True)
        verify_files([lab_entry], check_md5=False)
    ev("hashes_verified_before_inference", frozen_files=len(fz), unlabeled_test_file=not validate_only, labels_file_size_only=True)
    infos = check_run_info(manifest, repo_root)
    ev("run_info_checked", runs=len(infos))
    env = environment_info(infos)

    # ---- dev reproduction (train/dev data only)
    dev_df = preprocess_frame(pd.DataFrame(_load_dev(data_root)), mode=exp["mode"])
    if len(dev_df) != exp["dev_size"]:
        raise ProtocolError(f"dev split has {len(dev_df)} tweets, expected {exp['dev_size']}")
    dev_repro = check_dev_reproduction(manifest, repo_root, dev_df, infos, predict=predict, log=log)
    ev("dev_reproduction_passed", runs=len(dev_repro))
    if validate_only:
        ev("validation_only_complete")
        return {"validated": True, "events": events, "environment": env, "dev_reproduction": dev_repro}

    # ---- the unlabeled file is parsed and validated before anything is created on disk (still no labels involved)
    test_df = preprocess_frame(read_unlabeled_test(unl_path, exp["n_test"]), mode=exp["mode"])
    uids, texts = test_df["uid"].tolist(), test_df["text_proc"].tolist()
    ev("unlabeled_test_parsed", tweets=len(uids))

    staging.mkdir(parents=True)
    stage_b_started = False
    try:
        # ===== Stage A: predictions (NO labels) =====
        pred_hashes = {}
        for c in manifest["checkpoints"]:
            model, seed = c["model"], c["seed"]
            preds = predict(repo_root / c["path"], texts, manifest["models"][model]["max_length"], manifest["inference"])
            validate_predictions(uids, uids, preds, f"{model}/seed{seed} predictions")
            fp = staging / "predictions" / pred_name(model, seed)
            atomic_write(fp, csv_text(["uid", "pred"], zip(uids, preds)))
            back_uids, back_preds = read_predictions_csv(fp)
            validate_predictions(uids, back_uids, back_preds, f"saved {fp.name}")
            pred_hashes[fp.name] = md5_file(fp)
            log(f"  predicted {model}/seed{seed}")
        atomic_write_json(staging / "predictions_manifest.json", {
            "written_before_labels_were_read": True, "utc": utc_now(), "n_test": len(uids), "prediction_files_md5": pred_hashes,
            "inputs_md5": hashes_before})
        ev("predictions_saved_and_validated", files=len(pred_hashes))

        # ===== Stage B: labels (first and only read of the labels file) =====
        stage_b_started = True
        labels_md5 = md5_file(lab_path)
        if labels_md5 != lab_entry["md5"]:
            raise HashMismatchError(f"integrity check failed: labels file MD5 {labels_md5} != expected {lab_entry['md5']}")
        labels = read_labels_strict(lab_path)
        ev("labels_read_after_predictions_were_saved", labels=len(labels))
        if set(labels) != set(uids):
            raise ProtocolError(f"label coverage mismatch: {len(set(uids) - set(labels))} test uids without label, "
                                f"{len(set(labels) - set(uids))} unexpected labels")
        per_run = {}
        for model, seed in RUNS:
            p_uids, p_preds = read_predictions_csv(staging / "predictions" / pred_name(model, seed))  # metrics from SAVED files
            y_true = join_by_uid(p_uids, labels)
            per_run[(model, seed)] = full_metrics(y_true, p_preds)
            # the project's common format (uid,label,pred) for CMI and error analysis
            atomic_write(staging / "predictions_labelled" / pred_name(model, seed).replace("_predictions", ""),
                         csv_text(["uid", "label", "pred"], zip(p_uids, y_true, p_preds)))
        aggregate = aggregate_runs(per_run)
        diffs = seed_differences(per_run)
        ev("metrics_computed")

        hashes_after = verify_files(fz + [unl_entry, lab_entry], check_md5=True)
        hashes_before = {**hashes_before, lab_entry["label"]: labels_md5}
        if hashes_after != hashes_before:
            raise HashMismatchError("an input file changed during the evaluation")
        ev("hashes_verified_after_evaluation", files=len(hashes_after))

        figures, fig_err = [], None
        if make_figs:
            try:
                figures = make_figure(staging / "figures", aggregate)
            except Exception as e:  # noqa: BLE001  (the figure is cosmetic; the numbers are in the JSON and the report)
                fig_err = repr(e)[:200]
        atomic_write_json(staging / "metrics_per_run.json",
                          {m: {str(s): {k: v for k, v in per_run[(m, s)].items() if k != "confusion"} for s in SEEDS} for m in MODELS})
        atomic_write_json(staging / "metrics_summary.json",
                          {m: {k: v for k, v in aggregate[m].items() if k != "confusion_summed"} for m in MODELS}
                          | {"seed_differences": diffs, "_description": "mean and sample standard deviation (ddof=1) over seeds 42, 43, 44",
                             "_primary_metrics": list(PRIMARY)})
        atomic_write_json(staging / "confusion_matrices.json", {
            "labels_order": LABELS, "rows": "true", "columns": "predicted",
            "per_run": {m: {str(s): per_run[(m, s)]["confusion"] for s in SEEDS} for m in MODELS},
            "summed_over_seeds": {m: aggregate[m]["confusion_summed"] for m in MODELS}})
        atomic_write_json(staging / "dev_reproduction.json", dev_repro)
        report = render_report({"aggregate": aggregate, "per_run": per_run, "n_test": len(uids), "finished_utc": utc_now(), "figures": figures,
                                "figures_error": fig_err, "manifest": manifest, "differences": diffs, "dev_reproduction": dev_repro})
        atomic_write(staging / "transformer_final_test_report.md", report)
        ev("outputs_written", figures=len(figures))
        out_hashes = {str(p.relative_to(staging)).replace("\\", "/"): md5_file(p) for p in sorted(staging.rglob("*")) if p.is_file()}
        atomic_write_json(staging / "integrity_manifest.json", {
            "protocol": PROTOCOL_DOC, "frozen_manifest": str(manifest_path), "frozen_manifest_md5": md5_file(manifest_path),
            "environment": env, "events": events,
            "inputs": {"md5_before_and_after_identical": True, "md5": hashes_after, "labels_file_first_read_after_predictions_saved": True},
            "outputs_md5": out_hashes})
        os.replace(staging, out_dir)
        ev("published", out_dir=str(out_dir))
    except BaseException:
        if staging.exists():
            if stage_b_started:
                failed = out_dir.with_name(f"{out_dir.name}.failed_{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}")
                os.replace(staging, failed)
                log(f"evaluation did not finish after the labels file was read; partial outputs kept in {failed}")
            else:
                shutil.rmtree(staging, ignore_errors=True)
                log("evaluation stopped before the labels file was touched; staging folder removed (a rerun is possible after fixing the problem)")
        raise
    return {"validated": True, "per_run": per_run, "aggregate": aggregate, "differences": diffs, "events": events, "out_dir": out_dir,
            "environment": env, "dev_reproduction": dev_repro}


def _load_dev(data_root: Path) -> pd.DataFrame:
    from src.data import load_split

    return load_split("dev", root=data_root)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Final single test-set evaluation of the six frozen transformer checkpoints (see the protocol).")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--validate-only", action="store_true", help="pre-flight checks and dev reproduction only: no test file opened, nothing written")
    g.add_argument("--confirm-final-test", action="store_true", help="run the real, one-time test evaluation")
    ap.add_argument("--manifest", default=None, help=f"frozen manifest (default {DEFAULT_MANIFEST})")
    ap.add_argument("--data-root", default=None, help="folder with the official SentiMix files (default: the project's data folder)")
    ap.add_argument("--out-dir", default=None, help=f"new output folder (default {DEFAULT_OUT_DIR}); never overwritten")
    ap.add_argument("--repo-root", default=str(ROOT), help="root that the manifest's relative paths refer to (for testing)")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args(argv)
    repo = Path(a.repo_root)
    try:
        res = evaluate(a.manifest or repo / DEFAULT_MANIFEST, repo_root=repo, data_root=a.data_root, out_dir=a.out_dir,
                       validate_only=a.validate_only, confirm=a.confirm_final_test, make_figs=not a.no_figures)
    except (ProtocolError, FileExistsError) as e:
        print(f"ERROR ({type(e).__name__}): {e}", file=sys.stderr)
        return 1
    print("VALIDATION OK: no test file was read." if a.validate_only else f"DONE: results in {res['out_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
