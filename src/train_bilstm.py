"""Train and evaluate the word + character BiLSTM on the SentiMix Hinglish data.

Run from the repo root (see docs/bilstm.md):

    python -m src.train_bilstm --mode normalized --seeds 42 43 44
    python -m src.train_bilstm --mode raw        --seeds 42 43 44
    python -m src.train_bilstm --mode clean      --seeds 42 43 44

Rules this script follows
- Train on the official TRAIN split; early-stop and report on the official DEV split.
- Vocabularies are fitted on TRAIN only (src.bilstm.TextEncoder.fit), separately for each mode.
- Length limits (max words per tweet, max characters per word) are ONE pair of numbers shared by
  raw, clean and normalized: the maximum over the three modes of the limit derived from the TRAIN
  split (see `derive_shared_limits`), so truncation is identical in all modes.
- The TEST split is never loaded here. The final one-time test run is a separate step.
- The same config and the same seeds should be used for every `--mode` (fair comparison).
- Metrics come from src.metrics.compute_metrics (weighted F1 is the official score).

Outputs (under results/ unless --out-dir is given):
  predictions/bilstm_<run>_seed<N>_dev.csv   uid,label,pred
  bilstm_<run>_dev_scores.json               config, limits, per-seed metrics, mean/std over seeds
  experiment_log.csv                         one appended row per seed
  checkpoints/bilstm_<run>_seed<N>.pt        best-dev-epoch model + vocabularies + config + metadata
                                             (src.bilstm; git-ignored)

Existing results are protected (see `run_experiment`): the run stops BEFORE training if any of its outputs
(or log rows) already exist, unless --overwrite is given; outputs are staged and published only after all
seeds finished, so a failed run never touches earlier results; --overwrite moves old files to
<out-dir>/archive/<timestamp>/ instead of deleting them; experiment_log.csv is append-only.
"""
from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.bilstm import (ID2LABEL, POOLINGS, BiLSTMClassifier, SentimentDataset, TextEncoder, checkpoint_path, derive_length,
                        make_batch, save_checkpoint)
from src.metrics import LABELS, compute_metrics, confusion
from src.preprocess import PRESETS, load_processed_split, preprocess_text, tokenize
from src.seed import DEFAULT_SEED, set_seed

ROOT = Path(__file__).resolve().parents[1]
TEXT_COL = "text_proc"  # column added by src.preprocess (the text the model reads)
RAW_COL = "text"        # the original text from src.data; the shared length limits are derived from it
DATA_VERSIONS = {  # wording for the experiment log
    "raw": "raw (tokens joined by spaces)",
    "clean": "clean (src.preprocess: URLs, mentions, hashtags, contractions)",
    "normalized": "normalized (src.preprocess: clean + repeated letters + Hinglish spellings)",
}


# ----------------------------------------------------------------- config ----
@dataclass
class BiLSTMConfig:
    """All settings of one experiment. Change them via a JSON file (--config) or --set key=value."""

    mode: str = "normalized"            # raw | clean | normalized  (src.preprocess.PRESETS)
    # architecture
    use_word: bool = True
    use_char: bool = True
    word_dim: int = 100
    char_dim: int = 32
    char_filters: int = 32               # filters per kernel size
    char_kernels: List[int] = field(default_factory=lambda: [2, 3, 4])
    hidden_size: int = 128
    num_layers: int = 1
    dropout: float = 0.4
    pooling: str = "maxmean"            # maxmean | mean | max | last
    # vocabulary and lengths (all fitted on train only)
    word_min_freq: int = 2
    char_min_freq: int = 2
    lowercase_words: bool = True
    lowercase_chars: bool = False
    max_words: Optional[int] = None      # None -> derived from train, shared by all modes (derive_shared_limits)
    max_chars: Optional[int] = None      # None -> derived from train, shared by all modes (derive_shared_limits)
    chars_percentile: float = 99.9
    # optimisation
    batch_size: int = 64
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 5.0
    patience: int = 4                    # stop after this many epochs without a dev weighted-F1 gain
    device: str = "auto"                 # auto | cpu | cuda

    def validate(self) -> "BiLSTMConfig":
        """Raise ValueError if a setting cannot work (unknown mode, both inputs off, size below 1, ...).
        This way a typo fails at the start, not after training. Returns self."""
        if self.mode not in PRESETS:
            raise ValueError(f"mode must be one of {list(PRESETS)}, got {self.mode!r}")
        if self.pooling not in POOLINGS:
            raise ValueError(f"pooling must be one of {POOLINGS}, got {self.pooling!r}")
        if not (self.use_word or self.use_char):
            raise ValueError("use_word and use_char cannot both be False")
        for name in ("word_dim", "char_dim", "char_filters", "hidden_size", "num_layers", "batch_size", "epochs"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        for name in ("max_words", "max_chars"):
            if getattr(self, name) is not None and getattr(self, name) < 1:
                raise ValueError(f"{name} must be null (derive from train) or >= 1")
        if not self.char_kernels or min(self.char_kernels) < 1:
            raise ValueError("char_kernels must be a non-empty list of positive ints")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        return self

    def with_overrides(self, overrides: dict) -> "BiLSTMConfig":
        """Copy of the config with some values changed (for --set and --mode). Unknown keys raise an error."""
        known = {f.name for f in fields(self)}
        bad = set(overrides) - known
        if bad:
            raise ValueError(f"unknown config keys {sorted(bad)}; valid keys: {sorted(known)}")
        return replace(self, **overrides).validate()

    @classmethod
    def from_json(cls, path) -> "BiLSTMConfig":
        """Defaults plus the values from a JSON file."""
        return cls().with_overrides(json.loads(Path(path).read_text(encoding="utf-8")))

    def short(self) -> str:
        """One-line summary for experiment_log.csv."""
        return (f"mode={self.mode}; word={self.use_word}(d{self.word_dim}); char={self.use_char}"
                f"(cnn d{self.char_dim} f{self.char_filters} k{self.char_kernels}); "
                f"bilstm h{self.hidden_size} x{self.num_layers}; pool={self.pooling}; drop={self.dropout}; "
                f"min_freq={self.word_min_freq}/{self.char_min_freq}; lr={self.lr}; bs={self.batch_size}; "
                f"max_epochs={self.epochs}; patience={self.patience}")


# ----------------------------------------------------------- one training ----
def _resolve_device(name: str) -> torch.device:
    """auto = use the GPU if there is one, otherwise the CPU."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> List[int]:
    """Predicted label ids for every row of `loader` (no shuffling, eval mode)."""
    # evaluation mode: dropout is off and nothing is learned
    model.eval()
    preds: List[int] = []
    for b in loader:
        logits = model(b["words"].to(device), b["chars"].to(device), b["lengths"])
        preds.extend(logits.argmax(dim=1).tolist())
    return preds


def _evaluate(model, loader, device, true_labels):
    """Predict a whole split and score it. Returns (predicted labels, metrics)."""
    pred = [ID2LABEL[i] for i in predict(model, loader, device)]
    return pred, compute_metrics(true_labels, pred)


def build_model(cfg: BiLSTMConfig, encoder: TextEncoder) -> BiLSTMClassifier:
    """The model for `cfg`, sized to `encoder`'s vocabularies (also used when loading a checkpoint)."""
    return BiLSTMClassifier(
        len(encoder.word_vocab), len(encoder.char_vocab), len(LABELS), use_word=cfg.use_word,
        use_char=cfg.use_char, word_dim=cfg.word_dim, char_dim=cfg.char_dim, char_filters=cfg.char_filters,
        char_kernels=cfg.char_kernels, hidden_size=cfg.hidden_size, num_layers=cfg.num_layers,
        dropout=cfg.dropout, pooling=cfg.pooling)


def derive_shared_limits(raw_train_texts, chars_percentile: float = 99.9, words_percentile: float = 100.0) -> dict:
    """Length limits that are the same for raw, clean and normalized, derived from TRAIN text only.

    For each preprocessing mode, take the train tweets (original text), preprocess them with that mode and
    measure: longest tweet in words (words_percentile, default the maximum) and the 99.9th percentile of
    word length in characters. The shared limit is the MAXIMUM over the three modes, so no mode truncates
    train tweets more than it would with its own limit, and all modes truncate identically.

    Why not each mode's own limit: they differ (on the real data max_words is 56 / 53 / 53 and max_chars
    16 / 18 / 17 for raw / clean / normalized, because cleaning shortens tweets but rejoining contractions
    lengthens words). Why not raw only: it would cut more words in clean and normalized.
    Only train text is used; dev and test never influence the limits.
    Returns {"max_words", "max_chars", "per_mode": {mode: {"max_words", "max_chars"}}}.
    """
    # measure the train tweets under each preprocessing mode
    per_mode = {}
    for mode, opts in PRESETS.items():
        toks = [tokenize(preprocess_text(t, **opts)) for t in raw_train_texts]
        per_mode[mode] = {"max_words": derive_length([len(t) for t in toks], words_percentile),
                          "max_chars": derive_length([len(w) for t in toks for w in t], chars_percentile)}
    # the shared limit is the largest of the three, so every mode is cut the same way
    return {"max_words": max(v["max_words"] for v in per_mode.values()),
            "max_chars": max(v["max_chars"] for v in per_mode.values()), "per_mode": per_mode}


def resolve_limits(train_df: pd.DataFrame, cfg: BiLSTMConfig) -> dict:
    """The (max_words, max_chars) used for a run: explicit config values win, otherwise the shared
    train-derived limits. Needs the original `text` column of the TRAIN frame (from src.data)."""
    if cfg.max_words is not None and cfg.max_chars is not None:
        return {"max_words": cfg.max_words, "max_chars": cfg.max_chars, "source": "config", "per_mode": None}
    if RAW_COL not in train_df.columns:
        raise ValueError(f"train frame needs the original '{RAW_COL}' column to derive limits shared by all modes "
                         "(use load_processed_split), or set max_words and max_chars in the config")
    d = derive_shared_limits(list(train_df[RAW_COL]), cfg.chars_percentile)
    return {"max_words": cfg.max_words or d["max_words"], "max_chars": cfg.max_chars or d["max_chars"],
            "source": "max over raw/clean/normalized of train-derived limits", "per_mode": d["per_mode"]}


def train_one_seed(train_df: pd.DataFrame, dev_df: pd.DataFrame, cfg: BiLSTMConfig, seed: int,
                   verbose: bool = True, limits: Optional[dict] = None,
                   checkpoint_file: Optional[Path] = None) -> dict:
    """Train with one seed and return the best-dev-epoch result.

    Needs columns `text_proc` and `label` (and `uid` for saving predictions) in both frames, plus the
    original `text` column in the train frame unless `limits` or max_words/max_chars are given.
    The epoch with the best dev weighted F1 is kept (early stopping on dev). That dev score is
    therefore slightly optimistic; the final unbiased number comes from the one-time test run.
    `limits`: result of `resolve_limits` (computed once per experiment by `run_experiment`).
    `checkpoint_file`: if given, the best model is saved there (see src.bilstm).
    """
    # stop early if a setting is invalid
    cfg.validate()
    t0 = time.time()
    # fix the random numbers so the run can be repeated
    set_seed(seed)
    device = _resolve_device(cfg.device)

    train_texts, dev_texts = list(train_df[TEXT_COL]), list(dev_df[TEXT_COL])
    train_labels, dev_labels = list(train_df["label"]), list(dev_df["label"])
    limits = limits or resolve_limits(train_df, cfg)

    # everything learned from data comes from TRAIN only
    encoder = TextEncoder.fit(
        train_texts, word_min_freq=cfg.word_min_freq, char_min_freq=cfg.char_min_freq,
        max_words=limits["max_words"], max_chars=limits["max_chars"], chars_percentile=cfg.chars_percentile,
        lowercase_words=cfg.lowercase_words, lowercase_chars=cfg.lowercase_chars)
    truncation = {"train": encoder.count_truncated(train_texts), "dev": encoder.count_truncated(dev_texts)}
    train_ds = SentimentDataset(encoder, train_texts, train_labels)
    dev_ds = SentimentDataset(encoder, dev_texts, dev_labels)

    gen = torch.Generator()
    gen.manual_seed(seed)  # makes the shuffling order repeatable
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=make_batch, generator=gen)
    dev_loader = DataLoader(dev_ds, batch_size=256, shuffle=False, collate_fn=make_batch)

    # the model is sized to the vocabularies built from the training data
    model = build_model(cfg, encoder).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    # Adam optimiser; weight_decay is a small penalty against overfitting
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # standard loss for choosing one of several classes
    loss_fn = nn.CrossEntropyLoss()

    # remember the best epoch so far (judged by dev weighted F1)
    best = {"f1": -1.0, "epoch": 0, "state": None}
    history, bad_epochs = [], 0
    # training loop: one pass over all training tweets per epoch
    for epoch in range(1, cfg.epochs + 1):
        # training mode: dropout is on
        model.train()
        total, n = 0.0, 0
        for b in train_loader:
            # one step: clear old gradients, forward pass, loss, backward pass, clip, update
            opt.zero_grad()
            logits = model(b["words"].to(device), b["chars"].to(device), b["lengths"])
            loss = loss_fn(logits, b["labels"].to(device))
            loss.backward()
            # clipping stops very large gradient steps
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            total += loss.item() * len(b["labels"])
            n += len(b["labels"])
        # after each epoch, score the model on the development set
        _, m = _evaluate(model, dev_loader, device, dev_labels)
        history.append({"epoch": epoch, "train_loss": round(total / n, 4), "dev_weighted_f1": round(m["weighted_f1"], 4),
                        "dev_macro_f1": round(m["macro_f1"], 4), "dev_accuracy": round(m["accuracy"], 4)})
        if verbose:
            h = history[-1]
            print(f"  seed {seed} epoch {epoch:2d}  loss {h['train_loss']:.4f}  dev wF1 {h['dev_weighted_f1']:.4f}"
                  f"  macroF1 {h['dev_macro_f1']:.4f}  acc {h['dev_accuracy']:.4f}")
        # new best epoch: keep a copy of the weights and reset the patience counter
        if m["weighted_f1"] > best["f1"]:
            best = {"f1": m["weighted_f1"], "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
            bad_epochs = 0
        else:
            bad_epochs += 1
            # early stopping: no improvement for `patience` epochs in a row
            if bad_epochs >= cfg.patience:
                break

    model.load_state_dict(best["state"])  # back to the best epoch
    pred, metrics = _evaluate(model, dev_loader, device, dev_labels)
    saved = None
    if checkpoint_file is not None:  # only train-fitted objects, settings and dev metrics are stored
        saved = str(save_checkpoint(
            checkpoint_file, model, encoder, cfg, seed=seed, best_epoch=best["epoch"], metrics=metrics,
            extra={"limits": {"max_words": limits["max_words"], "max_chars": limits["max_chars"],
                              "source": limits["source"]},
                   "train_size": len(train_df), "dev_size": len(dev_df), "n_params": n_params,
                   "epochs_run": len(history), "truncation": truncation}))
    return {
        "seed": seed, "best_epoch": best["epoch"], "epochs_run": len(history), "metrics": metrics,
        "confusion": confusion(dev_labels, pred), "history": history, "pred": pred,
        "n_params": n_params, "word_vocab_size": len(encoder.word_vocab), "char_vocab_size": len(encoder.char_vocab),
        "max_words": encoder.max_words, "max_chars": encoder.max_chars, "truncation": truncation,
        "checkpoint": saved, "seconds": round(time.time() - t0, 1),
    }


# ---------------------------------------------------- experiment + logging ----
LOG_HEADER = ["date", "who", "model", "data_version", "settings", "seed", "dev_score", "notes", "lesson_learned"]
LOG_NAME = "experiment_log.csv"


# ---- output protection (never overwrite earlier results by accident) ----
def append_log_rows(path: Path, rows: List[list]) -> None:
    """Append rows to experiment_log.csv without ever changing its existing bytes.

    The new content (old bytes + new rows) is written to a temporary file next to the log and swapped in
    with os.replace, so an interruption cannot leave a truncated or half-written log. The header is written
    only if the file does not exist yet. Existing rows are never edited or removed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    old = path.read_bytes() if path.exists() else b""
    buf = io.StringIO(newline="")
    w = csv.writer(buf)  # default CRLF line ending, the same as the existing log
    if not old:
        w.writerow(LOG_HEADER)
    for r in rows:
        w.writerow(r)
    prefix = old if (not old or old.endswith(b"\n")) else old + b"\r\n"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(prefix + buf.getvalue().encode("utf-8"))
    os.replace(tmp, path)


def log_experiment(path: Path, row: list) -> None:
    """Append one row to experiment_log.csv (creating it with its header if missing)."""
    append_log_rows(path, [row])


def log_has_run(path: Path, run_name: str, seed: int) -> bool:
    """True if experiment_log.csv already holds a row written by run `run_name` for `seed`
    (the notes of every row written by this script start with `run=<run_name>;`)."""
    path = Path(path)
    if not path.exists():
        return False
    key = f"run={run_name};"
    with open(path, encoding="utf-8", newline="") as f:
        return any(r.get("seed") == str(seed) and (r.get("notes") or "").startswith(key) for r in csv.DictReader(f))


def prediction_path(out_dir, run_name: str, seed: int) -> Path:
    return Path(out_dir) / "predictions" / f"bilstm_{run_name}_seed{seed}_dev.csv"


def scores_path(out_dir, run_name: str) -> Path:
    return Path(out_dir) / f"bilstm_{run_name}_dev_scores.json"


def find_existing_outputs(out_dir, run_name: str, seeds: List[int], save_checkpoints: bool = True) -> List[str]:
    """Everything this run would write that already exists: files, and log rows for this run name and seed."""
    out = Path(out_dir)
    planned = [scores_path(out, run_name)]
    for s in seeds:
        planned.append(prediction_path(out, run_name, s))
        if save_checkpoints:
            planned.append(checkpoint_path(out, run_name, s))
    # anything that already exists must not be overwritten
    found = [str(p) for p in planned if p.exists()]
    log = out / LOG_NAME
    found += [f"{log} (row of run '{run_name}', seed {s})" for s in seeds if log_has_run(log, run_name, s)]
    return found


def _stamp() -> str:
    """Date and time as text, used to name the staging and archive folders."""
    return datetime.now().strftime("%Y%m%d-%H%M%S-%f")


def _publish(stage: Path, out_dir: Path, log_rows: List[list], overwrite: bool, stamp: str) -> List[str]:
    """Move every staged file to the same relative place in `out_dir`, then append the log rows.

    A file already at a destination is first MOVED (never deleted) to <out_dir>/archive/<stamp>/<same path>,
    and only if overwrite=True; otherwise nothing is moved and FileExistsError is raised (this re-checks
    right before publishing, in case something appeared while the run was training). Returns the archived paths."""
    files = sorted(p for p in stage.rglob("*") if p.is_file())
    dests = [out_dir / p.relative_to(stage) for p in files]
    # refuse to overwrite unless overwrite=True
    clash = [str(d) for d in dests if d.exists()]
    if clash and not overwrite:
        raise FileExistsError(f"outputs appeared while the run was training, nothing was published: {clash}")
    archived = []
    for src, dest in zip(files, dests):
        if dest.exists():
            # overwritten files are moved to archive/, never deleted
            old = out_dir / "archive" / stamp / dest.relative_to(out_dir)
            old.parent.mkdir(parents=True, exist_ok=True)
            os.replace(dest, old)
            archived.append(str(old))
        dest.parent.mkdir(parents=True, exist_ok=True)
        os.replace(src, dest)
    append_log_rows(out_dir / LOG_NAME, log_rows)  # last: a log row means the run completed
    shutil.rmtree(stage, ignore_errors=True)
    return archived


def run_experiment(train_df, dev_df, cfg: BiLSTMConfig, seeds: List[int], out_dir: Optional[Path] = None,
                   run_name: Optional[str] = None, who: str = "Ekta", verbose: bool = True,
                   save_checkpoints: bool = True, overwrite: bool = False) -> dict:
    """Run `train_one_seed` for every seed, summarise, and (if out_dir is given) save everything.

    With out_dir=None nothing is written (used for tests and quick debugging).
    The length limits are derived once from the train split (shared by all modes) and reused for every seed.

    Output protection (with out_dir):
    - BEFORE training, every output of this run is checked (scores JSON, per-seed predictions and checkpoints,
      and log rows of this run name and seed). If any exists, FileExistsError lists them and nothing is trained
      or written, unless overwrite=True. Change `run_name` to keep both.
    - The run writes into a hidden staging folder and publishes it only after ALL seeds finished, so a failure
      or interruption never touches existing results. The staged work of a failed run is kept in
      <out_dir>/failed_runs/<run>_<timestamp>/ (finished seeds are not lost).
    - overwrite=True moves existing files to <out_dir>/archive/<timestamp>/ before publishing (never deletes)
      and marks the new log rows with `rerun=yes`; old log rows stay (the log is append-only).
    Returns {"config", "limits", "per_seed": [...], "summary": {...}, "archived": [...]}.
    """
    cfg.validate()
    run_name = run_name or f"bilstm_{cfg.mode}"
    out = Path(out_dir) if out_dir is not None else None
    existing = find_existing_outputs(out, run_name, seeds, save_checkpoints) if out is not None else []
    if existing and not overwrite:
        raise FileExistsError(
            f"outputs of run '{run_name}' already exist, nothing was trained or written:\n  - " + "\n  - ".join(existing)
            + "\nUse another --run-name, or --overwrite to replace them "
              "(old files are then moved to <out-dir>/archive/<timestamp>/, never deleted).")
    limits = resolve_limits(train_df, cfg)
    if verbose:
        print(f"[{run_name}] length limits shared by all modes ({limits['source']}): "
              f"max_words={limits['max_words']} max_chars={limits['max_chars']}")

    stamp = _stamp()
    stage = out / f".staging_{run_name}_{stamp}" if out is not None else None
    per_seed, log_rows, archived = [], [], []
    try:
        if stage is not None:
            (stage / "predictions").mkdir(parents=True)
        for seed in seeds:
            if verbose:
                print(f"[{run_name}] seed {seed}")
            res = train_one_seed(train_df, dev_df, cfg, seed, verbose=verbose, limits=limits,
                                 checkpoint_file=checkpoint_path(stage, run_name, seed)
                                 if stage is not None and save_checkpoints else None)
            if res.get("checkpoint"):
                res["checkpoint"] = str(checkpoint_path(out, run_name, seed))  # where it will be after publishing
            per_seed.append(res)
            m = res["metrics"]
            if stage is not None:
                pd.DataFrame({"uid": dev_df["uid"].values, "label": dev_df["label"].values,
                              "pred": res["pred"]}).to_csv(prediction_path(stage, run_name, seed), index=False)
                log_rows.append(
                    [date.today().isoformat(), who, ("bilstm word+char (cnn)" if cfg.use_char else "bilstm word-only"), DATA_VERSIONS[cfg.mode], cfg.short(), seed,
                     f"{m['weighted_f1']:.4f}",
                     f"run={run_name}; " + ("rerun=yes; " if existing else "") +
                     f"macro_f1={m['macro_f1']:.4f}; accuracy={m['accuracy']:.4f}; best_epoch={res['best_epoch']}; "
                     f"max_words={res['max_words']}; max_chars={res['max_chars']}; "
                     f"dev_truncated_tweets={res['truncation']['dev']['tweets']}; params={res['n_params']}; "
                     "early-stopped on dev",
                     ""])
            if verbose:
                print(f"[{run_name}] seed {seed} best epoch {res['best_epoch']}: wF1 {m['weighted_f1']:.4f} "
                      f"macroF1 {m['macro_f1']:.4f} acc {m['accuracy']:.4f}")

        summary = {}
        for key in ("weighted_f1", "macro_f1", "accuracy"):
            vals = np.array([r["metrics"][key] for r in per_seed])
            summary[key] = {"mean": round(float(vals.mean()), 4),
                            "std": round(float(vals.std(ddof=1)) if len(vals) > 1 else 0.0, 4)}
        result = {"run_name": run_name, "config": asdict(cfg), "limits": limits, "seeds": list(seeds),
                  "summary": summary, "per_seed": [{k: v for k, v in r.items() if k != "pred"} for r in per_seed]}
        if stage is not None:
            scores_path(stage, run_name).write_text(json.dumps(result, indent=2), encoding="utf-8")
            archived = _publish(stage, out, log_rows, overwrite, stamp)
    except BaseException:
        if stage is not None and stage.exists():  # keep the work of a failed run; existing results stay untouched
            failed = out / "failed_runs" / f"{run_name}_{stamp}"
            failed.parent.mkdir(parents=True, exist_ok=True)
            os.replace(stage, failed)
            print(f"[{run_name}] run did not finish; its partial outputs are kept in {failed}. "
                  "Existing results were not touched.")
        raise
    result["archived"] = archived
    if verbose:
        s = summary
        print(f"[{run_name}] over seeds {list(seeds)}: wF1 {s['weighted_f1']['mean']:.4f} +/- {s['weighted_f1']['std']:.4f}, "
              f"macroF1 {s['macro_f1']['mean']:.4f} +/- {s['macro_f1']['std']:.4f}, "
              f"acc {s['accuracy']['mean']:.4f} +/- {s['accuracy']['std']:.4f}")
        if archived:
            print(f"[{run_name}] {len(archived)} earlier file(s) moved to {out / 'archive' / stamp}")
    return result


# -------------------------------------------------------------------- CLI ----
def _parse_set(items: Optional[List[str]]) -> dict:
    """Turn ["lr=0.002", "pooling=last"] into {"lr": 0.002, "pooling": "last"}."""
    out = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        k, v = item.split("=", 1)
        try:
            out[k.strip()] = json.loads(v)  # numbers, true/false, null, [2,3]
        except json.JSONDecodeError:
            out[k.strip()] = v              # plain strings such as pooling=last
    return out


def main(argv=None) -> dict:
    """Read the settings, load train and dev, and run the experiment."""
    p = argparse.ArgumentParser(description="Train the word+char BiLSTM (train split) and evaluate on dev.")
    p.add_argument("--config", help="JSON file with BiLSTMConfig settings (default: built-in defaults)")
    p.add_argument("--mode", choices=list(PRESETS), help="text version: raw | clean | normalized")
    p.add_argument("--seeds", type=int, nargs="+", default=[DEFAULT_SEED])
    p.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a config value, e.g. --set hidden_size=256")
    p.add_argument("--run-name", help="name used in output files (default: bilstm_<mode>)")
    p.add_argument("--who", default="Ekta", help="name written to experiment_log.csv")
    p.add_argument("--out-dir", default=str(ROOT / "results"))
    p.add_argument("--data-root", help="folder with the SentiMix Hinglish files (default: src.data.DEFAULT_ROOT)")
    p.add_argument("--limit", type=int, help="DEBUG: use only the first N train and dev tweets; nothing is saved or logged")
    p.add_argument("--no-checkpoints", action="store_true", help="do not save the best model of each seed")
    p.add_argument("--overwrite", action="store_true",
                   help="replace the outputs of an earlier run with the same run name; default: stop before training "
                        "if any exist. Old files are moved to <out-dir>/archive/<timestamp>/, never deleted")
    args = p.parse_args(argv)

    cfg = BiLSTMConfig.from_json(args.config) if args.config else BiLSTMConfig()
    overrides = _parse_set(args.set)
    if args.mode:
        overrides["mode"] = args.mode
    cfg = cfg.with_overrides(overrides)

    # only train and dev are ever loaded; test is deliberately not touched here
    train_df = load_processed_split("train", mode=cfg.mode, root=args.data_root)
    dev_df = load_processed_split("dev", mode=cfg.mode, root=args.data_root)
    out_dir = Path(args.out_dir)
    if args.limit:
        train_df, dev_df = train_df.head(args.limit), dev_df.head(args.limit)
        out_dir = None
        print(f"DEBUG run on {len(train_df)} train / {len(dev_df)} dev tweets: results are NOT saved or logged.")
    return run_experiment(train_df, dev_df, cfg, args.seeds, out_dir=out_dir, run_name=args.run_name, who=args.who,
                          save_checkpoints=not args.no_checkpoints, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
