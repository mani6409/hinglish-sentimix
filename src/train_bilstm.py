"""Train and evaluate the word + character BiLSTM on the SentiMix Hinglish data.

Run from the repo root (see docs/bilstm.md):

    python -m src.train_bilstm --mode normalized --seeds 42 43 44
    python -m src.train_bilstm --mode raw        --seeds 42 43 44
    python -m src.train_bilstm --mode clean      --seeds 42 43 44

Rules this script follows
- Train on the official TRAIN split; early-stop and report on the official DEV split.
- Vocabularies are fitted on TRAIN only (src.bilstm_data.TextEncoder.fit), separately for each mode.
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
                                             (src.bilstm_checkpoint; git-ignored)
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import date
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.bilstm_checkpoint import checkpoint_path, save_checkpoint
from src.bilstm_data import ID2LABEL, SentimentDataset, TextEncoder, derive_length, make_batch
from src.bilstm_model import POOLINGS, BiLSTMClassifier
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
        known = {f.name for f in fields(self)}
        bad = set(overrides) - known
        if bad:
            raise ValueError(f"unknown config keys {sorted(bad)}; valid keys: {sorted(known)}")
        return replace(self, **overrides).validate()

    @classmethod
    def from_json(cls, path) -> "BiLSTMConfig":
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
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> List[int]:
    """Predicted label ids for every row of `loader` (no shuffling, eval mode)."""
    model.eval()
    preds: List[int] = []
    for b in loader:
        logits = model(b["words"].to(device), b["chars"].to(device), b["lengths"])
        preds.extend(logits.argmax(dim=1).tolist())
    return preds


def _evaluate(model, loader, device, true_labels):
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
    per_mode = {}
    for mode, opts in PRESETS.items():
        toks = [tokenize(preprocess_text(t, **opts)) for t in raw_train_texts]
        per_mode[mode] = {"max_words": derive_length([len(t) for t in toks], words_percentile),
                          "max_chars": derive_length([len(w) for t in toks for w in t], chars_percentile)}
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
    `checkpoint_file`: if given, the best model is saved there (see src.bilstm_checkpoint).
    """
    cfg.validate()
    t0 = time.time()
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

    model = build_model(cfg, encoder).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    best = {"f1": -1.0, "epoch": 0, "state": None}
    history, bad_epochs = [], 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total, n = 0.0, 0
        for b in train_loader:
            opt.zero_grad()
            logits = model(b["words"].to(device), b["chars"].to(device), b["lengths"])
            loss = loss_fn(logits, b["labels"].to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            total += loss.item() * len(b["labels"])
            n += len(b["labels"])
        _, m = _evaluate(model, dev_loader, device, dev_labels)
        history.append({"epoch": epoch, "train_loss": round(total / n, 4), "dev_weighted_f1": round(m["weighted_f1"], 4),
                        "dev_macro_f1": round(m["macro_f1"], 4), "dev_accuracy": round(m["accuracy"], 4)})
        if verbose:
            h = history[-1]
            print(f"  seed {seed} epoch {epoch:2d}  loss {h['train_loss']:.4f}  dev wF1 {h['dev_weighted_f1']:.4f}"
                  f"  macroF1 {h['dev_macro_f1']:.4f}  acc {h['dev_accuracy']:.4f}")
        if m["weighted_f1"] > best["f1"]:
            best = {"f1": m["weighted_f1"], "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
            bad_epochs = 0
        else:
            bad_epochs += 1
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


def log_experiment(path: Path, row: list) -> None:
    """Append one row to experiment_log.csv (creating it with its header if missing)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(LOG_HEADER)
        w.writerow(row)


def run_experiment(train_df, dev_df, cfg: BiLSTMConfig, seeds: List[int], out_dir: Optional[Path] = None,
                   run_name: Optional[str] = None, who: str = "Ekta", verbose: bool = True,
                   save_checkpoints: bool = True, overwrite_checkpoints: bool = False) -> dict:
    """Run `train_one_seed` for every seed, summarise, and (if out_dir is given) save everything.

    With out_dir=None nothing is written (used for tests and quick debugging).
    With out_dir, each seed's best model is saved to <out_dir>/checkpoints/bilstm_<run>_seed<N>.pt unless
    save_checkpoints=False. Existing checkpoint files are never overwritten silently: the run stops BEFORE
    training (FileExistsError) unless overwrite_checkpoints=True.
    The length limits are derived once from the train split (shared by all modes) and reused for every seed.
    Returns {"config", "limits", "per_seed": [...], "summary": {metric: {mean, std}}}.
    """
    cfg.validate()
    run_name = run_name or f"bilstm_{cfg.mode}"
    ckpt_files = {s: (checkpoint_path(out_dir, run_name, s) if out_dir is not None and save_checkpoints else None)
                  for s in seeds}
    existing = [str(p) for p in ckpt_files.values() if p is not None and p.exists()]
    if existing and not overwrite_checkpoints:
        raise FileExistsError(f"checkpoint(s) already exist, nothing was trained: {existing}. "
                              "Use another --run-name, or --overwrite-checkpoints to replace them.")
    limits = resolve_limits(train_df, cfg)
    if verbose:
        print(f"[{run_name}] length limits shared by all modes ({limits['source']}): "
              f"max_words={limits['max_words']} max_chars={limits['max_chars']}")
    per_seed = []
    for seed in seeds:
        if verbose:
            print(f"[{run_name}] seed {seed}")
        res = train_one_seed(train_df, dev_df, cfg, seed, verbose=verbose, limits=limits,
                             checkpoint_file=ckpt_files[seed])
        per_seed.append(res)
        if out_dir is not None:
            out_dir = Path(out_dir)
            pdir = out_dir / "predictions"
            pdir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"uid": dev_df["uid"].values, "label": dev_df["label"].values, "pred": res["pred"]}).to_csv(
                pdir / f"bilstm_{run_name}_seed{seed}_dev.csv", index=False)
            m = res["metrics"]
            log_experiment(
                out_dir / "experiment_log.csv",
                [date.today().isoformat(), who, "bilstm word+char (cnn)", DATA_VERSIONS[cfg.mode], cfg.short(), seed,
                 f"{m['weighted_f1']:.4f}",
                 f"macro_f1={m['macro_f1']:.4f}; accuracy={m['accuracy']:.4f}; best_epoch={res['best_epoch']}; "
                 f"max_words={res['max_words']}; max_chars={res['max_chars']}; "
                 f"dev_truncated_tweets={res['truncation']['dev']['tweets']}; params={res['n_params']}; "
                 "early-stopped on dev",
                 ""])
        if verbose:
            m = res["metrics"]
            print(f"[{run_name}] seed {seed} best epoch {res['best_epoch']}: wF1 {m['weighted_f1']:.4f} "
                  f"macroF1 {m['macro_f1']:.4f} acc {m['accuracy']:.4f}")

    summary = {}
    for key in ("weighted_f1", "macro_f1", "accuracy"):
        vals = np.array([r["metrics"][key] for r in per_seed])
        summary[key] = {"mean": round(float(vals.mean()), 4),
                        "std": round(float(vals.std(ddof=1)) if len(vals) > 1 else 0.0, 4)}
    result = {"run_name": run_name, "config": asdict(cfg), "limits": limits, "seeds": list(seeds), "summary": summary,
              "per_seed": [{k: v for k, v in r.items() if k != "pred"} for r in per_seed]}
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / f"bilstm_{run_name}_dev_scores.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if verbose:
        s = summary
        print(f"[{run_name}] over seeds {list(seeds)}: wF1 {s['weighted_f1']['mean']:.4f} +/- {s['weighted_f1']['std']:.4f}, "
              f"macroF1 {s['macro_f1']['mean']:.4f} +/- {s['macro_f1']['std']:.4f}, "
              f"acc {s['accuracy']['mean']:.4f} +/- {s['accuracy']['std']:.4f}")
    return result


# -------------------------------------------------------------------- CLI ----
def _parse_set(items: Optional[List[str]]) -> dict:
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
    p.add_argument("--overwrite-checkpoints", action="store_true",
                   help="replace existing checkpoint files (default: stop before training if any exists)")
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
                          save_checkpoints=not args.no_checkpoints, overwrite_checkpoints=args.overwrite_checkpoints)


if __name__ == "__main__":
    main()
