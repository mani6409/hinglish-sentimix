"""Shared pieces for fine-tuning any Hugging Face encoder (mBERT, XLM-R, MuRIL, IndicBERT, ...)
on three-class Hinglish sentiment.

Nothing in this file is specific to one model: the checkpoint is a setting (`model_name`), and every
model goes through the same tokenisation, length limit, batching, prediction and checkpoint code.
`src.train_transformer` uses these pieces for training; see docs/transformer_pipeline.md.

Sequence-length analysis (run once per model, TRAIN text only):

    python -m src.transformer_utils --models mbert xlmr --mode clean

writes results/transformer_token_lengths.json with subword-length percentiles per model and the
max_length the training script would derive.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from src.metrics import LABELS
from src.preprocess import PRESETS

ROOT = Path(__file__).resolve().parents[1]

LABEL2ID = {lab: i for i, lab in enumerate(LABELS)}  # same order as src.metrics.LABELS
ID2LABEL = {i: lab for lab, i in LABEL2ID.items()}

# Short names accepted wherever a checkpoint is expected; anything else is passed to
# `from_pretrained` unchanged (a Hub id or a local folder).
MODEL_ALIASES = {
    "mbert": "bert-base-multilingual-cased",
    "xlmr": "xlm-roberta-base",
    "xlmr-large": "xlm-roberta-large",
    "muril": "google/muril-base-cased",
    "indicbert": "ai4bharat/indic-bert",
}


def resolve_model_name(name: str) -> str:
    return MODEL_ALIASES.get(name, name)


def model_slug(name: str) -> str:
    """A file-name friendly short name: the alias if one maps to `name`, else the last path part."""
    for alias, full in MODEL_ALIASES.items():
        if name in (alias, full):
            return alias
    return Path(name.rstrip("/\\")).name.replace(".", "-") or "model"


# ----------------------------------------------------------------- config ----
@dataclass
class TransformerConfig:
    """All settings of one fine-tuning experiment. Change them via a JSON file (--config) or --set key=value.

    The defaults follow common practice for base-size encoders on short texts (Devlin et al. 2019;
    Conneau et al. 2020): AdamW, linear warm-up then decay, lr in [1e-5, 5e-5], a few epochs.
    """

    model_name: str = "mbert"           # alias (MODEL_ALIASES), Hub id or local folder
    mode: str = "clean"                 # text version: raw | clean | normalized (src.preprocess.PRESETS)
    # sequence length
    max_length: Optional[int] = None    # None -> derived from TRAIN subword lengths (derive_max_length)
    length_percentile: float = 100.0    # percentile of train lengths the derived limit must cover
    # optimisation
    batch_size: int = 32
    eval_batch_size: int = 128
    epochs: int = 4
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    grad_clip: float = 1.0
    grad_accum_steps: int = 1
    patience: int = 2                   # stop after this many epochs without a dev weighted-F1 gain
    dropout: Optional[float] = None     # None -> the checkpoint's own dropout
    # hardware
    device: str = "auto"                # auto | cpu | cuda
    fp16: bool = True                   # mixed precision on CUDA only; ignored on CPU

    def validate(self) -> "TransformerConfig":
        if self.mode not in PRESETS:
            raise ValueError(f"mode must be one of {list(PRESETS)}, got {self.mode!r}")
        if not self.model_name:
            raise ValueError("model_name must be set")
        for name in ("batch_size", "eval_batch_size", "epochs", "grad_accum_steps", "patience"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.max_length is not None and self.max_length < 8:
            raise ValueError("max_length must be null (derive from train) or >= 8")
        if not 0 < self.length_percentile <= 100:
            raise ValueError("length_percentile must be in (0, 100]")
        if not self.lr > 0:
            raise ValueError("lr must be > 0")
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if self.dropout is not None and not 0 <= self.dropout < 1:
            raise ValueError("dropout must be null or in [0, 1)")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu or cuda")
        return self

    def with_overrides(self, overrides: dict) -> "TransformerConfig":
        known = {f.name for f in fields(self)}
        bad = set(overrides) - known
        if bad:
            raise ValueError(f"unknown config keys {sorted(bad)}; valid keys: {sorted(known)}")
        return replace(self, **overrides).validate()

    @classmethod
    def from_json(cls, path) -> "TransformerConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data = {k: v for k, v in data.items() if not k.startswith("_")}  # "_comment" keys are documentation
        return cls().with_overrides(data)

    def short(self) -> str:
        """One-line summary for experiment_log.csv."""
        return (f"model={resolve_model_name(self.model_name)}; mode={self.mode}; max_len={self.max_length or 'derived'}; "
                f"lr={self.lr}; bs={self.batch_size}x{self.grad_accum_steps}; max_epochs={self.epochs}; "
                f"warmup={self.warmup_ratio}; wd={self.weight_decay}; patience={self.patience}; fp16={self.fp16}")


def resolve_device(name: str):
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda requested but CUDA is not available; use device=auto or cpu")
    return torch.device(name)


# ---------------------------------------------------- model and tokenizer ----
def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(resolve_model_name(model_name))


def load_model(model_name: str, dropout: Optional[float] = None):
    """A sequence-classification head with our three labels on top of `model_name`."""
    from transformers import AutoConfig, AutoModelForSequenceClassification

    name = resolve_model_name(model_name)
    config = AutoConfig.from_pretrained(name, num_labels=len(LABELS), id2label=dict(ID2LABEL),
                                        label2id=dict(LABEL2ID))
    if dropout is not None:  # BERT-style and XLM-R configs share these names
        for attr in ("hidden_dropout_prob", "attention_probs_dropout_prob", "classifier_dropout"):
            if hasattr(config, attr):
                setattr(config, attr, dropout)
    return AutoModelForSequenceClassification.from_pretrained(name, config=config)


# ------------------------------------------------------- sequence lengths ----
def token_lengths(tokenizer, texts: Sequence[str]) -> List[int]:
    """Subword length of every text INCLUDING special tokens ([CLS]/[SEP] or <s>/</s>), untruncated."""
    enc = tokenizer(list(texts), add_special_tokens=True, truncation=False)
    return [len(ids) for ids in enc["input_ids"]]


def length_stats(lengths: Sequence[int]) -> dict:
    arr = np.asarray(lengths)
    pct = {f"p{p}": float(np.percentile(arr, p)) for p in (50, 90, 95, 99, 99.9)}
    return {"n": int(arr.size), "mean": round(float(arr.mean()), 2), **pct, "max": int(arr.max())}


def derive_max_length(lengths: Sequence[int], percentile: float = 100.0, multiple_of: int = 8,
                      model_limit: Optional[int] = 512) -> int:
    """Smallest multiple of `multiple_of` that covers the `percentile`-th TRAIN length, capped at the
    model's position limit. With percentile=100 nothing in train is truncated."""
    need = math.ceil(float(np.percentile(np.asarray(lengths), percentile)))
    out = int(math.ceil(need / multiple_of) * multiple_of)
    if model_limit:
        out = min(out, int(model_limit))
    return max(out, multiple_of)


def model_position_limit(tokenizer) -> int:
    """The tokenizer's max input length; some tokenizers report a huge sentinel, then 512 is used."""
    m = getattr(tokenizer, "model_max_length", 512) or 512
    return int(m) if m < 100_000 else 512


def count_truncated(lengths: Sequence[int], max_length: int) -> dict:
    over = [n for n in lengths if n > max_length]
    return {"tweets": len(over), "share": round(len(over) / max(len(lengths), 1), 5),
            "subwords_cut": int(sum(n - max_length for n in over))}


# ---------------------------------------------------------- data batches ----
class EncodedTweets:
    """Tokenised tweets (truncated to max_length, NOT padded) plus label ids, as a torch Dataset.

    Padding happens per batch in `make_collate`, so short batches stay short (dynamic padding)."""

    def __init__(self, tokenizer, texts: Sequence[str], labels: Optional[Sequence[str]], max_length: int):
        enc = tokenizer(list(texts), truncation=True, max_length=max_length, padding=False)
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.token_type_ids = enc.get("token_type_ids")
        if labels is not None:
            bad = set(labels) - set(LABEL2ID)
            if bad:
                raise ValueError(f"unknown labels {sorted(bad)}; expected {LABELS}")
        self.labels = [LABEL2ID[lab] for lab in labels] if labels is not None else None

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        item = {"input_ids": self.input_ids[i], "attention_mask": self.attention_mask[i]}
        if self.token_type_ids is not None:
            item["token_type_ids"] = self.token_type_ids[i]
        if self.labels is not None:
            item["labels"] = self.labels[i]
        return item


def make_collate(tokenizer):
    """Pad a list of EncodedTweets items to the longest in the batch; returns torch tensors."""
    import torch

    def collate(items):
        labels = [it.pop("labels") for it in items] if "labels" in items[0] else None
        batch = tokenizer.pad(items, padding="longest", return_tensors="pt")
        if labels is not None:
            batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch

    return collate


# ------------------------------------------------------------ prediction ----
def predict_ids(model, loader, device, fp16: bool = False) -> List[int]:
    """Arg-max label ids for every row of `loader` (eval mode, no gradients, order kept)."""
    import torch

    model.eval()
    preds: List[int] = []
    use_amp = fp16 and device.type == "cuda"
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(**batch).logits
            preds.extend(logits.argmax(dim=-1).tolist())
    return preds


def ids_to_labels(ids: Sequence[int]) -> List[str]:
    return [ID2LABEL[int(i)] for i in ids]


def prediction_frame(uids, labels, preds):
    """The common prediction format used by every model in this project: uid,label,pred."""
    import pandas as pd

    if not (len(uids) == len(labels) == len(preds)):
        raise ValueError("uids, labels and preds must have the same length")
    return pd.DataFrame({"uid": list(uids), "label": list(labels), "pred": list(preds)})


# ------------------------------------------------------------ checkpoints ----
def save_checkpoint_dir(path, model, tokenizer, info: dict) -> Path:
    """Save model + tokenizer with `save_pretrained` and a `run_info.json` (config, seed, dev metrics,
    max_length, ...). Reload with `load_checkpoint_dir`. The folder is git-ignored (checkpoints/)."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    (path / "run_info.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    return path


def load_checkpoint_dir(path):
    """(model, tokenizer, run_info) from a folder written by `save_checkpoint_dir`."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    path = Path(path)
    info = json.loads((path / "run_info.json").read_text(encoding="utf-8"))
    return (AutoModelForSequenceClassification.from_pretrained(path), AutoTokenizer.from_pretrained(path), info)


# ------------------------------------------------- length analysis (CLI) ----
def analyze_lengths(models: Sequence[str], mode: str = "clean", root=None, percentile: float = 100.0) -> Dict[str, dict]:
    """Subword-length statistics of the TRAIN and DEV tweets for each model (test is never loaded).
    The suggested max_length comes from TRAIN only; the dev numbers show how much it would truncate."""
    from src.preprocess import load_processed_split

    train = list(load_processed_split("train", mode=mode, root=root)["text_proc"])
    dev = list(load_processed_split("dev", mode=mode, root=root)["text_proc"])
    out = {}
    for m in models:
        tok = load_tokenizer(m)
        tr, dv = token_lengths(tok, train), token_lengths(tok, dev)
        max_len = derive_max_length(tr, percentile, model_limit=model_position_limit(tok))
        out[model_slug(m)] = {
            "checkpoint": resolve_model_name(m), "mode": mode, "percentile": percentile,
            "train": length_stats(tr), "dev": length_stats(dv), "derived_max_length": max_len,
            "truncated": {"train": count_truncated(tr, max_len), "dev": count_truncated(dv, max_len)},
        }
    return out


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description="Measure subword lengths of the Hinglish tweets per tokenizer.")
    p.add_argument("--models", nargs="+", default=["mbert", "xlmr"])
    p.add_argument("--mode", default="clean", choices=list(PRESETS))
    p.add_argument("--percentile", type=float, default=100.0)
    p.add_argument("--data-root")
    p.add_argument("--out", default=str(ROOT / "results" / "transformer_token_lengths.json"))
    args = p.parse_args(argv)
    res = analyze_lengths(args.models, args.mode, args.data_root, args.percentile)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    for name, r in res.items():
        print(f"{name:10s} train mean {r['train']['mean']:6.1f}  p99 {r['train']['p99']:6.1f}  max {r['train']['max']:4d}"
              f"  -> max_length {r['derived_max_length']}  (dev truncated: {r['truncated']['dev']['tweets']})")
    return res


if __name__ == "__main__":
    main()
