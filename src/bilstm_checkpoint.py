"""Save and load BiLSTM checkpoints (best dev-weighted-F1 epoch of one run).

A checkpoint is one `.pt` file that holds everything needed to evaluate the model or predict
later without retraining:

    format_version  int
    model_state     the weights (CPU tensors)
    config          every BiLSTMConfig setting (dict), including the preprocessing `mode`
    encoder         word and character vocabularies (id -> token lists), length limits, lowercase flags
    labels          label names in id order: ["negative", "neutral", "positive"]
    meta            mode, seed, best_epoch, dev metrics, preprocessing options, a fingerprint of the
                    Hinglish dictionary, train/dev sizes, length limits and truncation counts, versions

Only train-fitted objects, dev metrics and settings are stored. No test tweets, test labels or
test-derived numbers ever enter a checkpoint (the training script never loads the test split).

Files are stored at   <out_dir>/checkpoints/bilstm_<run_name>_seed<seed>.pt
(`checkpoints/` and `*.pt` are git-ignored). Loading uses `torch.load(weights_only=True)`, so the file
can contain only tensors and plain Python containers; it cannot run code.
"""
from __future__ import annotations

import hashlib
import os
import warnings
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import List, Optional, Sequence

import torch
from torch.utils.data import DataLoader

from src.bilstm_data import ID2LABEL, SentimentDataset, TextEncoder, make_batch
from src.preprocess import DEFAULT_VARIANTS_PATH, PRESETS, preprocess_text

FORMAT_VERSION = 1
REQUIRED_KEYS = ("format_version", "model_state", "config", "encoder", "labels", "meta")


def checkpoint_path(out_dir, run_name: str, seed: int) -> Path:
    """The predictable location of one run's checkpoint."""
    return Path(out_dir) / "checkpoints" / f"bilstm_{run_name}_seed{seed}.pt"


def variants_fingerprint() -> str:
    """SHA-256 of configs/hinglish_variants.json, to notice if the dictionary changed after training."""
    return hashlib.sha256(Path(DEFAULT_VARIANTS_PATH).read_bytes()).hexdigest()


def save_checkpoint(path, model, encoder: TextEncoder, cfg, *, seed: int, best_epoch: int, metrics: dict,
                    extra: Optional[dict] = None) -> Path:
    """Write the checkpoint atomically (temp file, then rename) and return its path.

    `metrics` is the dev metrics dict from src.metrics.compute_metrics for the saved (best) epoch.
    `extra` is merged into `meta` (limits, sizes, truncation counts, ...).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": FORMAT_VERSION,
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "config": asdict(cfg),
        "encoder": encoder.to_dict(),
        "labels": [ID2LABEL[i] for i in range(len(ID2LABEL))],
        "meta": {
            "mode": cfg.mode,
            "seed": int(seed),
            "best_epoch": int(best_epoch),
            "selected_on": "dev weighted_f1",
            "dev_metrics": metrics,
            "preprocess_options": dict(PRESETS[cfg.mode]),
            "variants_sha256": variants_fingerprint(),
            "torch_version": str(torch.__version__),  # plain str: torch.__version__ is a str subclass the safe loader rejects
            "created": date.today().isoformat(),
            **(extra or {}),
        },
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


@dataclass
class LoadedCheckpoint:
    """A model restored from a checkpoint (eval mode), with its encoder, config and metadata."""

    model: torch.nn.Module
    encoder: TextEncoder
    config: object  # src.train_bilstm.BiLSTMConfig
    meta: dict

    def predict_texts(self, texts: Sequence[str], preprocess: bool = False, batch_size: int = 256,
                      device: Optional[str] = None) -> List[str]:
        """Predicted label names for `texts`.

        preprocess=False: `texts` are already preprocessed (the `text_proc` column).
        preprocess=True : `texts` are original tweets; the checkpoint's own preprocessing mode is applied first.
        """
        from src.train_bilstm import predict  # lazy: train_bilstm imports this module

        texts = list(texts)
        if preprocess:
            opts = PRESETS[self.config.mode]
            texts = [preprocess_text(t, **opts) for t in texts]
        dev = torch.device(device or "cpu")
        self.model.to(dev)
        loader = DataLoader(SentimentDataset(self.encoder, texts), batch_size=batch_size, shuffle=False,
                            collate_fn=make_batch)
        return [ID2LABEL[i] for i in predict(self.model, loader, dev)]


def load_checkpoint(path, map_location="cpu") -> LoadedCheckpoint:
    """Rebuild the model, encoder and config from a checkpoint file.

    Raises ValueError for a file in an unknown format. Warns if the Hinglish dictionary has changed
    since the model was trained (the preprocessing at inference would then differ from training).
    """
    from src.train_bilstm import BiLSTMConfig, build_model  # lazy: train_bilstm imports this module

    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    missing = [k for k in REQUIRED_KEYS if k not in payload]
    if missing:
        raise ValueError(f"{path} is not a BiLSTM checkpoint (missing {missing})")
    if payload["format_version"] != FORMAT_VERSION:
        raise ValueError(f"unsupported checkpoint format {payload['format_version']} (expected {FORMAT_VERSION})")
    cfg = BiLSTMConfig().with_overrides(payload["config"])
    encoder = TextEncoder.from_dict(payload["encoder"])
    model = build_model(cfg, encoder)
    model.load_state_dict(payload["model_state"], strict=True)
    model.eval()
    meta = payload["meta"]
    if meta.get("variants_sha256") != variants_fingerprint():
        warnings.warn("configs/hinglish_variants.json changed since this checkpoint was trained; "
                      "preprocessing of new text will differ from training", stacklevel=2)
    return LoadedCheckpoint(model, encoder, cfg, meta)
