"""BiLSTM for Hinglish sentiment: data encoding, model and checkpoints in one file.

Word embedding + character CNN -> BiLSTM -> pooling -> 3-class softmax. Training is in src/train_bilstm.py.
Sections: 1. data (vocabularies, encoding, batching)  2. model  3. checkpoints (save/load).
See docs/bilstm.md.
"""
from __future__ import annotations

import hashlib
import math
import os
import warnings
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader

from src.metrics import LABELS
from src.preprocess import DEFAULT_VARIANTS_PATH, MENTION_PLACEHOLDER, PRESETS, URL_PLACEHOLDER, preprocess_text, tokenize


# ====================================================================================================
# 1. Data: vocabularies, encoding and batching
# Vocabularies, encoding and batching for the word + character BiLSTM.
#
# Everything that is *learned from data* (vocabularies, length limits) is fitted on
# the TRAIN split only, by `TextEncoder.fit`. Dev and test texts are only *encoded*:
# a word or character never seen in train becomes `<unk>`.
#
# The text fed in here is the `text_proc` column from `src.preprocess`
# (`load_processed_split`), already split into whitespace tokens.
#
# Nothing in this module imports torch at import time; only `make_batch` does.
# ====================================================================================================

PAD, UNK = "<pad>", "<unk>"
PAD_ID, UNK_ID = 0, 1
# The words src.preprocess writes in place of mentions and URLs. They are reserved word-vocabulary entries
# (ids 2 and 3 in every mode) and are never lowercased, so they cannot collide with the ordinary words
# "user" and "url". Only the exact upper-case spelling is the placeholder.
PLACEHOLDER_WORDS = (MENTION_PLACEHOLDER, URL_PLACEHOLDER)
LABEL2ID = {label: i for i, label in enumerate(LABELS)}  # negative=0, neutral=1, positive=2
ID2LABEL = {i: label for label, i in LABEL2ID.items()}


# ------------------------------------------------------------------ Vocab ----
class Vocab:
    """Token <-> id table. Id 0 is `<pad>`, id 1 is `<unk>`; real tokens start at 2."""

    def __init__(self, itos: Sequence[str]):
        if list(itos[:2]) != [PAD, UNK]:
            raise ValueError("a vocabulary must start with [<pad>, <unk>]")
        if len(set(itos)) != len(itos):
            raise ValueError("duplicate tokens in vocabulary")
        self.itos = list(itos)
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    @classmethod
    def build(cls, sequences: Iterable[Iterable[str]], min_freq: int = 1, max_size: Optional[int] = None,
              specials: Sequence[str] = ()):
        """Count tokens in `sequences` (an iterable of token lists) and keep those seen
        at least `min_freq` times. Order is by frequency, then alphabetically, so the
        result is identical on every run.

        `specials` are reserved tokens that always get the ids right after `<pad>` and `<unk>`
        (2, 3, ...), whether or not they occur in `sequences`, and are exempt from `min_freq` and
        `max_size`. Their ids therefore do not depend on the data (or on the preprocessing mode)."""
        specials = list(specials)
        if len(set(specials)) != len(specials) or {PAD, UNK} & set(specials):
            raise ValueError("specials must be unique and different from <pad> and <unk>")
        # special tokens are never counted as ordinary words
        reserved = {PAD, UNK, *specials}
        # count how often each token appears
        counts = Counter(t for seq in sequences for t in seq if t not in reserved)
        # most frequent first; ties sorted alphabetically so the result is the same on every run
        items = sorted(((t, c) for t, c in counts.items() if c >= min_freq), key=lambda x: (-x[1], x[0]))
        if max_size is not None:
            items = items[:max_size]
        return cls([PAD, UNK, *specials] + [t for t, _ in items])

    def __len__(self) -> int:
        return len(self.itos)

    def __contains__(self, token: str) -> bool:
        return token in self.stoi

    def encode(self, tokens: Iterable[str]) -> List[int]:
        """Ids of `tokens`; anything not in the vocabulary becomes `<unk>`."""
        # tokens not in the vocabulary become <unk>
        return [self.stoi.get(t, UNK_ID) for t in tokens]


# ------------------------------------------------------------ TextEncoder ----
def derive_length(lengths: Sequence[int], percentile: float = 100.0) -> int:
    """A length limit taken from data: the given percentile of `lengths`, rounded up (>= 1)."""
    if len(lengths) == 0:
        return 1
    return max(1, int(math.ceil(np.percentile(lengths, percentile))))


def _normalize_token(token: str, lowercase: bool, special_words: Sequence[str]) -> str:
    """Lowercase `token` if asked, except the reserved placeholder words, which keep their exact spelling."""
    return token if (not lowercase or token in special_words) else token.lower()


class TextEncoder:
    """Turns one preprocessed tweet into word ids and per-word character ids.

    Fit on train texts with `TextEncoder.fit(...)`, then call `encode(text)` on any split.
    Limits: at most `max_words` tokens per tweet and `max_chars` characters per token;
    longer ones are cut (the beginning is kept).

    Placeholders: the words in `special_words` (default `USER` and `URL`, what src.preprocess writes for
    mentions and URLs) have reserved word-vocabulary ids 2 and 3 in every vocabulary, even if a mode or a
    dataset never contains them, and are exempt from lowercasing (for words and for characters). So the
    placeholder `USER` and the ordinary word `user` (or `User`) always have different ids, and ordinary
    words keep their own representation.
    """

    def __init__(self, word_vocab: Vocab, char_vocab: Vocab, max_words: int, max_chars: int,
                 lowercase_words: bool = True, lowercase_chars: bool = False,
                 special_words: Sequence[str] = PLACEHOLDER_WORDS):
        self.word_vocab, self.char_vocab = word_vocab, char_vocab
        self.max_words, self.max_chars = max_words, max_chars
        self.lowercase_words, self.lowercase_chars = lowercase_words, lowercase_chars
        self.special_words = tuple(special_words)

    @classmethod
    def fit(cls, train_texts: Sequence[str], *, word_min_freq: int = 2, char_min_freq: int = 2,
            max_words: Optional[int] = None, max_chars: Optional[int] = None,
            words_percentile: float = 100.0, chars_percentile: float = 99.9,
            lowercase_words: bool = True, lowercase_chars: bool = False,
            special_words: Sequence[str] = PLACEHOLDER_WORDS) -> "TextEncoder":
        """Build vocabularies and (unless given) length limits from TRAIN texts only.

        max_words defaults to the longest train tweet (words_percentile=100), so nothing
        in train is cut; max_chars defaults to the 99.9th percentile of train token length,
        so a few very long tokens (for example glued hashtags) are truncated.
        """
        special_words = tuple(special_words)
        # vocabularies are built from TRAIN texts only, never from dev or test
        train_tokens = [tokenize(t) for t in train_texts]
        word_seqs = [[_normalize_token(w, lowercase_words, special_words) for w in toks] for toks in train_tokens]
        char_seqs = [[ch for w in toks for ch in _normalize_token(w, lowercase_chars, special_words)]
                     for toks in train_tokens]
        # words seen fewer than word_min_freq times are left out and later become <unk>
        word_vocab = Vocab.build(word_seqs, min_freq=word_min_freq, specials=special_words)
        char_vocab = Vocab.build(char_seqs, min_freq=char_min_freq)
        # length limits come from the train data unless they are given
        if max_words is None:
            max_words = derive_length([len(t) for t in train_tokens], words_percentile)
        if max_chars is None:
            max_chars = derive_length([len(w) for toks in train_tokens for w in toks], chars_percentile)
        return cls(word_vocab, char_vocab, max_words, max_chars, lowercase_words, lowercase_chars, special_words)

    def count_truncated(self, texts: Sequence[str]) -> dict:
        """How much `encode` cuts from `texts`: {"tweets": tweets with more than max_words
        tokens, "words": tokens longer than max_chars characters (among the kept tokens)}."""
        tweets = words = 0
        for t in texts:
            toks = tokenize(t)
            tweets += len(toks) > self.max_words
            words += sum(len(w) > self.max_chars for w in toks[: self.max_words])
        return {"tweets": int(tweets), "words": int(words)}

    def to_dict(self) -> dict:
        """Plain-Python form (lists, ints, bools) used inside checkpoints."""
        return {"word_itos": list(self.word_vocab.itos), "char_itos": list(self.char_vocab.itos),
                "max_words": int(self.max_words), "max_chars": int(self.max_chars),
                "lowercase_words": bool(self.lowercase_words), "lowercase_chars": bool(self.lowercase_chars),
                "special_words": list(self.special_words)}

    @classmethod
    def from_dict(cls, d: dict) -> "TextEncoder":
        return cls(Vocab(d["word_itos"]), Vocab(d["char_itos"]), d["max_words"], d["max_chars"],
                   d["lowercase_words"], d["lowercase_chars"], d["special_words"])

    def encode(self, text):
        """Return (word_ids, char_ids): word_ids is a list of ints (length 1..max_words) and
        char_ids a list with one list of ints (length 1..max_chars) per word.
        An empty tweet becomes a single `<unk>` word so that every tweet has length >= 1."""
        # cut tweets that are longer than max_words
        tokens = tokenize(text)[: self.max_words]
        # an empty tweet becomes one unknown word so the model always gets at least one word
        if not tokens:
            return [UNK_ID], [[UNK_ID]]
        words = self.word_vocab.encode(
            [_normalize_token(t, self.lowercase_words, self.special_words) for t in tokens])
        char_src = [_normalize_token(t, self.lowercase_chars, self.special_words) for t in tokens]
        # each word becomes a list of character ids, cut to max_chars
        chars = [self.char_vocab.encode(list(t[: self.max_chars])) for t in char_src]
        return words, chars


# ---------------------------------------------------------------- dataset ----
class SentimentDataset:
    """Encoded tweets with integer labels. Works with `torch.utils.data.DataLoader`
    (it only needs `__len__` and `__getitem__`). `labels=None` gives label -1 for every row."""

    def __init__(self, encoder: TextEncoder, texts: Sequence[str], labels: Optional[Sequence[str]] = None):
        if labels is not None and len(labels) != len(texts):
            raise ValueError(f"{len(texts)} texts but {len(labels)} labels")
        if labels is not None:
            unknown = set(labels) - set(LABEL2ID)
            if unknown:
                raise ValueError(f"unknown labels {sorted(unknown)}; expected {LABELS}")
        # encode every tweet once, up front
        self.items = [encoder.encode(t) for t in texts]
        self.labels = [LABEL2ID[l] for l in labels] if labels is not None else [-1] * len(texts)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        words, chars = self.items[i]
        return words, chars, self.labels[i]


def make_batch(items):
    """Collate function: pad a list of (word_ids, char_ids, label) to rectangular tensors.

    Padding is per batch (dynamic), so a batch is only as wide as its longest tweet / word.
    Returns a dict:
      words   (B, T)    word ids, 0 = <pad>
      chars   (B, T, C) character ids, 0 = <pad>
      lengths (B,)      true number of words per tweet
      labels  (B,)      label ids (-1 if unlabeled)
    """
    import torch

    # find the longest tweet and longest word in this batch and pad everything to that size
    B = len(items)
    lengths = [len(w) for w, _, _ in items]
    T = max(lengths)
    C = max(len(c) for _, chars, _ in items for c in chars)
    # start with all padding, then fill in the real ids
    words = torch.full((B, T), PAD_ID, dtype=torch.long)
    chars = torch.full((B, T, C), PAD_ID, dtype=torch.long)
    for i, (w, cs, _) in enumerate(items):
        words[i, : len(w)] = torch.tensor(w, dtype=torch.long)
        for j, c in enumerate(cs):
            chars[i, j, : len(c)] = torch.tensor(c, dtype=torch.long)
    return {
        "words": words,
        "chars": chars,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "labels": torch.tensor([lab for _, _, lab in items], dtype=torch.long),
    }


# ====================================================================================================
# 2. Model: word embedding + character CNN + BiLSTM
# Word + character BiLSTM sentiment classifier (PyTorch).
#
# Data flow for a batch of B tweets with T words each and C characters per word:
#
#     words (B,T) --word embedding--------------------------------------+
#                                                                        +--concat--> (B,T,Dw+Dc)
#     chars (B,T,C) --char embedding--CNN (kernels 2,3,4)--max-pool------+
#         --dropout--> word-level BiLSTM --> masked pooling --> dropout --> Linear --> 3 logits
#
# Why a CNN for characters (and not a character BiLSTM): it is much cheaper (every word of every
# tweet is encoded), and its n-gram filters pick up sub-word patterns such as "-ahut" / "-ohot"
# that make different spellings of one Hindi word look alike. A character BiLSTM would be
# slower for little expected gain on 3-12 character words.
# ====================================================================================================

POOLINGS = ("maxmean", "mean", "max", "last")


class CharCNN(nn.Module):
    """Encode each word from its characters: embedding -> 1-D convolutions -> max over positions.

    Input  (B, T, C) character ids  ->  output (B, T, n_filters * len(kernel_sizes)).

    Padding invariance: a word gets the same vector whatever the batch's padding length C is.
    A convolution window that overlaps padding would still produce an output, and which such
    windows exist depends on C, so they are excluded from the max. For a word of true length L
    and kernel size k only the windows lying fully inside the word (positions 0 .. L-k) count.
    A word shorter than k has no full window; it uses its single window at position 0, which
    holds its characters followed by zero vectors (padding embeddings are exactly zero), and
    that is the same for every C.
    """

    def __init__(self, n_chars: int, char_dim: int = 32, n_filters: int = 32,
                 kernel_sizes: Sequence[int] = (2, 3, 4), pad_id: int = 0):
        super().__init__()
        # remember which id means padding so it can be ignored later
        self.pad_id = pad_id
        self.kernel_sizes = tuple(kernel_sizes)
        # one small vector per character; the padding id always stays zero
        self.embedding = nn.Embedding(n_chars, char_dim, padding_idx=pad_id)
        # one convolution per kernel size (2, 3, 4 characters wide); each looks for character patterns
        self.convs = nn.ModuleList(nn.Conv1d(char_dim, n_filters, k) for k in self.kernel_sizes)
        # the widest kernel needs at least this many characters
        self.min_len = max(self.kernel_sizes)
        # the word vector is the filters of all kernel sizes joined together
        self.out_dim = n_filters * len(self.kernel_sizes)

    def forward(self, chars: torch.Tensor) -> torch.Tensor:
        B, T, C = chars.shape
        # treat every word of every tweet as its own short sequence of characters
        flat = chars.reshape(B * T, C)
        word_len = (flat != self.pad_id).sum(dim=1)  # true length of each word (padding is on the right)
        x = self.embedding(flat).transpose(1, 2)  # (B*T, char_dim, C)
        if C < self.min_len:  # very short batches: zero-pad so every kernel fits
            x = F.pad(x, (0, self.min_len - C))
        # run each convolution and keep the strongest match for every filter
        feats = []
        for conv, k in zip(self.convs, self.kernel_sizes):
            h = F.relu(conv(x))  # (B*T, n_filters, windows)
            n_valid = (word_len - k + 1).clamp(min=1)  # windows fully inside the word (at least 1)
            valid = torch.arange(h.size(2), device=h.device)[None, :] < n_valid[:, None]  # (B*T, windows)
            feats.append(h.masked_fill(~valid[:, None, :], float("-inf")).max(dim=2).values)
        # join the filters and go back to the (tweet, word) shape
        return torch.cat(feats, dim=1).reshape(B, T, self.out_dim)


class BiLSTMClassifier(nn.Module):
    """Word-level BiLSTM over [word embedding ; character-CNN vector] for each word.

    use_word / use_char switch the two inputs on or off (at least one must be on), which
    gives the "word only", "char only" and "word + char" variants from one class.
    pooling: how the per-word BiLSTM outputs become one tweet vector
      "maxmean" (default) = concat(max over words, mean over words); "mean"; "max";
      "last" = final forward and backward hidden states. Padding is always masked out.
    """

    def __init__(self, n_words: int, n_chars: int, n_classes: int = 3, *, use_word: bool = True,
                 use_char: bool = True, word_dim: int = 100, char_dim: int = 32, char_filters: int = 32,
                 char_kernels: Sequence[int] = (2, 3, 4), hidden_size: int = 128, num_layers: int = 1,
                 dropout: float = 0.4, pooling: str = "maxmean", pad_id: int = 0):
        super().__init__()
        if not (use_word or use_char):
            raise ValueError("at least one of use_word / use_char must be True")
        if pooling not in POOLINGS:
            raise ValueError(f"pooling must be one of {POOLINGS}, got {pooling!r}")
        # switches so one class can build word-only, char-only or word+char models
        self.use_word, self.use_char, self.pooling = use_word, use_char, pooling

        # the LSTM input is the word vector and/or the character vector, joined side by side
        in_dim = 0
        if use_word:
            self.word_emb = nn.Embedding(n_words, word_dim, padding_idx=pad_id)
            in_dim += word_dim
        if use_char:
            self.char_cnn = CharCNN(n_chars, char_dim, char_filters, char_kernels, pad_id)
            in_dim += self.char_cnn.out_dim
        # dropout on the inputs and before the classifier reduces overfitting
        self.embed_dropout = nn.Dropout(dropout)
        # bidirectional LSTM: reads each tweet left-to-right and right-to-left
        self.lstm = nn.LSTM(in_dim, hidden_size, num_layers=num_layers, batch_first=True,
                            bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)
        # size of the tweet vector: 2 directions, doubled when max and mean pooling are joined
        pooled = 2 * hidden_size * (2 if pooling == "maxmean" else 1)
        self.out_dropout = nn.Dropout(dropout)
        # final layer: one score per class (negative, neutral, positive)
        self.classifier = nn.Linear(pooled, n_classes)

    def forward(self, words: torch.Tensor, chars: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """words (B,T), chars (B,T,C), lengths (B,) -> logits (B, n_classes)."""
        B, T = words.shape
        # look up the vectors for each word and for its characters
        parts = []
        if self.use_word:
            parts.append(self.word_emb(words))
        if self.use_char:
            parts.append(self.char_cnn(chars))
        x = self.embed_dropout(torch.cat(parts, dim=-1))  # (B, T, in_dim)

        # pack the batch so the LSTM skips the padding positions
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, (h, _) = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=T)  # (B, T, 2H)

        # mask is True for real words and False for padding
        mask = (torch.arange(T, device=out.device)[None, :] < lengths.to(out.device)[:, None]).unsqueeze(-1)
        if self.pooling == "last":
            vec = torch.cat([h[-2], h[-1]], dim=1)  # last layer: forward, backward
        else:
            # average over real words only (padding is not counted)
            mean = (out * mask).sum(dim=1) / lengths.to(out.device).unsqueeze(1).clamp(min=1)
            # strongest value per feature over real words (padding is set to -infinity so it never wins)
            mx = out.masked_fill(~mask, float("-inf")).max(dim=1).values
            vec = {"mean": mean, "max": mx, "maxmean": torch.cat([mx, mean], dim=1)}[self.pooling]
        # raw scores (logits); the softmax is inside the loss function
        return self.classifier(self.out_dropout(vec))


# ====================================================================================================
# 3. Checkpoints: save and load a trained model
# Save and load BiLSTM checkpoints (best dev-weighted-F1 epoch of one run).
#
# A checkpoint is one `.pt` file that holds everything needed to evaluate the model or predict
# later without retraining:
#
#     format_version  int
#     model_state     the weights (CPU tensors)
#     config          every BiLSTMConfig setting (dict), including the preprocessing `mode`
#     encoder         word and character vocabularies (id -> token lists), length limits, lowercase flags
#     labels          label names in id order: ["negative", "neutral", "positive"]
#     meta            mode, seed, best_epoch, dev metrics, preprocessing options, a fingerprint of the
#                     Hinglish dictionary, train/dev sizes, length limits and truncation counts, versions
#
# Only train-fitted objects, dev metrics and settings are stored. No test tweets, test labels or
# test-derived numbers ever enter a checkpoint (the training script never loads the test split).
#
# Files are stored at   <out_dir>/checkpoints/bilstm_<run_name>_seed<seed>.pt
# (`checkpoints/` and `*.pt` are git-ignored). Loading uses `torch.load(weights_only=True)`, so the file
# can contain only tensors and plain Python containers; it cannot run code.
# ====================================================================================================

FORMAT_VERSION = 2  # 2: word vocabularies reserve the placeholder ids and the encoder stores `special_words`
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
    # make sure the folder exists
    path.parent.mkdir(parents=True, exist_ok=True)
    # everything needed to rebuild and use the model later; no test data is inside
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
    # write to a temporary file first, then rename, so a crash never leaves a broken checkpoint
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
        # apply the same preprocessing mode the model was trained with
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

    # weights_only=True: the file can only hold tensors and plain data, so loading cannot run code
    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    # refuse files that are not BiLSTM checkpoints
    missing = [k for k in REQUIRED_KEYS if k not in payload]
    if missing:
        raise ValueError(f"{path} is not a BiLSTM checkpoint (missing {missing})")
    if payload["format_version"] != FORMAT_VERSION:
        raise ValueError(f"unsupported checkpoint format {payload['format_version']} (expected {FORMAT_VERSION})")
    # rebuild the same architecture that was trained, then load the saved weights into it
    cfg = BiLSTMConfig().with_overrides(payload["config"])
    encoder = TextEncoder.from_dict(payload["encoder"])
    model = build_model(cfg, encoder)
    model.load_state_dict(payload["model_state"], strict=True)
    # evaluation mode: dropout is off
    model.eval()
    meta = payload["meta"]
    # warn if the Hinglish spelling dictionary changed since training
    if meta.get("variants_sha256") != variants_fingerprint():
        warnings.warn("configs/hinglish_variants.json changed since this checkpoint was trained; "
                      "preprocessing of new text will differ from training", stacklevel=2)
    return LoadedCheckpoint(model, encoder, cfg, meta)
