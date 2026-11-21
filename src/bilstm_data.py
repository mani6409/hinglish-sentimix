"""Vocabularies, encoding and batching for the word + character BiLSTM.

Everything that is *learned from data* (vocabularies, length limits) is fitted on
the TRAIN split only, by `TextEncoder.fit`. Dev and test texts are only *encoded*:
a word or character never seen in train becomes `<unk>`.

The text fed in here is the `text_proc` column from `src.preprocess`
(`load_processed_split`), already split into whitespace tokens.

Nothing in this module imports torch at import time; only `make_batch` does.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, List, Optional, Sequence

import numpy as np

from src.metrics import LABELS
from src.preprocess import MENTION_PLACEHOLDER, URL_PLACEHOLDER, tokenize

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
