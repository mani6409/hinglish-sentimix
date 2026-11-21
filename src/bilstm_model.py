"""Word + character BiLSTM sentiment classifier (PyTorch).

Data flow for a batch of B tweets with T words each and C characters per word:

    words (B,T) --word embedding--------------------------------------+
                                                                       +--concat--> (B,T,Dw+Dc)
    chars (B,T,C) --char embedding--CNN (kernels 2,3,4)--max-pool------+
        --dropout--> word-level BiLSTM --> masked pooling --> dropout --> Linear --> 3 logits

Why a CNN for characters (and not a character BiLSTM): it is much cheaper (every word of every
tweet is encoded), and its n-gram filters pick up sub-word patterns such as "-ahut" / "-ohot"
that make different spellings of one Hindi word look alike. A character BiLSTM would be
slower for little expected gain on 3-12 character words.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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
