"""A tiny synthetic dataset in the real SentiMix CoNLL format, shared by the BiLSTM tests.

The label is decided by keywords, so a working model must learn it. Some tweets contain a
mention, a spelling variant ("acha", "bohot") and a split-up URL so that preprocessing is exercised.
Only train and dev files are written, on purpose: any attempt to load the test split would fail.
"""
import random

from src.metrics import LABELS

KEYWORDS = {
    "positive": ["badhiya", "mast", "zabardast", "love"],
    "negative": ["bakwas", "bura", "hate", "worst"],
    "neutral": ["kal", "match", "news", "today"],
}
FILLER = ["yaar", "hai", "the", "aaj", "bhai", "ye", "toh", "so", "kya", "log"]

# small, fast model/training settings for tests
SMALL = dict(word_dim=16, char_dim=8, char_filters=8, hidden_size=16, epochs=8, patience=8, batch_size=32, lr=0.01,
             word_min_freq=1, char_min_freq=1, device="cpu")


def write_conll(path, n, seed, start_uid):
    rng = random.Random(seed)
    lines = []
    for i in range(n):
        label = LABELS[i % 3]
        toks = ["@", "user", "bohot", "acha", "hai"] if i % 5 == 0 else []  # exercises preprocessing
        toks += rng.sample(FILLER, 3) + rng.sample(KEYWORDS[label], 2) + rng.sample(FILLER, 2)
        if i % 4 == 0:
            toks += ["https", "//", "t", ".", "co", "/", "abc123"]
        lines.append(f"meta\t{start_uid + i}\t{label}")
        lines += [f"{t}\tHin" for t in toks]
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_data_root(tmp_path, n_train=240, n_dev=60):
    root = tmp_path / "Hinglish"
    root.mkdir()
    write_conll(root / "Hinglish_train_14k_split_conll.txt", n_train, 1, 1000)
    write_conll(root / "Hinglish_dev_3k_split_conll.txt", n_dev, 2, 5000)
    return root  # note: no test files on purpose
