"""Tests for the BiLSTM model, config, and the training / evaluation pipeline.

The pipeline tests build a tiny synthetic dataset in the real CoNLL format (labels are decided
by keywords, so a working model must learn it) and run it through the real loader,
preprocessing, training and logging code. No TEST file is created, so any attempt to load the
test split would fail.
"""
import csv
import json

import pytest

torch = pytest.importorskip("torch")

import pandas as pd  # noqa: E402

import src.train_bilstm as tb  # noqa: E402
from src.bilstm import PLACEHOLDER_WORDS, SentimentDataset, TextEncoder, make_batch  # noqa: E402
from src.bilstm import POOLINGS, BiLSTMClassifier, CharCNN  # noqa: E402
from src.metrics import LABELS  # noqa: E402
from src.preprocess import load_processed_split  # noqa: E402
from src.train_bilstm import BiLSTMConfig, derive_shared_limits, main, run_experiment, train_one_seed  # noqa: E402
from tests.synthetic_conll import SMALL  # noqa: E402


# -------------------------------------------------------------- the model ----
def _batch(n_words=50, n_chars=30, B=4, T=6, C=7):
    g = torch.Generator().manual_seed(0)
    words = torch.randint(2, n_words, (B, T), generator=g)
    chars = torch.randint(2, n_chars, (B, T, C), generator=g)
    lengths = torch.tensor([6, 4, 2, 1])
    for i, L in enumerate(lengths.tolist()):
        words[i, L:] = 0
        chars[i, L:] = 0
    return words, chars, lengths


def test_char_cnn_output_shape_including_very_short_words():
    cnn = CharCNN(30, char_dim=8, n_filters=5, kernel_sizes=(2, 3, 4))
    assert cnn(torch.randint(2, 30, (3, 4, 7))).shape == (3, 4, 15)
    assert cnn(torch.randint(2, 30, (3, 4, 1))).shape == (3, 4, 15)  # word shorter than the kernels


@pytest.mark.parametrize("pooling", POOLINGS)
@pytest.mark.parametrize("use_word,use_char", [(True, True), (True, False), (False, True)])
def test_output_has_three_class_logits(pooling, use_word, use_char):
    words, chars, lengths = _batch()
    m = BiLSTMClassifier(50, 30, 3, use_word=use_word, use_char=use_char, word_dim=8, char_dim=6, char_filters=4,
                         hidden_size=10, pooling=pooling)
    out = m(words, chars, lengths)
    assert out.shape == (4, 3) and torch.isfinite(out).all()


def test_two_layers_and_prediction_ids_are_valid_classes():
    words, chars, lengths = _batch()
    m = BiLSTMClassifier(50, 30, num_layers=2, word_dim=8, char_dim=6, char_filters=4, hidden_size=10).eval()
    pred = m(words, chars, lengths).argmax(dim=1)
    assert set(pred.tolist()) <= {0, 1, 2}  # negative / neutral / positive


@pytest.mark.parametrize("pooling", POOLINGS)
def test_padding_does_not_change_the_prediction(pooling):
    """A tweet must get the same logits alone and when padded next to a longer tweet."""
    words, chars, lengths = _batch()
    m = BiLSTMClassifier(50, 30, word_dim=8, char_dim=6, char_filters=4, hidden_size=10, pooling=pooling).eval()
    full = m(words, chars, lengths)
    for i in range(4):
        L = lengths[i].item()
        alone = m(words[i:i + 1, :L], chars[i:i + 1, :L], lengths[i:i + 1])
        assert torch.allclose(full[i:i + 1], alone, atol=1e-5)


def test_invalid_model_arguments():
    with pytest.raises(ValueError):
        BiLSTMClassifier(10, 10, use_word=False, use_char=False)
    with pytest.raises(ValueError):
        BiLSTMClassifier(10, 10, pooling="median")


def test_gradients_reach_both_embedding_tables():
    words, chars, lengths = _batch()
    m = BiLSTMClassifier(50, 30, word_dim=8, char_dim=6, char_filters=4, hidden_size=10)
    m(words, chars, lengths).sum().backward()
    assert m.word_emb.weight.grad.abs().sum() > 0
    assert m.char_cnn.embedding.weight.grad.abs().sum() > 0


# ---- regression tests for review issue C1: character CNN must not depend on padding ----
@pytest.mark.parametrize("kernels", [(2, 3, 4), (3,), (1, 5)])
@pytest.mark.parametrize("word_len", [1, 2, 3, 4, 5, 7])
def test_char_cnn_same_word_same_vector_for_any_right_padding(kernels, word_len):
    """The same word must get the identical vector whatever the padding length C is,
    including words shorter than, equal to and longer than the kernel sizes."""
    torch.manual_seed(0)
    cnn = CharCNN(30, char_dim=6, n_filters=5, kernel_sizes=kernels).eval()
    word = [3, 7, 2, 9, 4, 11, 5][:word_len]
    outs = [cnn(torch.tensor([[word + [0] * (C - word_len)]])) for C in range(max(word_len, 1), word_len + 14)]
    for o in outs[1:]:
        assert torch.allclose(outs[0], o, atol=1e-6), "char vector changed with the amount of padding"


def test_char_cnn_word_unaffected_by_other_words_in_the_batch():
    """Put a word next to a much longer word (which forces extra padding) and compare with the word alone."""
    torch.manual_seed(1)
    cnn = CharCNN(30, char_dim=6, n_filters=5).eval()
    short, long_ = [3, 7, 2], [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    alone = cnn(torch.tensor([[short]]))
    batch = cnn(torch.tensor([[short + [0] * 9, long_]]))  # (1, 2, ...): short word padded next to a long one
    assert torch.allclose(alone[0, 0], batch[0, 0], atol=1e-6)


def test_char_cnn_all_padding_slot_is_finite():
    cnn = CharCNN(30, char_dim=6, n_filters=5).eval()
    out = cnn(torch.zeros(2, 3, 7, dtype=torch.long))  # words made only of padding (padded word slots)
    assert torch.isfinite(out).all()


def test_whole_model_prediction_independent_of_batch_composition():
    """End to end: tweets with words of different lengths give the same logits alone and batched together."""
    torch.manual_seed(2)
    B, T, C = 4, 5, 9
    words = torch.randint(2, 50, (B, T))
    chars = torch.zeros(B, T, C, dtype=torch.long)
    lengths = torch.tensor([5, 4, 3, 1])
    for i in range(B):
        for j in range(lengths[i]):
            n = int(torch.randint(1, C + 1, (1,)))  # word lengths 1..C, so padding varies per word
            chars[i, j, :n] = torch.randint(2, 30, (n,))
        words[i, lengths[i]:] = 0
    chars[:, :, :] = torch.where(words.unsqueeze(-1) == 0, torch.zeros_like(chars), chars)
    m = BiLSTMClassifier(50, 30, word_dim=8, char_dim=6, char_filters=4, hidden_size=10).eval()
    full = m(words, chars, lengths)
    for i in range(B):
        L = lengths[i].item()
        c = chars[i:i + 1, :L]
        c = c[:, :, : max(1, int((c != 0).sum(-1).max()))]  # trim to this tweet's own longest word
        alone = m(words[i:i + 1, :L], c, lengths[i:i + 1])
        assert torch.allclose(full[i:i + 1], alone, atol=1e-5)


def test_char_cnn_gradients_reach_embedding_and_conv_filters_but_not_padding_row():
    torch.manual_seed(3)
    cnn = CharCNN(30, char_dim=6, n_filters=5)
    chars = torch.tensor([[[3, 7, 2, 0, 0, 0], [4, 5, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]])
    cnn(chars).sum().backward()
    assert cnn.embedding.weight.grad[1:].abs().sum() > 0  # real characters are trained
    assert all(conv.weight.grad.abs().sum() > 0 for conv in cnn.convs)  # every kernel size gets gradient
    assert cnn.embedding.weight.grad[0].abs().sum() == 0  # <pad> stays a fixed zero vector


# ----------------------------------------------------------------- config ----
def test_default_json_matches_dataclass_defaults():
    assert BiLSTMConfig.from_json(tb.ROOT / "configs" / "bilstm_default.json") == BiLSTMConfig()


def test_config_validation_and_overrides():
    cfg = BiLSTMConfig().with_overrides({"hidden_size": 64, "char_kernels": [3], "pooling": "last"})
    assert (cfg.hidden_size, cfg.char_kernels, cfg.pooling) == (64, [3], "last")
    for bad in ({"mode": "fancy"}, {"pooling": "x"}, {"dropout": 1.0}, {"hidden_size": 0}, {"char_kernels": []},
                {"use_word": False, "use_char": False}, {"nonexistent_key": 1}):
        with pytest.raises(ValueError):
            BiLSTMConfig().with_overrides(bad)


def test_set_parsing():
    assert tb._parse_set(["hidden_size=256", "pooling=last", "char_kernels=[2,3]", "use_char=false", "max_words=null"]) == {
        "hidden_size": 256, "pooling": "last", "char_kernels": [2, 3], "use_char": False, "max_words": None}


# ------------------------------------------------- synthetic CoNLL dataset ----
# The synthetic dataset, the `data_root` fixture (tests/conftest.py) and the SMALL settings are shared
# with tests/test_bilstm_checkpoint.py; they live in tests/synthetic_conll.py.
def _frames(root, mode="normalized"):
    return load_processed_split("train", mode=mode, root=root), load_processed_split("dev", mode=mode, root=root)


# ------------------------------------------------------ training pipeline ----
def test_training_learns_the_synthetic_task_and_returns_full_metrics(data_root):
    tr, dv = _frames(data_root)
    res = train_one_seed(tr, dv, BiLSTMConfig(**SMALL), seed=42, verbose=False)
    m = res["metrics"]
    assert {"weighted_f1", "macro_f1", "accuracy", "per_class"} <= set(m)
    assert m["weighted_f1"] > 0.85 and m["accuracy"] > 0.85  # chance would be about 0.33
    assert len(res["pred"]) == len(dv) and set(res["pred"]) <= set(LABELS)
    assert sum(map(sum, res["confusion"])) == len(dv)
    assert 1 <= res["best_epoch"] <= res["epochs_run"] <= 8
    shared = derive_shared_limits(list(tr["text"]))  # the limit is shared by all modes and derived from train only
    assert (res["max_words"], res["max_chars"]) == (shared["max_words"], shared["max_chars"])
    assert res["max_words"] >= max(len(t.split()) for t in tr["text_proc"])  # never cuts a train tweet


def test_same_seed_gives_identical_results(data_root):
    tr, dv = _frames(data_root)
    cfg = BiLSTMConfig(**{**SMALL, "epochs": 3})
    a = train_one_seed(tr, dv, cfg, seed=7, verbose=False)
    b = train_one_seed(tr, dv, cfg, seed=7, verbose=False)
    assert a["pred"] == b["pred"] and a["metrics"] == b["metrics"]


def test_word_only_and_char_only_variants_run(data_root):
    tr, dv = _frames(data_root)
    for extra in ({"use_char": False}, {"use_word": False}):
        res = train_one_seed(tr, dv, BiLSTMConfig(**{**SMALL, "epochs": 2, **extra}), seed=1, verbose=False)
        assert len(res["pred"]) == len(dv)


def test_early_stopping_stops_before_max_epochs(data_root):
    tr, dv = _frames(data_root)
    cfg = BiLSTMConfig(**{**SMALL, "epochs": 40, "patience": 2, "lr": 0.05})
    res = train_one_seed(tr, dv, cfg, seed=3, verbose=False)
    assert res["epochs_run"] < 40


def test_dev_only_words_never_enter_the_vocabulary(data_root):
    tr, dv = _frames(data_root)
    enc = TextEncoder.fit(list(tr["text_proc"]), word_min_freq=1)
    # ordinary words are lowercased; the placeholder words USER / URL keep their spelling and are always reserved
    train_words = {w if w in PLACEHOLDER_WORDS else w.lower() for t in tr["text_proc"] for w in t.split()}
    assert set(enc.word_vocab.itos[2:]) == train_words | set(PLACEHOLDER_WORDS)
    ds = SentimentDataset(enc, ["totally unseen words here"], None)
    assert ds[0][0] == [1, 1, 1, 1]  # all <unk>
    assert make_batch([ds[0]])["words"].shape == (1, 4)


def test_run_experiment_saves_results_in_project_format(data_root, tmp_path):
    out = tmp_path / "results"
    out.mkdir()
    (out / "experiment_log.csv").write_text(",".join(tb.LOG_HEADER) + "\nold,row,x,x,x,1,0.5,n,l\n", encoding="utf-8")
    tr, dv = _frames(data_root)
    cfg = BiLSTMConfig(**{**SMALL, "epochs": 2})
    result = run_experiment(tr, dv, cfg, [42, 43], out_dir=out, who="Tester", verbose=False)

    rows = list(csv.DictReader(open(out / "experiment_log.csv", encoding="utf-8")))
    assert len(rows) == 3 and rows[0]["date"] == "old"  # existing row kept, 2 appended
    new = rows[1:]
    assert [r["seed"] for r in new] == ["42", "43"] and all(r["who"] == "Tester" for r in new)
    assert all(0 <= float(r["dev_score"]) <= 1 and "macro_f1=" in r["notes"] and "accuracy=" in r["notes"] for r in new)
    assert new[0]["data_version"] == tb.DATA_VERSIONS["normalized"]

    pred = pd.read_csv(out / "predictions" / "bilstm_bilstm_normalized_seed42_dev.csv")
    assert list(pred.columns) == ["uid", "label", "pred"] and len(pred) == len(dv)
    assert pred["uid"].tolist() == dv["uid"].tolist()  # row order and uids unchanged

    saved = json.loads((out / "bilstm_bilstm_normalized_dev_scores.json").read_text(encoding="utf-8"))
    assert saved["seeds"] == [42, 43] and set(saved["summary"]) == {"weighted_f1", "macro_f1", "accuracy"}
    assert saved["config"]["mode"] == "normalized" and len(saved["per_seed"]) == 2
    assert result["summary"] == saved["summary"]


def test_run_experiment_without_out_dir_writes_nothing(data_root, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tr, dv = _frames(data_root)
    run_experiment(tr, dv, BiLSTMConfig(**{**SMALL, "epochs": 1}), [1], out_dir=None, verbose=False)
    assert list(tmp_path.iterdir()) == [data_root.parent / "Hinglish"]  # only the fixture's folder


# ---------------------------------------------------- the three text modes ----
@pytest.mark.parametrize("mode", ["raw", "clean", "normalized"])
def test_cli_runs_each_mode_and_never_loads_test(data_root, tmp_path, monkeypatch, mode):
    requested = []
    real = tb.load_processed_split

    def spy(split, **kw):
        requested.append((split, kw.get("mode")))
        return real(split, **kw)

    monkeypatch.setattr(tb, "load_processed_split", spy)
    out = tmp_path / "out"
    res = main(["--mode", mode, "--seeds", "5", "--out-dir", str(out), "--data-root", str(data_root), "--who", "T",
                *sum((["--set", f"{k}={json.dumps(v)}"] for k, v in {**SMALL, "epochs": 2}.items()), [])])
    assert requested == [("train", mode), ("dev", mode)]  # test split never requested
    assert res["config"]["mode"] == mode
    assert (out / "experiment_log.csv").exists()


def test_modes_use_identical_settings_and_differ_only_in_text(data_root):
    raw_tr, _ = _frames(data_root, "raw")
    clean_tr, _ = _frames(data_root, "clean")
    norm_tr, _ = _frames(data_root, "normalized")
    assert (raw_tr["uid"].tolist() == clean_tr["uid"].tolist() == norm_tr["uid"].tolist())
    assert (raw_tr["label"].tolist() == norm_tr["label"].tolist())
    assert raw_tr["text_proc"].tolist() == raw_tr["text"].tolist()  # raw = text as loaded
    assert any("https" in t for t in raw_tr["text_proc"]) and not any("https" in t for t in clean_tr["text_proc"])
    assert any(" acha " in f" {t} " for t in clean_tr["text_proc"])  # clean leaves spellings alone
    assert not any(" acha " in f" {t} " for t in norm_tr["text_proc"])  # normalized maps acha -> achha
    assert BiLSTMConfig(mode="raw").with_overrides({"mode": "normalized"}) == BiLSTMConfig(mode="normalized")


def test_limit_flag_saves_nothing(data_root, tmp_path):
    out = tmp_path / "out"
    main(["--limit", "30", "--out-dir", str(out), "--data-root", str(data_root),
          *sum((["--set", f"{k}={json.dumps(v)}"] for k, v in {**SMALL, "epochs": 1}.items()), [])])
    assert not out.exists()
