"""Tests for the shared transformer pipeline (src.transformer_utils, src.train_transformer).

No download is needed: a tiny BERT (1 layer, hidden size 32) with a WordPiece vocabulary built from the
synthetic tweets is saved to a temp folder and passed as `model_name`, exactly like a Hub checkpoint.
"""
import csv
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast  # noqa: E402

from src.metrics import LABELS  # noqa: E402
from src.preprocess import load_processed_split  # noqa: E402
from src.train_transformer import (  # noqa: E402
    checkpoint_dir,
    main,
    prediction_path,
    resolve_max_length,
    run_experiment,
    scores_path,
)
from src.transformer_utils import (  # noqa: E402
    EncodedTweets,
    TransformerConfig,
    count_truncated,
    derive_max_length,
    load_checkpoint_dir,
    make_collate,
    model_slug,
    predict_ids,
    resolve_model_name,
    token_lengths,
)
from tests.synthetic_conll import FILLER, KEYWORDS  # noqa: E402

ROOT_CONFIG = "configs/transformer_config.json"


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    """A randomly initialised 1-layer BERT + tokenizer saved like a pretrained checkpoint."""
    d = tmp_path_factory.mktemp("tiny_bert")
    words = sorted({w for ws in KEYWORDS.values() for w in ws} | set(FILLER) |
                   {"url", "user", "bohot", "acha", "hai", "@", "/", ".", "#", ":"})
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + words + [f"##{c}" for c in "abcdefghijklmnopqrstuvwxyz"]
    vocab += list("abcdefghijklmnopqrstuvwxyz0123456789")
    tok = BertTokenizerFast(vocab={w: i for i, w in enumerate(dict.fromkeys(vocab))}, do_lower_case=True,
                            model_max_length=64)
    assert tok.tokenize("mast hai")[0] == "mast"  # guard against an all-[UNK] tokenizer
    cfg = BertConfig(vocab_size=tok.vocab_size, hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                     intermediate_size=64, max_position_embeddings=64, num_labels=3)
    torch.manual_seed(0)
    BertForSequenceClassification(cfg).save_pretrained(d)
    tok.save_pretrained(d)
    return str(d)


@pytest.fixture()
def frames(data_root):
    return (load_processed_split("train", mode="clean", root=data_root),
            load_processed_split("dev", mode="clean", root=data_root))


def small_cfg(model, **kw):
    base = dict(model_name=model, epochs=10, patience=10, batch_size=16, lr=1e-3, warmup_ratio=0.0,
                device="cpu", fp16=False)
    return TransformerConfig(**{**base, **kw}).validate()


# ------------------------------------------------------------------ config ----
def test_repo_config_loads_and_matches_defaults():
    cfg = TransformerConfig.from_json(ROOT_CONFIG)
    assert cfg == TransformerConfig()  # the JSON documents the defaults


@pytest.mark.parametrize("bad", [{"mode": "fancy"}, {"batch_size": 0}, {"max_length": 4}, {"warmup_ratio": 1.0},
                                 {"device": "tpu"}, {"length_percentile": 0}])
def test_config_rejects_bad_values(bad):
    with pytest.raises(ValueError):
        TransformerConfig().with_overrides(bad)


def test_config_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown config keys"):
        TransformerConfig().with_overrides({"learning_rate": 1e-5})


def test_aliases():
    assert resolve_model_name("mbert") == "bert-base-multilingual-cased"
    assert resolve_model_name("xlmr") == "xlm-roberta-base"
    assert resolve_model_name("some/other-model") == "some/other-model"
    assert model_slug("xlm-roberta-base") == "xlmr"
    assert model_slug("C:/tmp/tiny_bert0") == "tiny_bert0"


# ---------------------------------------------------------------- lengths ----
def test_derive_max_length_rounds_up_and_caps():
    assert derive_max_length([10, 17, 30]) == 32
    assert derive_max_length([10, 17, 33]) == 40
    assert derive_max_length([10] * 99 + [400], percentile=99) <= 16
    assert derive_max_length([900], model_limit=512) == 512


def test_count_truncated():
    assert count_truncated([5, 10, 12], 10) == {"tweets": 1, "share": round(1 / 3, 5), "subwords_cut": 2}


def test_max_length_from_train_covers_all_train_tweets(tiny_model, frames):
    from src.transformer_utils import load_tokenizer

    tok = load_tokenizer(tiny_model)
    info = resolve_max_length(tok, list(frames[0]["text_proc"]), small_cfg(tiny_model))
    assert info["max_length"] >= info["train_max"]
    assert info["max_length"] % 8 == 0
    assert info["train_truncated"]["tweets"] == 0
    fixed = resolve_max_length(tok, list(frames[0]["text_proc"]), small_cfg(tiny_model, max_length=16))
    assert fixed["max_length"] == 16 and fixed["source"] == "config"


# ------------------------------------------------------- tokens and batches ----
def test_encoding_truncates_and_pads_per_batch(tiny_model):
    from src.transformer_utils import load_tokenizer

    tok = load_tokenizer(tiny_model)
    texts = ["mast " * 30, "bakwas hai"]
    ds = EncodedTweets(tok, texts, ["positive", "negative"], max_length=12)
    assert len(ds[0]["input_ids"]) == 12 and ds[0]["input_ids"][-1] == tok.sep_token_id
    assert len(ds[1]["input_ids"]) == 4  # [CLS] bakwas hai [SEP], not padded yet
    batch = make_collate(tok)([ds[0], ds[1]])
    assert batch["input_ids"].shape == (2, 12)
    assert batch["attention_mask"][1].tolist() == [1] * 4 + [0] * 8
    assert batch["labels"].tolist() == [LABELS.index("positive"), LABELS.index("negative")]
    assert token_lengths(tok, texts) == [32, 4]


def test_encoding_rejects_unknown_labels(tiny_model):
    from src.transformer_utils import load_tokenizer

    with pytest.raises(ValueError, match="unknown labels"):
        EncodedTweets(load_tokenizer(tiny_model), ["x"], ["happy"], max_length=8)


# ------------------------------------------------------------- training ----
def test_training_learns_and_writes_common_outputs(tiny_model, frames, tmp_path):
    tr, dv = frames
    res = run_experiment(tr, dv, small_cfg(tiny_model), [1, 2], out_dir=tmp_path, run_name="tiny", verbose=False)

    assert res["summary"]["weighted_f1"]["mean"] > 0.6  # keywords decide the label; chance is about 0.33
    for seed in (1, 2):
        rows = list(csv.DictReader(open(prediction_path(tmp_path, "tiny", seed), encoding="utf-8")))
        assert list(rows[0]) == ["uid", "label", "pred"]
        assert [int(r["uid"]) for r in rows] == list(dv["uid"])
        assert {r["pred"] for r in rows} <= set(LABELS)
        assert checkpoint_dir(tmp_path, "tiny", seed).joinpath("run_info.json").exists()
    scores = json.loads(scores_path(tmp_path, "tiny").read_text(encoding="utf-8"))
    assert scores["seeds"] == [1, 2] and len(scores["per_seed"]) == 2
    log = list(csv.DictReader(open(tmp_path / "experiment_log.csv", encoding="utf-8", newline="")))
    assert [r["seed"] for r in log] == ["1", "2"] and all(r["notes"].startswith("run=tiny;") for r in log)
    assert not list(tmp_path.glob(".staging_*"))


def test_checkpoint_reloads_to_the_same_predictions(tiny_model, frames, tmp_path):
    tr, dv = frames
    run_experiment(tr, dv, small_cfg(tiny_model, epochs=2), [3], out_dir=tmp_path, run_name="tiny", verbose=False)
    model, tok, info = load_checkpoint_dir(checkpoint_dir(tmp_path, "tiny", 3))
    ds = EncodedTweets(tok, list(dv["text_proc"]), None, info["max_length"])
    loader = torch.utils.data.DataLoader(ds, batch_size=32, collate_fn=make_collate(tok))
    preds = [LABELS[i] for i in predict_ids(model, loader, torch.device("cpu"))]
    saved = [r["pred"] for r in csv.DictReader(open(prediction_path(tmp_path, "tiny", 3), encoding="utf-8"))]
    assert preds == saved
    assert info["seed"] == 3 and info["selected_on"] == "dev weighted_f1"


def test_same_seed_same_result(tiny_model, frames):
    tr, dv = frames
    a = run_experiment(tr, dv, small_cfg(tiny_model, epochs=2), [5], verbose=False)
    b = run_experiment(tr, dv, small_cfg(tiny_model, epochs=2), [5], verbose=False)
    assert a["per_seed"][0]["history"] == b["per_seed"][0]["history"]


def test_existing_outputs_stop_before_training(tiny_model, frames, tmp_path, monkeypatch):
    tr, dv = frames
    prediction_path(tmp_path, "tiny", 1).parent.mkdir(parents=True)
    prediction_path(tmp_path, "tiny", 1).write_text("keep me", encoding="utf-8")
    import src.train_transformer as tt

    monkeypatch.setattr(tt, "train_one_seed", lambda *a, **k: pytest.fail("must not train"))
    with pytest.raises(FileExistsError, match="already exist"):
        run_experiment(tr, dv, small_cfg(tiny_model), [1], out_dir=tmp_path, run_name="tiny", verbose=False)
    assert prediction_path(tmp_path, "tiny", 1).read_text(encoding="utf-8") == "keep me"


def test_cli_debug_run_writes_nothing(tiny_model, data_root, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    res = main(["--model", tiny_model, "--data-root", str(data_root), "--out-dir", str(out), "--limit", "60",
                "--set", "epochs=1", "--set", "device=cpu", "--set", "batch_size=16"])
    assert res["seeds"] == [42]
    assert list(out.iterdir()) == []
