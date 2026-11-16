"""Review item I1: best-model checkpoints (creation, loading, metadata, safety)."""
import csv
import json

import pytest

torch = pytest.importorskip("torch")

import src.bilstm_checkpoint as ck  # noqa: E402
from src.bilstm_checkpoint import checkpoint_path, load_checkpoint  # noqa: E402
from src.bilstm_data import PLACEHOLDER_WORDS  # noqa: E402
from src.metrics import compute_metrics  # noqa: E402
from src.preprocess import PRESETS, load_processed_split  # noqa: E402
from src.train_bilstm import BiLSTMConfig, main, run_experiment  # noqa: E402
from tests.synthetic_conll import SMALL  # noqa: E402

CFG = {**SMALL, "epochs": 3}


def _frames(root, mode="normalized"):
    return load_processed_split("train", mode=mode, root=root), load_processed_split("dev", mode=mode, root=root)


@pytest.fixture()
def trained(data_root, tmp_path):
    """One real (tiny) run of two seeds with checkpoints written under tmp_path/results."""
    out = tmp_path / "results"
    tr, dv = _frames(data_root)
    cfg = BiLSTMConfig(**CFG)
    res = run_experiment(tr, dv, cfg, [42, 43], out_dir=out, verbose=False)
    return {"out": out, "tr": tr, "dv": dv, "cfg": cfg, "res": res}


def _strings(obj):
    """Every dict key and string value inside a nested checkpoint payload."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from _strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _strings(v)
    elif isinstance(obj, str):
        yield obj


# ------------------------------------------------------------ creation --------
def test_checkpoints_are_written_to_predictable_separate_paths(trained):
    out = trained["out"]
    for seed in (42, 43):
        p = checkpoint_path(out, "bilstm_normalized", seed)
        assert p == out / "checkpoints" / f"bilstm_bilstm_normalized_seed{seed}.pt" and p.is_file() and p.stat().st_size > 0
    assert sorted(f.name for f in (out / "checkpoints").iterdir()) == [
        "bilstm_bilstm_normalized_seed42.pt", "bilstm_bilstm_normalized_seed43.pt"]  # no leftover .tmp files
    assert all(r["checkpoint"] for r in trained["res"]["per_seed"])  # the result records where each one went


def test_different_modes_get_different_checkpoint_paths(data_root, tmp_path):
    out = tmp_path / "results"
    for m in ("raw", "clean"):
        tr, dv = _frames(data_root, m)
        run_experiment(tr, dv, BiLSTMConfig(**{**CFG, "mode": m, "epochs": 1}), [7], out_dir=out, verbose=False)
    assert sorted(f.name for f in (out / "checkpoints").iterdir()) == [
        "bilstm_bilstm_clean_seed7.pt", "bilstm_bilstm_raw_seed7.pt"]
    assert load_checkpoint(out / "checkpoints" / "bilstm_bilstm_raw_seed7.pt").config.mode == "raw"
    assert load_checkpoint(out / "checkpoints" / "bilstm_bilstm_clean_seed7.pt").config.mode == "clean"


# ------------------------------------------------- loading and consistency ----
def test_loaded_checkpoint_reproduces_the_saved_dev_predictions_and_scores(trained):
    out, dv = trained["out"], trained["dv"]
    ckpt = load_checkpoint(checkpoint_path(out, "bilstm_normalized", 42))
    saved = list(csv.DictReader(open(out / "predictions" / "bilstm_bilstm_normalized_seed42_dev.csv", encoding="utf-8")))
    pred = ckpt.predict_texts(list(dv["text_proc"]))
    assert pred == [r["pred"] for r in saved]  # identical predictions to the training run
    m = compute_metrics(list(dv["label"]), pred)
    assert m["weighted_f1"] == pytest.approx(ckpt.meta["dev_metrics"]["weighted_f1"])
    assert m["macro_f1"] == pytest.approx(ckpt.meta["dev_metrics"]["macro_f1"])
    assert m["accuracy"] == pytest.approx(ckpt.meta["dev_metrics"]["accuracy"])


def test_inference_from_original_tweets_applies_the_checkpoints_own_preprocessing(trained):
    ckpt = load_checkpoint(checkpoint_path(trained["out"], "bilstm_normalized", 43))
    dv = trained["dv"]
    assert ckpt.predict_texts(list(dv["text"]), preprocess=True) == ckpt.predict_texts(list(dv["text_proc"]))
    assert ckpt.predict_texts(["", "totally unseen words"], preprocess=True)  # edge inputs still give labels
    assert set(ckpt.predict_texts(list(dv["text"]), preprocess=True)) <= {"negative", "neutral", "positive"}


def test_loaded_model_is_in_eval_mode_and_deterministic(trained):
    ckpt = load_checkpoint(checkpoint_path(trained["out"], "bilstm_normalized", 42))
    assert not ckpt.model.training
    texts = list(trained["dv"]["text_proc"])[:20]
    assert ckpt.predict_texts(texts) == ckpt.predict_texts(texts)


def test_metadata_matches_the_run(trained):
    res, cfg = trained["res"], trained["cfg"]
    for r in res["per_seed"]:
        ckpt = load_checkpoint(r["checkpoint"])
        meta = ckpt.meta
        assert meta["mode"] == "normalized" == ckpt.config.mode
        assert meta["seed"] == r["seed"] and meta["best_epoch"] == r["best_epoch"] and meta["epochs_run"] == r["epochs_run"]
        assert meta["selected_on"] == "dev weighted_f1"
        assert meta["dev_metrics"]["weighted_f1"] == pytest.approx(r["metrics"]["weighted_f1"])
        assert meta["dev_metrics"]["per_class"].keys() == {"negative", "neutral", "positive"}
        assert meta["preprocess_options"] == PRESETS["normalized"]
        assert meta["limits"]["max_words"] == res["limits"]["max_words"] == ckpt.encoder.max_words
        assert meta["limits"]["max_chars"] == res["limits"]["max_chars"] == ckpt.encoder.max_chars
        assert (meta["train_size"], meta["dev_size"]) == (len(trained["tr"]), len(trained["dv"]))
        assert meta["n_params"] == r["n_params"] and meta["truncation"] == r["truncation"]
        assert meta["variants_sha256"] == ck.variants_fingerprint()
        assert ckpt.config == cfg  # every setting restored
        assert len(ckpt.encoder.word_vocab) == r["word_vocab_size"] and len(ckpt.encoder.char_vocab) == r["char_vocab_size"]


def test_vocabularies_in_the_checkpoint_come_from_train_only(trained):
    ckpt = load_checkpoint(checkpoint_path(trained["out"], "bilstm_normalized", 42))
    train_words = {w if w in PLACEHOLDER_WORDS else w.lower() for t in trained["tr"]["text_proc"] for w in t.split()}
    assert set(ckpt.encoder.word_vocab.itos[2:]) == train_words | set(PLACEHOLDER_WORDS)


def test_checkpoint_holds_no_test_information(trained):
    payload = torch.load(checkpoint_path(trained["out"], "bilstm_normalized", 42), weights_only=True)
    assert not [s for s in _strings(payload) if "test" in s.lower()]  # no key or string mentions the test split
    assert set(payload) == set(ck.REQUIRED_KEYS)
    assert payload["labels"] == ["negative", "neutral", "positive"]
    json.dumps({k: v for k, v in payload.items() if k != "model_state"})  # everything but the weights is plain data


# ----------------------------------------------------------- loader safety ----
def test_load_rejects_files_that_are_not_checkpoints_or_have_a_wrong_format(tmp_path):
    bad = tmp_path / "bad.pt"
    torch.save({"x": 1}, bad)
    with pytest.raises(ValueError, match="not a BiLSTM checkpoint"):
        load_checkpoint(bad)


def test_load_rejects_unknown_format_version(trained, tmp_path):
    p = checkpoint_path(trained["out"], "bilstm_normalized", 42)
    payload = torch.load(p, weights_only=True)
    payload["format_version"] = 99
    torch.save(payload, tmp_path / "v99.pt")
    with pytest.raises(ValueError, match="unsupported"):
        load_checkpoint(tmp_path / "v99.pt")


def test_load_warns_when_the_hinglish_dictionary_changed(trained, monkeypatch):
    monkeypatch.setattr(ck, "variants_fingerprint", lambda: "different")
    with pytest.warns(UserWarning, match="hinglish_variants"):
        load_checkpoint(checkpoint_path(trained["out"], "bilstm_normalized", 42))


# ------------------------------------------------ overwrite and opt-out -------
def test_existing_checkpoints_are_not_overwritten_silently(trained):
    out, tr, dv, cfg = trained["out"], trained["tr"], trained["dv"], trained["cfg"]
    p = checkpoint_path(out, "bilstm_normalized", 42)
    before, log_before = p.read_bytes(), (out / "experiment_log.csv").read_text(encoding="utf-8")
    with pytest.raises(FileExistsError, match="nothing was trained"):
        run_experiment(tr, dv, cfg, [42, 99], out_dir=out, verbose=False)
    assert p.read_bytes() == before and not checkpoint_path(out, "bilstm_normalized", 99).exists()  # nothing written
    assert (out / "experiment_log.csv").read_text(encoding="utf-8") == log_before  # and nothing logged
    run_experiment(tr, dv, cfg, [42], out_dir=out, verbose=False, overwrite=True)  # explicit opt-in works
    assert p.is_file()


def test_checkpoints_can_be_switched_off_and_are_not_written_without_out_dir(data_root, tmp_path):
    tr, dv = _frames(data_root)
    cfg = BiLSTMConfig(**{**CFG, "epochs": 1})
    out = tmp_path / "r1"
    res = run_experiment(tr, dv, cfg, [1], out_dir=out, verbose=False, save_checkpoints=False)
    assert not (out / "checkpoints").exists() and res["per_seed"][0]["checkpoint"] is None
    res2 = run_experiment(tr, dv, cfg, [1], out_dir=None, verbose=False)
    assert res2["per_seed"][0]["checkpoint"] is None


def test_cli_saves_checkpoint_and_respects_flags(data_root, tmp_path):
    args = ["--mode", "normalized", "--seeds", "5", "--data-root", str(data_root), "--who", "T",
            *sum((["--set", f"{k}={json.dumps(v)}"] for k, v in {**SMALL, "epochs": 1}.items()), [])]
    out = tmp_path / "out"
    main([*args, "--out-dir", str(out)])
    p = checkpoint_path(out, "bilstm_normalized", 5)
    assert p.is_file()
    with pytest.raises(FileExistsError):
        main([*args, "--out-dir", str(out)])
    main([*args, "--out-dir", str(out), "--overwrite"])
    out2 = tmp_path / "out2"
    main([*args, "--out-dir", str(out2), "--no-checkpoints"])
    assert not (out2 / "checkpoints").exists()
