"""Tests for src/zero_shot_llm.py with fake backends (no model, no network, no API key)."""
import pandas as pd
import pytest

from src import zero_shot_llm as z


@pytest.mark.parametrize("text,label,valid", [
    ("positive", "positive", True),
    ("Negative.", "negative", True),
    ("  'NEUTRAL' \n", "neutral", True),
    ("Sentiment: positive", "positive", True),
    ("Positive. The tweet is positive.", "positive", True),
    ("positive or negative", "neutral", False),   # ambiguous -> invalid, fallback label
    ("I cannot help with that.", "neutral", False),
    ("", "neutral", False),
    (None, "neutral", False),
    ("unpositive", "neutral", False),             # word boundary: not the word 'positive'
])
def test_parse_response(text, label, valid):
    got_label, got_valid, _ = z.parse_response(text)
    assert (got_label, got_valid) == (label, valid)


def test_prompt_is_zero_shot_and_contains_tweet_and_all_labels():
    for v in z.PROMPTS:
        p = z.build_prompt("yeh movie bakwas thi", v)
        assert "yeh movie bakwas thi" in p
        assert all(lab in p for lab in ("positive", "negative", "neutral"))
        assert "Tweet:" in p and p.count("Tweet:") == 1  # a single tweet, no labelled examples
    assert "Hinglish" in z.build_prompt("x", "v2")
    with pytest.raises(ValueError):
        z.build_prompt("x", "v9")


def test_classify_handles_backend_failure_without_crashing():
    def boom(prompt):
        raise RuntimeError("rate limit")
    r = z.classify("tweet", boom)
    assert r["valid"] is False and r["parsed"] == z.FALLBACK_LABEL and "RuntimeError" in r["reason"]
    assert z.classify("tweet", lambda p: "Negative")["parsed"] == "negative"


def _subset():
    return pd.DataFrame({"uid": [1, 2, 3, 4], "label": ["positive", "negative", "neutral", "neutral"],
                         "text": ["a", "b", "c", "d"]})


def test_run_resumes_and_evaluates(tmp_path):
    calls = []
    answers = {"a": "positive", "b": "negative", "c": "hmm maybe", "d": "neutral"}

    def backend(prompt):
        tweet = prompt.split("Tweet: ")[1].split("\n")[0]
        calls.append(tweet)
        return answers[tweet]

    raw_path = tmp_path / "raw.csv"
    sub = _subset()
    z.run(sub.iloc[:2], backend, "v2", raw_path, save_every=1)
    assert calls == ["a", "b"]
    raw = z.run(sub, backend, "v2", raw_path, save_every=1)   # resume: only c and d are new
    assert calls == ["a", "b", "c", "d"] and len(raw) == 4
    metrics, merged = z.evaluate(sub, raw)
    assert metrics["n_invalid"] == 1 and metrics["invalid_rate"] == 0.25
    assert metrics["all_answers"]["accuracy"] == 1.0          # invalid 'c' falls back to neutral, which is correct
    assert metrics["valid_answers_only"]["accuracy"] == 1.0
    assert list(merged["parsed"]) == ["positive", "negative", "neutral", "neutral"]


def test_run_refuses_to_mix_prompt_versions(tmp_path):
    raw_path = tmp_path / "raw.csv"
    z.run(_subset().iloc[:1], lambda p: "positive", "v1", raw_path, save_every=1)
    with pytest.raises(SystemExit):
        z.run(_subset(), lambda p: "positive", "v2", raw_path)


def test_sample_dev_is_reproducible_and_sorted():
    dev = pd.DataFrame({"uid": range(100), "label": ["neutral"] * 100, "text": ["x"] * 100})
    a, b = z.sample_dev(dev, 10), z.sample_dev(dev, 10)
    assert list(a["uid"]) == list(b["uid"]) == sorted(a["uid"]) and len(set(a["uid"])) == 10
    assert len(z.sample_dev(dev, None)) == 100


def test_anthropic_backend_needs_key_from_environment(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        z.make_anthropic_backend("any-model")
