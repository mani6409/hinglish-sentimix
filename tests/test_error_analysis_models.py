"""Tests for src/error_analysis_models.py on tiny hand-made data (no real dataset needed)."""
import numpy as np
import pandas as pd
import pytest

from src import error_analysis_models as ea
from src.cmi_evaluation import bucket_edges


def preds(uids, labels, predicted):
    return pd.DataFrame({"uid": uids, "label": labels, "pred": predicted})


def test_setting_name_merges_seeds_and_strips_suffix():
    assert ea.setting_name("bilstm_full_raw_s42_seed42_dev") == "bilstm_full_raw"
    assert ea.setting_name("tfidf_char_2-5_dev") == "tfidf_char_2-5"
    assert ea.setting_name("muril_dev_predictions") == "muril"
    assert ea.seed_of("bilstm_full_raw_s43_seed43_dev") == 43


def test_merge_seeds_majority_and_tie_goes_to_first_run():
    a = preds([1, 2], ["neutral", "neutral"], ["positive", "negative"])
    b = preds([1, 2], ["neutral", "neutral"], ["positive", "neutral"])
    c = preds([1, 2], ["neutral", "neutral"], ["neutral", "positive"])
    out = ea.merge_seeds([a, b, c]).set_index("uid")["pred"]
    assert out[1] == "positive"          # 2 of 3
    assert out[2] == "negative"          # 1-1-1 tie -> first run
    assert ea.merge_seeds([a])["n_runs"].iloc[0] == 1


def test_flags_negation_slang_sarcasm_short_and_spelling():
    f = ea.flag_row("yeh acha nahi hai", ["yeh", "acha", "nahi", "hai"], "negative", "neutral", True, False)
    assert f["negation"] and f["spelling_variant"] and f["very_short"] and not f["sarcasm_cue"]
    f = ea.flag_row("waah kya baat hai 😂", ["a"] * 20, "negative", "positive", False, True)
    assert f["sarcasm_cue"] and f["high_cmi"] and not f["very_short"]
    # sarcasm cue needs a negative gold label
    assert not ea.flag_row("waah great", ["a"] * 20, "positive", "positive", False, False)["sarcasm_cue"]
    assert ea.flag_row("soooo bad", ["a"] * 20, "negative", "x", False, False)["slang_informal"]
    assert ea.flag_row("THIS IS BAD", ["a"] * 20, "negative", "x", False, False)["slang_informal"]
    assert not ea.flag_row("this is bad", ["a"] * 20, "negative", "x", False, False)["slang_informal"]


def test_anonymise_replaces_mentions():
    assert ea.anonymise("hi @ sherryontopp ok @bob") == "hi @USER ok @USER"


def test_load_predictions_rejects_bad_files(tmp_path):
    dev = pd.DataFrame({"uid": [1, 2], "label": ["neutral", "positive"]})
    ok = tmp_path / "m_dev.csv"
    preds([1, 2], ["neutral", "positive"], ["neutral", "neutral"]).to_csv(ok, index=False)
    assert ea.load_predictions([ok], dev)["m"].shape[0] == 2
    dup = tmp_path / "d_dev.csv"
    preds([1, 1], ["neutral", "neutral"], ["neutral", "neutral"]).to_csv(dup, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        ea.load_predictions([dup], dev)
    unk = tmp_path / "u_dev.csv"
    preds([1, 3], ["neutral", "neutral"], ["neutral", "neutral"]).to_csv(unk, index=False)
    with pytest.raises(ValueError, match="not dev uids"):
        ea.load_predictions([unk], dev)
    lab = tmp_path / "l_dev.csv"
    preds([1, 2], ["neutral", "negative"], ["neutral", "neutral"]).to_csv(lab, index=False)
    with pytest.raises(ValueError, match="gold labels differ"):
        ea.load_predictions([lab], dev)
    badpred = tmp_path / "b_dev.csv"
    preds([1, 2], ["neutral", "positive"], ["neutral", "great"]).to_csv(badpred, index=False)
    with pytest.raises(ValueError, match="unknown predicted"):
        ea.load_predictions([badpred], dev)


def _tiny():
    dev = pd.DataFrame({
        "uid": [1, 2, 3, 4], "label": ["neutral", "positive", "negative", "neutral"],
        "text": ["a nahi b", "great @ user1", "bad", "ok"], "tokens": [["a", "nahi", "b"], ["g"], ["b"], ["o"]],
    })
    cmi = pd.DataFrame({"uid": [1, 2, 3, 4], "cmi": [10.0, 40.0, 20.0, 50.0]})
    edges = bucket_edges(pd.Series(np.arange(0, 60, 5.0)))
    s = {
        "m1": preds([1, 2, 3, 4], dev["label"], ["neutral", "neutral", "positive", "neutral"]),
        "m2": preds([1, 2, 3, 4], dev["label"], ["neutral", "neutral", "positive", "negative"]),
    }
    for df in s.values():
        df["n_runs"] = 1
    return dev, s, cmi, edges


def test_error_table_extracts_exactly_the_misclassified_tweets():
    dev, s, cmi, edges = _tiny()
    table, wrong = ea.build_error_table(dev, s, cmi, edges)
    assert sorted(table["uid"]) == [2, 3, 4]            # uid 1 is right everywhere
    row = table.set_index("uid")
    assert row.at[2, "n_settings_wrong"] == 2 and row.at[4, "n_settings_wrong"] == 1
    assert row.at[3, "most_common_wrong_pred"] == "positive"
    assert row.at[4, "wrong_settings"] == "m2"
    assert row.at[2, "text"] == "great @USER"            # usernames anonymised in the output
    assert wrong.loc[1].sum() == 0


def test_agreement_only_counts_commonly_covered_tweets():
    wrong = pd.DataFrame({"a": [True, False, np.nan], "b": [True, True, True]}, index=[1, 2, 3])
    t = ea.agreement_table(wrong).iloc[0]
    assert t["n_common_tweets"] == 2 and t["wrong_both"] == 1 and t["jaccard"] == 0.5


def test_wilson_interval_contains_rate_and_is_bounded():
    lo, hi = ea.wilson(30, 100)
    assert 0 <= lo < 0.30 < hi <= 1
    assert np.isnan(ea.wilson(0, 0)[0])


def test_cmi_error_rates_counts():
    dev, s, cmi, edges = _tiny()
    r = ea.cmi_error_rates(s, cmi, edges)
    assert r.groupby("setting")["n"].sum().tolist() == [4, 4]
    assert r.groupby("setting")["errors"].sum().tolist() == [2, 3]


def test_manual_sample_has_empty_hand_label_columns():
    dev, s, cmi, edges = _tiny()
    table, _ = ea.build_error_table(dev, s, cmi, edges)
    summary = ea.model_summary(s, {"m1": 0.5, "m2": 0.4})
    sample = ea.manual_sample(table, summary, n=3)
    assert (sample["manual_category"] == "").all() and len(sample) >= 1
    assert sample["best_setting"].nunique() == 1
