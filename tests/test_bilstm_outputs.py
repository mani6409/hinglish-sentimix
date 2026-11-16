"""Review item I6: experiment outputs are never overwritten by accident.

Scenarios: outputs already exist, outputs partly exist, explicit overwrite, a run that fails midway
(including multi-seed runs and Ctrl+C), a conflicting file appearing while a run trains, and the
append-only experiment log.
"""
import csv
import json

import pytest

pytest.importorskip("torch")

import src.train_bilstm as tb  # noqa: E402
from src.bilstm_checkpoint import checkpoint_path  # noqa: E402
from src.preprocess import load_processed_split  # noqa: E402
from src.train_bilstm import (  # noqa: E402
    LOG_HEADER,
    BiLSTMConfig,
    append_log_rows,
    find_existing_outputs,
    main,
    prediction_path,
    run_experiment,
    scores_path,
)
from tests.synthetic_conll import SMALL  # noqa: E402

CFG = {**SMALL, "epochs": 1}
RUN = "bilstm_normalized"


@pytest.fixture()
def frames(data_root):
    return (load_processed_split("train", mode="normalized", root=data_root),
            load_processed_split("dev", mode="normalized", root=data_root))


def run(frames, out, seeds=(1,), cfg=None, **kw):
    tr, dv = frames
    return run_experiment(tr, dv, cfg or BiLSTMConfig(**CFG), list(seeds), out_dir=out, verbose=False, **kw)


def snapshot(out):
    """Every file under `out` with its bytes."""
    return {p.relative_to(out).as_posix(): p.read_bytes() for p in out.rglob("*") if p.is_file()}


def log_rows(out):
    return list(csv.DictReader(open(out / "experiment_log.csv", encoding="utf-8", newline="")))


@pytest.fixture()
def train_calls(monkeypatch):
    """Counts the seeds that were actually trained."""
    calls = []
    real = tb.train_one_seed

    def wrapper(tr, dv, cfg, seed, **kw):
        calls.append(seed)
        return real(tr, dv, cfg, seed, **kw)

    monkeypatch.setattr(tb, "train_one_seed", wrapper)
    return calls


def fail_on_seed(monkeypatch, bad_seed, exc):
    real = tb.train_one_seed

    def wrapper(tr, dv, cfg, seed, **kw):
        if seed == bad_seed:
            raise exc("boom")
        return real(tr, dv, cfg, seed, **kw)

    monkeypatch.setattr(tb, "train_one_seed", wrapper)


# ------------------------------------------------------------ a clean run -----
def test_a_new_run_publishes_all_outputs_and_leaves_no_temporary_files(frames, tmp_path):
    out = tmp_path / "results"
    res = run(frames, out, seeds=(1, 2))
    files = set(snapshot(out))
    assert files == {
        f"bilstm_{RUN}_dev_scores.json", f"predictions/bilstm_{RUN}_seed1_dev.csv",
        f"predictions/bilstm_{RUN}_seed2_dev.csv", f"checkpoints/bilstm_{RUN}_seed1.pt",
        f"checkpoints/bilstm_{RUN}_seed2.pt", "experiment_log.csv"}  # no staging, archive, failed_runs or .tmp files
    rows = log_rows(out)
    assert [r["seed"] for r in rows] == ["1", "2"]
    assert all(r["notes"].startswith(f"run={RUN}; macro_f1=") and "rerun" not in r["notes"] for r in rows)
    assert res["archived"] == []
    saved = json.loads(scores_path(out, RUN).read_text(encoding="utf-8"))
    assert [p["checkpoint"] for p in saved["per_seed"]] == [str(checkpoint_path(out, RUN, s)) for s in (1, 2)]  # final paths


# ------------------------------------------------------ existing outputs ------
def test_rerun_with_the_same_name_is_refused_before_training_and_changes_nothing(frames, tmp_path, train_calls):
    out = tmp_path / "results"
    run(frames, out, seeds=(1, 2))
    before, trained = snapshot(out), list(train_calls)
    with pytest.raises(FileExistsError, match="nothing was trained or written") as e:
        run(frames, out, seeds=(1, 2))
    assert train_calls == trained  # not a single seed was trained
    assert snapshot(out) == before  # every byte of every file, including the log, is unchanged
    msg = str(e.value)
    for p in (scores_path(out, RUN), prediction_path(out, RUN, 1), prediction_path(out, RUN, 2),
              checkpoint_path(out, RUN, 1), checkpoint_path(out, RUN, 2)):
        assert str(p) in msg  # the message names every existing output
    assert "row of run 'bilstm_normalized', seed 1" in msg and "--overwrite" in msg and "--run-name" in msg


def test_other_run_names_and_other_directories_are_not_blocked(frames, tmp_path):
    out = tmp_path / "results"
    run(frames, out, seeds=(1,))
    before = snapshot(out)
    run(frames, out, seeds=(1,), run_name="other_name")  # different name, same folder: fine
    run(frames, tmp_path / "elsewhere", seeds=(1,))  # same name, different folder: fine
    after = snapshot(out)
    assert all(after[k] == v for k, v in before.items() if k != "experiment_log.csv")  # earlier files untouched
    assert after["experiment_log.csv"].startswith(before["experiment_log.csv"])  # log only grew
    assert [r["notes"].split(";")[0] for r in log_rows(out)] == [f"run={RUN}", "run=other_name"]


def test_new_seeds_under_an_existing_run_name_are_refused_because_the_scores_file_belongs_to_the_run(frames, tmp_path):
    out = tmp_path / "results"
    run(frames, out, seeds=(1,))
    with pytest.raises(FileExistsError, match="dev_scores.json"):
        run(frames, out, seeds=(2,))
    run(frames, out, seeds=(2,), run_name=RUN + "_s2")  # the supported way: another run name


# ------------------------------------------------------- partial outputs ------
LEFTOVERS = ["scores", "prediction_seed2", "checkpoint_seed1", "log_row_seed2"]


@pytest.mark.parametrize("leftover", LEFTOVERS)
def test_a_single_leftover_output_blocks_the_whole_multi_seed_run(frames, tmp_path, train_calls, leftover):
    """E.g. an earlier run crashed while publishing, or a file was copied in by hand."""
    out = tmp_path / "results"
    out.mkdir()
    if leftover == "scores":
        scores_path(out, RUN).write_text("{}", encoding="utf-8")
    elif leftover == "prediction_seed2":
        prediction_path(out, RUN, 2).parent.mkdir()
        prediction_path(out, RUN, 2).write_text("uid,label,pred\n", encoding="utf-8")
    elif leftover == "checkpoint_seed1":
        checkpoint_path(out, RUN, 1).parent.mkdir()
        checkpoint_path(out, RUN, 1).write_bytes(b"precious")
    else:  # only a log row of this run (seed 2) plus a row of another run
        append_log_rows(out / "experiment_log.csv", [
            ["2026-01-01", "x", "m", "d", "s", 2, "0.5", f"run={RUN}; macro_f1=0.5", ""],
            ["2026-01-01", "x", "m", "d", "s", 2, "0.5", "run=someone_else; macro_f1=0.5", ""]])
    before = snapshot(out)
    with pytest.raises(FileExistsError, match="nothing was trained or written"):
        run(frames, out, seeds=(1, 2))
    assert train_calls == []  # seed 1 was not trained either
    assert snapshot(out) == before  # the leftover is intact and nothing else was created (no staging folder)


def test_other_runs_log_rows_do_not_block(frames, tmp_path):
    out = tmp_path / "results"
    out.mkdir()
    append_log_rows(out / "experiment_log.csv", [
        ["2026-01-01", "Mani", "tfidf", "raw", "C=1", 42, "0.61", "best of C", ""],  # a row without run=, like the TF-IDF ones
        ["2026-01-01", "x", "m", "d", "s", 1, "0.5", "run=someone_else; macro_f1=0.5", ""]])
    run(frames, out, seeds=(1,))
    assert len(log_rows(out)) == 3


def test_checkpoint_files_do_not_block_when_checkpoints_are_switched_off(frames, tmp_path):
    out = tmp_path / "results"
    checkpoint_path(out, RUN, 1).parent.mkdir(parents=True)
    checkpoint_path(out, RUN, 1).write_bytes(b"precious")
    assert find_existing_outputs(out, RUN, [1], save_checkpoints=False) == []
    assert find_existing_outputs(out, RUN, [1], save_checkpoints=True) == [str(checkpoint_path(out, RUN, 1))]
    run(frames, out, seeds=(1,), save_checkpoints=False)
    assert checkpoint_path(out, RUN, 1).read_bytes() == b"precious"


# ---------------------------------------------------------- overwrite ---------
def test_overwrite_archives_the_old_files_publishes_new_ones_and_keeps_the_log_history(frames, tmp_path):
    out = tmp_path / "results"
    run(frames, out, seeds=(1, 2))
    old = snapshot(out)
    res = run(frames, out, seeds=(1, 2), cfg=BiLSTMConfig(**{**CFG, "epochs": 2}), overwrite=True)  # a different config
    new = snapshot(out)

    archive_dirs = [p for p in (out / "archive").iterdir()]
    assert len(archive_dirs) == 1
    archived = {p.relative_to(archive_dirs[0]).as_posix(): p.read_bytes() for p in archive_dirs[0].rglob("*") if p.is_file()}
    replaced = {k for k in old if k != "experiment_log.csv"}
    assert set(archived) == replaced and all(archived[k] == old[k] for k in replaced)  # old results kept byte for byte
    assert len(res["archived"]) == len(replaced)
    assert new[f"bilstm_{RUN}_dev_scores.json"] != old[f"bilstm_{RUN}_dev_scores.json"]  # and the new ones are in place
    assert all(not k.startswith(("archive", ".staging", "failed_runs")) or k.startswith("archive") for k in new)

    assert new["experiment_log.csv"].startswith(old["experiment_log.csv"])  # old rows untouched, new rows appended
    rows = log_rows(out)
    assert [r["seed"] for r in rows] == ["1", "2", "1", "2"]
    assert all("rerun" not in r["notes"] for r in rows[:2]) and all("rerun=yes" in r["notes"] for r in rows[2:])


def test_overwrite_with_nothing_to_overwrite_is_a_normal_run(frames, tmp_path):
    out = tmp_path / "results"
    res = run(frames, out, seeds=(1,), overwrite=True)
    assert res["archived"] == [] and not (out / "archive").exists()
    assert "rerun" not in log_rows(out)[0]["notes"]


def test_overwrite_twice_keeps_both_generations(frames, tmp_path):
    out = tmp_path / "results"
    run(frames, out, seeds=(1,))
    run(frames, out, seeds=(1,), overwrite=True)
    run(frames, out, seeds=(1,), overwrite=True)
    assert len(list((out / "archive").iterdir())) == 2  # nothing was ever deleted


# --------------------------------------------------------- failure midway -----
@pytest.mark.parametrize("exc", [RuntimeError, KeyboardInterrupt])
def test_failure_midway_leaves_earlier_results_untouched_even_with_overwrite(frames, tmp_path, monkeypatch, exc):
    out = tmp_path / "results"
    run(frames, out, seeds=(1, 2))
    before = snapshot(out)
    fail_on_seed(monkeypatch, 2, exc)  # seed 1 trains fine, seed 2 blows up
    with pytest.raises(exc):
        run(frames, out, seeds=(1, 2), overwrite=True)
    after = snapshot(out)
    assert {k: v for k, v in after.items() if not k.startswith("failed_runs/")} == before  # nothing changed, nothing archived
    assert not any(p.name.startswith(".staging") for p in out.iterdir())
    kept = [k for k in after if k.startswith("failed_runs/")]  # the finished seed is not lost
    assert any(k.endswith(f"bilstm_{RUN}_seed1_dev.csv") for k in kept) and any(k.endswith(f"bilstm_{RUN}_seed1.pt") for k in kept)
    assert not any("seed2" in k for k in kept)  # the seed that failed produced nothing


def test_failure_midway_in_a_fresh_folder_publishes_nothing(frames, tmp_path, monkeypatch):
    out = tmp_path / "results"
    fail_on_seed(monkeypatch, 3, RuntimeError)
    with pytest.raises(RuntimeError):
        run(frames, out, seeds=(1, 2, 3))
    top = {p.name for p in out.iterdir()}
    assert top == {"failed_runs"}  # no predictions, checkpoints, scores or log in the real locations
    assert not (out / "experiment_log.csv").exists()
    kept = list((out / "failed_runs").rglob("*.csv")) + list((out / "failed_runs").rglob("*.pt"))
    assert len(kept) == 4  # predictions and checkpoints of the two finished seeds


def test_a_failed_run_can_be_retried_after_a_failure_with_the_same_name(frames, tmp_path, monkeypatch):
    out = tmp_path / "results"
    fail_on_seed(monkeypatch, 2, RuntimeError)
    with pytest.raises(RuntimeError):
        run(frames, out, seeds=(1, 2))
    monkeypatch.undo()  # the problem is fixed: the same command works, no --overwrite needed
    run(frames, out, seeds=(1, 2))
    assert [r["seed"] for r in log_rows(out)] == ["1", "2"]


def test_a_file_that_appears_while_the_run_trains_is_not_overwritten(frames, tmp_path, monkeypatch):
    out = tmp_path / "results"
    real = tb.train_one_seed

    def wrapper(tr, dv, cfg, seed, **kw):
        res = real(tr, dv, cfg, seed, **kw)
        if seed == 1:  # someone else writes the scores file during the run
            scores_path(out, RUN).write_text("someone else's results", encoding="utf-8")
        return res

    monkeypatch.setattr(tb, "train_one_seed", wrapper)
    with pytest.raises(FileExistsError, match="appeared while the run was training"):
        run(frames, out, seeds=(1, 2))
    assert scores_path(out, RUN).read_text(encoding="utf-8") == "someone else's results"
    assert not prediction_path(out, RUN, 1).exists() and not (out / "experiment_log.csv").exists()  # nothing half-published
    assert any((out / "failed_runs").rglob(f"bilstm_{RUN}_seed1_dev.csv"))  # and the finished work is kept


# ------------------------------------------------------------- the log --------
def test_log_append_never_changes_existing_bytes_and_survives_odd_endings(tmp_path):
    log = tmp_path / "experiment_log.csv"
    old = (",".join(LOG_HEADER) + "\r\n").encode() + b"2026-01-01,Mani,m,d,s,42,0.6,note,lesson\xc3\xa9"  # no final newline
    log.write_bytes(old)
    append_log_rows(log, [["2026-01-02", "Ekta", "m", "d", "s,with,commas", 1, "0.5", "run=x; a=b", ""]])
    new = log.read_bytes()
    assert new.startswith(old) and new[len(old):].startswith(b"\r\n")  # old bytes first, then a line break
    assert new.count(b"date,who,model") == 1  # header not duplicated
    assert [r["seed"] for r in csv.DictReader(open(log, encoding="utf-8", newline=""))] == ["42", "1"]
    assert not (tmp_path / "experiment_log.csv.tmp").exists()  # the temporary file is gone


def test_log_is_created_with_header_when_missing_and_uses_crlf_like_the_existing_log(tmp_path):
    log = tmp_path / "sub" / "experiment_log.csv"
    append_log_rows(log, [["d", "w", "m", "dv", "s", 1, "0.5", "n", ""]])
    data = log.read_bytes()
    assert data.startswith((",".join(LOG_HEADER) + "\r\n").encode()) and data.endswith(b"\r\n")


# ------------------------------------------------------------- the CLI --------
def test_cli_refuses_then_allows_a_new_run_name_then_overwrite(data_root, tmp_path):
    base = ["--mode", "normalized", "--seeds", "1", "--data-root", str(data_root), "--out-dir", str(tmp_path / "out"),
            *sum((["--set", f"{k}={json.dumps(v)}"] for k, v in CFG.items()), [])]
    main(base)
    out = tmp_path / "out"
    before = snapshot(out)
    with pytest.raises(FileExistsError):
        main(base)
    assert snapshot(out) == before
    main([*base, "--run-name", "second"])  # a new name works
    main([*base, "--overwrite"])  # explicit overwrite works and archives the previous files
    assert (out / "archive").is_dir() and any("rerun=yes" in r["notes"] for r in log_rows(out))
