# BiLSTM final test evaluation protocol

> **Note:** the scripts and tests named in this file (`src/evaluate_bilstm_test.py`, `src/analyze_bilstm_test.py` and their tests) were removed from the repository to keep it small (commit `8e6339e`). They are still in the git history before that commit. The saved results below were produced by them and are unchanged.

**Status: frozen before the official test labels were read for any model purpose.** This document, `configs/bilstm_frozen_checkpoints.json`, `src/evaluate_bilstm_test.py` and `tests/test_evaluate_bilstm_test.py` define one single evaluation of nine already-trained models. Nothing in them may be changed after the test results are seen.

## 1. Purpose

Measure, once, how the three BiLSTM preprocessing variants (raw, clean, normalized) generalise to the official SemEval-2020 Task 9 SentiMix Hinglish **test** set, using the nine models that were trained and compared on the development set. The test results are measurements only: they are not used to select a model, tune anything, or change preprocessing.

## 2. The frozen models

Nine checkpoints: three preprocessing modes x seeds 42, 43, 44. All nine are final. **No checkpoint selection is made** (no "best seed", no "best mode" chosen from test results); every checkpoint is evaluated and reported.

| Mode | Seed | Checkpoint (git-ignored) | Size (bytes) | MD5 |
|---|---|---|---|---|
| raw | 42 | `results/checkpoints/bilstm_full_raw_s42_seed42.pt` | 8,025,085 | `45f166e48d73532f89c3e044c54c80fc` |
| raw | 43 | `results/checkpoints/bilstm_full_raw_s43_seed43.pt` | 8,025,085 | `540987f2af85d646c368572f0b07fd61` |
| raw | 44 | `results/checkpoints/bilstm_full_raw_s44_seed44.pt` | 8,025,085 | `a8f54c8cd8836155c8d7ea35b6806895` |
| clean | 42 | `results/checkpoints/bilstm_full_clean_s42_seed42.pt` | 6,865,645 | `190bc0c38b5da17fd4515439f30f0d32` |
| clean | 43 | `results/checkpoints/bilstm_full_clean_s43_seed43.pt` | 6,865,645 | `a481a1a2eb7159ad8ee7bfbe996dd6ad` |
| clean | 44 | `results/checkpoints/bilstm_full_clean_s44_seed44.pt` | 6,865,645 | `65d71b5961eeaff0e07141b1e2023a67` |
| normalized | 42 | `results/checkpoints/bilstm_full_normalized_s42_seed42.pt` | 6,836,645 | `afc64d40cec38a52d40672a23c3894fa` |
| normalized | 43 | `results/checkpoints/bilstm_full_normalized_s43_seed43.pt` | 6,836,645 | `d1b162e07e1d08fbddbd6ad619b5affb` |
| normalized | 44 | `results/checkpoints/bilstm_full_normalized_s44_seed44.pt` | 6,836,645 | `d56fd14f34725e74d9dc32e403a4f912` |

The machine-readable record is `configs/bilstm_frozen_checkpoints.json` (it also holds the expected limits, sizes and dictionary fingerprint). The evaluation refuses to run if any checkpoint differs from it.

**What a checkpoint contains and how it was trained** (see `docs/bilstm.md`): a word + character BiLSTM; every setting identical across the nine runs except the preprocessing mode (`configs/bilstm_default.json`: word embedding 100, character CNN with kernels 2/3/4 and 32 filters each, BiLSTM 128 per direction, dropout 0.4, Adam lr 0.001, batch 64, at most 20 epochs, early stopping with patience 4); vocabularies fitted on the **train** split only; shared length limits 56 words and 18 characters for all modes (derived from train); `USER`/`URL` placeholders have reserved ids. The best epoch of each run was selected by **development-set weighted F1**. The runs were trained at repository commit `a7d6f9e` (code unchanged since; frozen at `48b325e`).

## 3. Datasets and how each split is used

| Split | Size | Use in this project |
|---|---|---|
| Train | 14,000 | Training the BiLSTM; fitting vocabularies and length limits |
| Dev | 3,000 | Early stopping / choice of the best epoch; model comparison and analysis (reports in `reports/`) |
| **Test** | 3,000 | **Final evaluation only, once, by `src/evaluate_bilstm_test.py`** |

The official splits are used unchanged (Zenodo record 3974927, `Semeval_2020_task9_data.zip`, MD5 `4d93553c213b3a966b6d55b7a7c03be7`). The test set consists of two separate official files:

| File | Size | MD5 (from the official archive) |
|---|---|---|
| `Hinglish_test_unalbelled_conll_updated.txt` (tweets and language tags, no labels) | 810,391 | `4706975d6fa025634c176111d3271487` |
| `Hinglish_test_labels.txt` (`Uid,Sentiment`) | 46,222 | `60199636a5da7e8195f0db359332163e` |

## 4. Metrics (fixed in advance)

Class order everywhere: `negative, neutral, positive`. Precision, recall and F1 use `zero_division = 0`. All values are computed from the saved prediction files.

- **Primary:** **Macro F1** and **Weighted F1**.
- **Secondary:** Accuracy; Macro precision and recall; Weighted precision and recall (weighted recall equals accuracy by definition).
- **Class-wise:** precision, recall and F1 for each of the three classes.
- **Confusion matrices:** rows = true class, columns = predicted class, per run and summed over the three seeds of a mode.

The script cross-checks its numbers against the project's `src/metrics.py` (`compute_metrics`, `confusion`) and refuses to continue if they disagree.

## 5. Aggregation and reporting

- **Primary result for each preprocessing mode:** mean ± **sample standard deviation** (ddof = 1) of each metric over seeds 42, 43, 44. All nine per-run values are reported as well.
- Class-wise metrics are aggregated the same way. Confusion matrices are summed over the three seeds for each mode (per-run matrices are saved too).
- Metrics computed on *pooled* predictions are not headline numbers (the same tweets would appear three times).
- **No winner is declared by the evaluation.** A difference between modes smaller than the seed-to-seed spread is not interpreted as a real difference. Any further test-set analysis (for example bootstrap intervals or error analysis) is descriptive, must be labelled as such, and must not feed back into any model.

## 6. Procedure (two stages: predictions first, labels afterwards)

Command-line entry points (run from the repository root in the PyTorch environment):

```
python -m src.evaluate_bilstm_test --validate-only          # pre-flight; no labels, no inference, writes nothing
python -m src.evaluate_bilstm_test --confirm-final-test     # the one real evaluation
```

**Pre-flight (both modes).** The manifest must list exactly the nine runs. Every checkpoint is size- and MD5-verified, loaded, and checked to be the run the manifest claims (mode, seed, preprocessing options, length limits, placeholder ids, training/dev sizes, selection criterion, Hinglish-dictionary fingerprint, identical hyper-parameters across the nine). The output folder must not exist (see below). In `--validate-only` the two test files are only checked for existence and size: they are neither opened nor hashed.

**Stage A: predictions (no labels involved).**
1. Verify the MD5 of the unlabeled test file (the labels file is only size-checked).
2. Parse the unlabeled test file; require 3,000 tweets with unique uids and no label column.
3. Each of the nine checkpoints predicts all 3,000 tweets from the original tweet text, applying its own preprocessing mode; no retraining, evaluation mode, no gradients.
4. Each prediction set is validated (one valid label per uid, complete coverage), written atomically to `predictions/bilstm_final_test_<mode>_seed<seed>_predictions.csv` (`uid,pred`), read back from disk and validated again. A `predictions_manifest.json` with their hashes is written.

**Stage B: labels (the labels file is first touched here).**
5. Verify the MD5 of the labels file, then read it with strict validation (header `Uid,Sentiment`, unique uids, valid labels).
6. Require that the label uids and the test uids are exactly the same set.
7. For each run, read the **saved** prediction file and attach gold labels **by uid** (a dictionary lookup, never by row position); compute all metrics.
8. Aggregate over seeds, write the report, JSON files and confusion-matrix figures.

**Integrity at the end.** All hashed inputs (nine checkpoints, both test files) are hashed again and must equal the values from before; an `integrity_manifest.json` records the hashes, the time of every stage, the environment (Python, PyTorch, library versions) and the hash of every output file.

**Outputs and overwrite protection.** Everything is written into `<out-dir>.staging` and published with one atomic folder rename to `<out-dir>` (default `results/bilstm_final_test`). There is **no overwrite option**: if the output folder, a staging folder or a `.failed_*` folder already exists the run refuses to start. If anything fails *before* the labels file is touched, the staging folder is discarded and the corrected run can be repeated. If it fails *after* the labels file was touched, the saved predictions are kept in `<out-dir>.failed_<time>`, which blocks a silent rerun until a person decides what to do. The explicit flag `--confirm-final-test` is required; without exactly one of the two flags the program does nothing.

## 7. Rules

1. No model selection, hyper-parameter tuning, preprocessing change, dictionary change or checkpoint choice is based on test results.
2. The evaluation is run once. A rerun is allowed only to repeat the same measurement after an operational failure or after a defect in the evaluation code is found; in that case the defect and both runs are reported openly, and no result is picked.
3. The frozen files are not edited after the run. The manifest hashes are the proof of which models were measured.
4. Test results are reported for all nine runs and for each mode as mean ± sample standard deviation, never for a single chosen seed.

## 8. Transparent disclosure: earlier descriptive access to the test split

The official test split has **not** been used for training, early stopping, vocabulary building, hyper-parameter choice, preprocessing design or model selection. It has, however, been read for *descriptive* purposes before this evaluation, and this must be stated in any report:

- **Team EDA (`src/eda_part1.py`, `src/eda_part2.py`)** loaded the test split together with its labels. It computed the test class distribution (900 negative / 1,100 neutral / 1,000 positive), tweet-length and language-tag statistics, text overlap between the splits (including one text shared by train and test), and the code-mixing index (CMI) of every test tweet. The per-tweet CMI file `results/cmi_per_tweet.csv` therefore contains test rows with their gold labels. These numbers appear in `docs/eda_part1.md`, `docs/eda_part2.md` and the `results/eda_part*_stats.json` files.
- **Loader and data checks** (including checks run during the setup of this repository) loaded the test split once to confirm that the labels join by uid, counted labels and uid overlaps, and verified file sizes. No model was run on test data.
- **Unit tests** (`tests/test_data_seed.py`, `tests/test_preprocess.py`) load the test split with labels to check counts and uids.
- The BiLSTM training script never loads the test split (`src/train_bilstm.py` requests only train and dev; a test enforces this), and no hyper-parameter search was ever run: the configuration file was committed before training and never changed.
- Other team members' models (transformers, LLM) are outside this protocol; their own test access is theirs to document.

Because the class distribution and some corpus statistics of the test set were known beforehand, these descriptive facts could in principle have shaped expectations; they did not influence any model or setting.

**Revision note (2026-11-18, pull-request review).** The wording of one sentence in this section was clarified. No rule, checkpoint, hash, metric or result was changed.

## 9. Limitations

- Three seeds per mode: standard deviations are rough and differences between modes need not be real.
- One fixed test sample of 3,000 tweets, so scores carry a sampling error of roughly ±0.01 on F1 that this evaluation does not estimate.
- The development scores were optimistically selected (best epoch by dev weighted F1); the test scores are the unbiased estimate for these frozen models.
- Hash checks protect against accidental change and tampering with the frozen files, not against a wrong decision about which files to freeze.
- Reproducibility is verified on CPU; a GPU run could differ in the last digits.

## 10. Files

| File | Role |
|---|---|
| `docs/bilstm_final_evaluation_protocol.md` | this protocol |
| `configs/bilstm_frozen_checkpoints.json` | frozen checkpoint hashes, expectations, official test-file hashes |
| `src/evaluate_bilstm_test.py` | the evaluation program |
| `tests/test_evaluate_bilstm_test.py` | tests with synthetic data only (never the official test files) |

## 11. Checklist before the run (to be ticked by a person)

- [ ] PyTorch environment available (`python -m venv .venv`, `pip install -r requirements.txt`; ideally PyTorch 2.15.0, the version recorded in the checkpoints)
- [ ] `git status` clean; protocol, manifest, script and tests committed
- [ ] `python -m pytest tests/test_evaluate_bilstm_test.py` passes
- [ ] `python -m src.evaluate_bilstm_test --validate-only` prints `VALIDATION OK`
- [ ] Team agrees that the nine checkpoints are final and the rules in section 7 apply
- [ ] Then, once: `python -m src.evaluate_bilstm_test --confirm-final-test`
