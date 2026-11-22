# Transformer final test evaluation protocol

**Status: frozen before the official test labels were read for any transformer purpose.** Five files define one single evaluation of six already-trained models:
- this document;
- `configs/transformer_frozen_checkpoints.json`;
- `src/evaluate_transformer_test.py` and its helpers in `src/eval_protocol_utils.py`;
- `tests/test_evaluate_transformer_test.py`.

Nothing in them may be changed after the test results are seen. The protocol follows the BiLSTM protocol (`docs/bilstm_final_evaluation_protocol.md`) and uses the same code for hashing, parsing, joining by uid and metrics. That code now lives in `src/eval_protocol_utils.py`, copied unchanged from the BiLSTM evaluation script, which was later removed from the repository.

## 1. Purpose

Measure, once, how the fine-tuned mBERT and XLM-R models generalise to the official SemEval-2020 Task 9 SentiMix Hinglish **test** set. The models are the ones trained and compared on the development set in issue #6 (`docs/transformer_results.md`).

The test results are measurements only. They are not used to select a model, a seed or a learning rate, to tune anything, or to change preprocessing.

## 2. The frozen models

There are six checkpoints: two models x seeds 42, 43 and 44.
- **Same settings.** Every setting is identical across the six runs except the pretrained model and its learning rate. Each run uses the shared `configs/transformer_config.json` with `epochs=6`.
- **All six are evaluated.** No checkpoint selection is made: there is no "best seed" and no "best model" chosen from test results. Every checkpoint is evaluated and reported.

| Model | Pretrained checkpoint | lr (chosen on dev) | `max_length` | Seeds | Checkpoint folders (git-ignored) |
|---|---|---|---|---|---|
| mBERT | `bert-base-multilingual-cased` | 2e-5 | 80 | 42, 43, 44 | `results/checkpoints/transformer_mbert_clean_ep6_lr2e-5_seed42`, `..._lr2e-5_final_seed43`, `..._lr2e-5_final_seed44` |
| XLM-R | `xlm-roberta-base` | 1e-5 | 96 | 42, 43, 44 | `results/checkpoints/transformer_xlmr_clean_ep6_lr1e-5_seed42`, `..._lr1e-5_final_seed43`, `..._lr1e-5_final_seed44` |

**The XLM-R learning rate is effectively a tie.**
- **The scores.** On the single search seed (42), lr 1e-5 scored 0.6301 dev weighted F1 and 2e-5 scored 0.6300. The selection rule picked 1e-5 by a margin of 0.0001, which is noise (`results/transformer_lr_search_ep6.csv`).
- **What the freeze means.** It makes this choice final, so the test results describe XLM-R at lr 1e-5 only. They say nothing about whether 2e-5 would be better or worse.
- **mBERT is less close.** Its choice (2e-5, 0.6129) led 1e-5 (0.6088) by 0.004 on one seed.

**What the manifest records.** `configs/transformer_frozen_checkpoints.json` lists, for each run:
- the MD5 and size of **every file** in the checkpoint folder: `config.json`, `model.safetensors`, `run_info.json`, `tokenizer.json` and `tokenizer_config.json`;
- the MD5 and size of the run's frozen **dev** prediction file.

The evaluation refuses to run in any of these cases:
- any file differs from the manifest;
- a folder contains any other file;
- a checkpoint's `run_info.json` does not describe exactly the run the manifest names.

The `run_info.json` check covers the model, the pretrained checkpoint, the seed, the learning rate, the text mode, the epochs, `max_length`, the preprocessing options, the train and dev sizes, and the selection criterion. Within that check, the six runs must share every training setting except the model and its learning rate.

**How the models were trained** (`docs/transformer_pipeline.md`, `docs/transformer_results.md`):
- **Data.** Text mode `clean`. The train split was used for training only.
- **Optimiser and schedule.** AdamW with weight decay 0.01. Linear warm-up over 10%, then linear decay. 6 epochs, early stopping with patience 2.
- **Batches and precision.** Batch size 32, fp16 on CUDA.
- **Selection on dev.** The best epoch of each run was chosen by **dev weighted F1**. The learning rate of each model was chosen on dev with seed 42, from 1e-5, 2e-5 and 3e-5.
- **Code version.** The runs were trained with the code at commit `76a161e`. The training code has not changed since.

## 3. Datasets and how each split is used

| Split | Size | Use |
|---|---|---|
| Train | 14,000 | Fine-tuning; deriving `max_length` |
| Dev | 3,000 | Best epoch, learning rate, model comparison (issue #6). Also the **dev reproduction check** below |
| **Test** | 3,000 | **Final evaluation only, once, by `src/evaluate_transformer_test.py`** |

The official files are used unchanged. The test files, their sizes and their MD5 hashes from the official archive are the same as in the BiLSTM protocol:
- `Hinglish_test_unalbelled_conll_updated.txt`, MD5 `4706975d6fa025634c176111d3271487`;
- `Hinglish_test_labels.txt`, MD5 `60199636a5da7e8195f0db359332163e`.

## 4. Inference (fixed in advance)

Test inference uses the same path as the dev evaluation during training:
- the text is preprocessed with mode `clean` (`src.preprocess.preprocess_frame`);
- the text is tokenised with each checkpoint's own saved tokenizer and truncated to the model's `max_length`. No test tweet is expected to exceed it, because dev and train maxima are 69 and 79 subwords for mBERT, and 90 and 90 for XLM-R;
- batches are padded dynamically to the longest tweet;
- the batch size is 128;
- inference runs on **CUDA with fp16 autocast**, in eval mode with no gradients;
- the prediction is the arg-max class.

**Dev reproduction check.** Before any test file is opened, every checkpoint predicts the dev split again. Its predictions are compared with the run's frozen dev predictions. The check requires both:
- an agreement of at least **0.99**;
- a weighted F1 within **0.005** of the value recorded at training time.

These tolerances allow for the small non-determinism of GPU fp16 kernels. This proves that the frozen files and this inference path really are the models that were selected on dev.

In the pre-flight run on 22 Nov 2026, all six checkpoints reproduced their dev predictions exactly: agreement 1.0000, and identical weighted F1.

## 5. Metrics (fixed in advance)

These are the same definitions as in the BiLSTM protocol:
- **Class order.** Everywhere, `negative, neutral, positive`.
- **Zero division.** Precision, recall and F1 use `zero_division = 0`.
- **Primary metrics.** **Macro F1** and **Weighted F1**.
- **Secondary metrics.** Accuracy, macro precision and recall, weighted precision and recall.
- **Class-wise metrics.** Precision, recall and F1 for each class.
- **Confusion matrices.** Rows are true classes and columns are predicted classes, given per run and summed over the three seeds of each model.
- **Cross-check.** The numbers are checked against `src.metrics.compute_metrics`.

## 6. Aggregation and reporting

- **Primary result per model.** Mean ± **sample standard deviation** (ddof = 1) over seeds 42, 43 and 44. All six per-run values are reported too.
- **Pre-specified descriptive comparison.** mBERT minus XLM-R for each seed, on macro F1 and weighted F1. No winner is declared by the evaluation. A difference smaller than the seed-to-seed spread, or smaller than the sampling error of about 0.01 on 3,000 tweets, is not interpreted as a real difference.
- **Further analysis is descriptive only.** Any further test-set analysis, such as bootstrap intervals, CMI buckets or error analysis, is descriptive and must be labelled as such. It never feeds back into any model.
- **Shared format.** The labelled predictions are also saved as `uid,label,pred`, the project's common format, for that analysis.

## 7. Procedure

```
python -m src.evaluate_transformer_test --validate-only          # pre-flight + dev reproduction; no test file opened, writes nothing
python -m src.evaluate_transformer_test --confirm-final-test     # the one real evaluation
```

1. **Pre-flight (both modes).**
   - Validate the manifest: exactly the six runs.
   - MD5-check every checkpoint file and every frozen dev prediction file.
   - Check each `run_info.json`.
   - Run the dev reproduction check.
   - In `--validate-only` the two test files are only checked for existence and size. They are not opened or hashed, and the program stops here.
2. **Stage A: predictions (no labels involved).**
   - Verify the MD5 of the unlabeled test file.
   - Parse it: 3,000 tweets, unique uids, no label column.
   - Each of the six checkpoints predicts every tweet. Each prediction set is validated, written atomically to `predictions/transformer_final_test_<model>_seed<seed>_predictions.csv` (`uid,pred`), read back and validated again.
   - Write `predictions_manifest.json` with their hashes.
3. **Stage B: labels (the labels file is first touched here).**
   - Verify the MD5 of the labels file, then read it strictly.
   - Require that the label uids and the test uids are the same set.
   - For each run, read the **saved** prediction file and attach the gold labels **by uid**. Compute all metrics.
   - Write the labelled `uid,label,pred` copies, the report, the JSON files and the confusion-matrix figure.
4. **Integrity.**
   - All hashed inputs are hashed again and must be unchanged.
   - `integrity_manifest.json` records:
     - the hashes;
     - the time of every stage;
     - the environment: Python, PyTorch, transformers, the GPU;
     - the dev reproduction results;
     - the hash of every output file.
5. **Outputs.**
   - **Location.** Everything is staged in `<out-dir>.staging` and published by one atomic rename to `results/transformer_final_test`.
   - **No overwriting.** There is no overwrite option.
   - **Failure before the labels are read.** The staging folder is discarded, so a corrected rerun is possible.
   - **Failure after the labels are read.** The work is kept in `<out-dir>.failed_<time>`, which blocks a silent rerun.

## 8. Rules

1. No model, seed, learning rate, epoch, preprocessing or inference setting is chosen or changed because of test results.
2. The evaluation is run once. A rerun is allowed only to repeat the same measurement after an operational failure or after a defect in the evaluation code is found. In that case, the defect and both runs are reported openly, and no result is picked.
3. The frozen files are not edited after the run. The manifest hashes are the proof of which models were measured.
4. Results are reported for all six runs and as mean ± sample sd per model, never for a single chosen seed.
5. **The run is started only after the team agrees** that these six checkpoints are final and that this protocol applies (the checklist below).

## 9. Disclosure: earlier access to the test split

The test split has **not** been used for any transformer training, early stopping, learning-rate choice, epoch choice, `max_length` derivation, preprocessing design or model selection:
- the training scripts load only train and dev;
- `max_length` was derived from train.

It has been read for descriptive purposes before. The BiLSTM protocol, section 8, lists these reads: team EDA, label counts, CMI per tweet, data checks and unit tests. In addition:
- The BiLSTM test evaluation (`results/bilstm_final_test/`) was run and its results were published before this protocol was frozen. The BiLSTM test scores were known to the team. They did not influence any transformer setting, which had all been fixed on dev by then (issue #6).
- `results/transformer_token_lengths.json` contains subword lengths of **train and dev** only.

**Revision note (22 Nov 2026, PR #21 review, before any test run).** Three changes, none of which touches a checkpoint, a hash, an inference setting or a metric:
- the helper code moved to `src/eval_protocol_utils.py`, because the BiLSTM evaluation script it came from was removed from `main`;
- the XLM-R learning-rate tie is now stated in section 2;
- a reference to a file not yet on `main` was removed from `src/final_comparison.py`.

The dev reproduction check was run again after the change and still reproduced all six runs exactly.

## 10. Limitations

- **Rough spread.** Three seeds per model, so the standard deviations are rough.
- **Sampling error.** One fixed test sample of 3,000 tweets, so F1 carries a sampling error of roughly ±0.01 that this evaluation does not estimate.
- **Optimistic dev scores.** The best epoch and the learning rate were chosen on dev, and seed 42 also took part in the learning-rate choice. So the dev scores are optimistic, and the test scores are the unbiased estimate for these frozen models.
- **Hash checks.** They protect against accidental change of the frozen files, not against a wrong decision about which files to freeze.
- **Hardware.** The evaluation is fixed to CUDA with fp16, the same as training (RTX 4050 Laptop GPU). On other hardware the last digits could differ; the dev reproduction check would show it.

## 11. Files

| File | Role |
|---|---|
| `docs/transformer_final_evaluation_protocol.md` | this protocol |
| `configs/transformer_frozen_checkpoints.json` | frozen checkpoint and dev-prediction hashes, models, inference settings, tolerances, official test-file hashes |
| `src/evaluate_transformer_test.py` | the evaluation program |
| `src/eval_protocol_utils.py` | shared helpers: hashing, atomic writes, strict parsing, join by uid, metrics |
| `tests/test_evaluate_transformer_test.py` | tests with synthetic data only (never the official test files) |

## 12. Checklist before the run (to be ticked by a person)

- [ ] `git status` clean; protocol, manifest, script and tests committed (and pushed, so the freeze is visible before the run)
- [ ] `python -m pytest tests/test_evaluate_transformer_test.py` passes
- [ ] `python -m src.evaluate_transformer_test --validate-only` prints `VALIDATION OK`
- [ ] Team agrees that the six checkpoints are final and the rules in section 8 apply
- [ ] Other GPU-heavy programs are closed
- [ ] Then, once: `python -m src.evaluate_transformer_test --confirm-final-test`
