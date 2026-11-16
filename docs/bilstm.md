# BiLSTM with word and character embeddings

Code: `src/bilstm_data.py` (vocabularies, encoding, batching), `src/bilstm_model.py` (model), `src/train_bilstm.py` (training, evaluation, logging, command line). Default settings: `configs/bilstm_default.json`. Tests: `tests/test_bilstm_data.py`, `tests/test_bilstm_model.py`.

**Status:** the code and tests are written and the tests pass on a small synthetic dataset. **No result on the real SentiMix data has been produced yet** (the dataset was not available when this was written), so there are no scores in this document and no claim about which text version is better.

## Architecture

For every tweet (a list of words), each word is turned into one vector built from two sources:

```
word            --> word embedding (100-d, learned)  ----------------------+
                                                                           +--> concat (100 + 96 = 196-d)
characters      --> char embedding (32-d) --> CNN, kernels 2,3,4,           |
 of the word        32 filters each, max over positions (96-d) -------------+

concat --> dropout --> word-level BiLSTM (128 per direction)
       --> masked pooling: concat(max over words, mean over words)  (512-d)
       --> dropout --> Linear(512 -> 3)  --> negative / neutral / positive
```

How the two embeddings work together:
- The **word embedding** knows whole words seen in training (`bahut`, `bakwas`). A word never seen in train, or seen only once, is `<unk>` and carries no information.
- The **character CNN** reads the spelling of every word, so it still gives a useful vector for an unseen word, a misspelling or a spelling variant. The filters of width 2 to 4 act like learned character n-grams, and max-pooling keeps the strongest match anywhere in the word.
- Concatenating the two gives the BiLSTM both "which word is this" and "what does it look like". The BiLSTM reads the tweet left to right and right to left, so each word's output also reflects its context (for example a negation nearby).
- Pooling over words, with padding masked out, turns the sequence into one tweet vector for the classifier.

Why a CNN for characters and not a character BiLSTM: every word of every tweet must be encoded (about 26 per tweet), so a CNN is much cheaper and simpler, and n-gram detectors suit short words and spelling variants. A character BiLSTM is a reasonable alternative to try later.

Variants from the same class: `use_char=false` gives word-only, `use_word=false` gives char-only, and `pooling` can be `maxmean` (default), `mean`, `max` or `last`.

Embeddings are randomly initialised and trained from scratch (no pretrained vectors are downloaded).

## Data pipeline

1. `load_processed_split(split, mode)` (from `src/preprocess.py`) gives the official train or dev split with a `text_proc` column. `mode` is `raw`, `clean` or `normalized`.
2. `TextEncoder.fit(train_texts)` builds everything that is learned from data, **from the train split only**:
   - word vocabulary: words seen at least `word_min_freq` (2) times, lowercased by default, plus two **reserved placeholder entries** (see "Placeholders" below);
   - character vocabulary: characters seen at least `char_min_freq` (2) times, case kept (capital letters can signal shouting);
   - `max_words` and `max_chars`: **one pair of limits shared by raw, clean and normalized** (see "Length limits" below).
3. `encode` maps any split's tweet to ids. Anything not in the train vocabularies becomes `<unk>` (id 1). Padding is `<pad>` (id 0). An empty tweet becomes one `<unk>` word.
4. `make_batch` pads each batch only to its own longest tweet and longest word, and the model ignores padding: packed sequences and masked pooling at the word level, and at the character level the CNN only lets convolution windows that lie fully inside a word take part in the max-pool (a word shorter than a kernel uses its single window of characters followed by zero vectors). So a word gets the same character vector whatever the batch's padding is, and a tweet gets the same logits alone or batched with others. Tests cover both (`test_char_cnn_same_word_same_vector_for_any_right_padding` and `test_whole_model_prediction_independent_of_batch_composition`).

Dev and test words are never added to a vocabulary. The test split is never loaded by `train_bilstm.py` (a test checks the exact splits requested).

### Placeholders (`USER`, `URL`)

`src/preprocess.py` writes `USER` for a mention and `URL` for a link (in `clean` and `normalized` mode). The word vocabulary reserves **ids 2 and 3 for exactly these two words** (`<pad>`=0, `<unk>`=1, `USER`=2, `URL`=3, then the ordinary words), and they are exempt from lowercasing, for words and for characters. So:
- the placeholder `USER` and the ordinary word `user` (or `User`) always have **different ids**, and likewise `URL` and `url`. Before this was fixed, lowercasing merged them: on the real train split the word `user` would have merged 12,519 placeholders with 3 genuine uses, `url` 7,823 placeholders with none;
- ordinary words are lowercased as before and keep their own embeddings;
- the ids 2 and 3 are the same in `raw`, `clean` and `normalized` (in `raw` there are no placeholders, so those two rows are simply unused), and do not depend on the data or on `word_min_freq`;
- the mapping is saved inside checkpoints (the encoder stores `special_words`) and restored on loading. Checkpoint format version 2; version-1 files are rejected.

Only the exact upper-case spelling counts as the placeholder. A person typing `USER` or `URL` in capitals in a tweet would still look like the placeholder after cleaning; there are no such tokens in the raw train split, and avoiding it entirely would need a different placeholder in `src/preprocess.py`, which was left unchanged.

### Length limits (same for all three modes)

A tweet is cut to its first `max_words` words and each word to its first `max_chars` characters. If every mode derived its own limits, the three experiments would truncate differently, which is an unwanted difference between them. On the real train split the limits per mode would be:

| Mode | Longest tweet (words) | 99.9th percentile of word length (chars) |
|---|---|---|
| raw | 56 | 16 |
| clean | 53 | 18 |
| normalized | 53 | 17 |

(Cleaning removes tokens, so tweets get shorter, but it rejoins split contractions such as `wouldn ’ t` into `wouldn't`, so some words get longer. That is why raw's limit cannot simply be used for every mode.)

Rule (`derive_shared_limits` in `src/train_bilstm.py`): preprocess the **train** tweets with each of the three modes, measure each mode's limits, and use the **maximum over the three modes** for each limit. On the real data that is **56 words and 18 characters** for every mode. Reasons:
- **Fair:** raw, clean and normalized apply exactly the same limits, and the limits do not change when you run one mode at a time (each command derives the same numbers from the same train split). The number of words actually cut still differs a little between modes because the texts differ (on dev with 56 / 18: 42, 42 and 40 words for raw, clean, normalized).
- **Train only:** the limits come from the train tweets alone. Dev and test never influence them (a test checks that a very long dev tweet changes nothing).
- **Little is lost:** with 56 / 18 on the real data, no train or dev tweet is cut by the word limit in any mode, and under 0.1% of train words (0.06% to 0.08%) are cut by the character limit.
- **Reproducible:** the derivation is deterministic and independent of row order.

Dev tweets longer than the limits are cut (the same in every mode). The number of cut dev tweets and words is stored per seed in the scores JSON (`truncation`) and in the log notes (`dev_truncated_tweets`). To force other values use `--set max_words=64 --set max_chars=20`.

The vocabularies still differ between modes, by design: they are what is being compared.

## How to run

Install the project requirements first (`pip install -r requirements.txt`; PyTorch is already listed) and download the data as described in `docs/data_format.md`.

```bash
# the three experiments: same config, same seeds, only the text version differs
python -m src.train_bilstm --mode raw        --seeds 42 43 44
python -m src.train_bilstm --mode clean      --seeds 42 43 44
python -m src.train_bilstm --mode normalized --seeds 42 43 44

# word-only baseline (same data, no character encoder)
python -m src.train_bilstm --mode normalized --seeds 42 43 44 --set use_char=false --run-name bilstm_normalized_wordonly

# change settings: a JSON file and/or --set key=value (valid keys: the fields of BiLSTMConfig)
python -m src.train_bilstm --config configs/bilstm_default.json --set hidden_size=256 --set char_kernels=[2,3]

# quick check that everything works (first 500 tweets, NOTHING is saved or logged, no checkpoint)
python -m src.train_bilstm --mode normalized --limit 500 --set epochs=2

# the dataset lives in data/raw, but src.data looks elsewhere by default, so pass the folder explicitly
python -m src.train_bilstm --data-root data/raw --mode normalized --seeds 42 43 44
```

Checkpoints are saved by default; `--no-checkpoints` turns this off.

### Existing results are protected

- **Check before training:** a run lists every file it will write (the scores JSON, the predictions and the checkpoint of each seed) and the log rows of its run name and seed. If any of them exists, it stops **before training** with a `FileExistsError` that names them all, and writes nothing. Use another `--run-name` to keep both, or `--overwrite` to replace.
- **One partly existing output blocks the whole run**, including a multi-seed run, so you never get a mixture of old and new seeds. Because the scores JSON belongs to the run, adding seeds later needs a new `--run-name`, not the old one.
- **All or nothing:** the run writes into a hidden `.staging_*` folder and publishes it only after **all seeds** finished. If a seed crashes, or you press Ctrl+C, nothing existing is touched, no half-written file appears, and the finished work is kept in `<out-dir>/failed_runs/<run>_<timestamp>/` (delete it when you no longer need it). Re-running the same command afterwards needs no `--overwrite`, because nothing was published.
- **Overwrite never deletes:** with `--overwrite`, the old files are moved to `<out-dir>/archive/<timestamp>/` (same sub-paths) and the new ones take their place. Files in `archive/` and `failed_runs/` are generated by the script; do not commit them (checkpoints are already git-ignored).
- **The log is append-only:** `experiment_log.csv` is never edited or truncated: new rows are added with an atomic swap, so existing rows and their hand-written lessons stay byte for byte. Rows written by this script start their notes with `run=<run name>;`, which is how a run finds its own earlier rows. After `--overwrite` the new rows say `rerun=yes`, and the older rows remain as history (the last one is current).

Use the same `--set` values for all three modes. For a fair comparison do not tune hyperparameters separately per mode.

Training uses Adam (lr 0.001), batch size 64, gradient clipping at 5, up to 20 epochs, and stops when the dev weighted F1 has not improved for 4 epochs. The epoch with the best dev weighted F1 is kept. Randomness is fixed per seed with `src.seed.set_seed` (also the order in which batches are shuffled).

## Outputs

| File | Content |
|---|---|
| `results/experiment_log.csv` | one appended row per seed: dev weighted F1 in `dev_score`; macro F1, accuracy, best epoch, length limits, dev truncation and parameter count in `notes` |
| `results/bilstm_<run>_dev_scores.json` | config, shared length limits, per-seed metrics (incl. per-class precision/recall/F1, confusion matrix, training history, truncation counts, checkpoint path), mean and standard deviation over seeds |
| `results/predictions/bilstm_<run>_seed<N>_dev.csv` | `uid,label,pred` for every dev tweet, same format as the TF-IDF predictions, so the CMI-bucket and error analysis scripts can use it |
| `results/checkpoints/bilstm_<run>_seed<N>.pt` | the best model (highest dev weighted F1) of that seed, with everything needed to reuse it (about 1.8 MB; git-ignored) |

`<run>` is `bilstm_<mode>` unless `--run-name` is given, so the checkpoint for mode `normalized`, seed 42 is `results/checkpoints/bilstm_bilstm_normalized_seed42.pt`. Metrics (accuracy, macro F1, weighted F1, per class precision/recall) all come from `src/metrics.py`. Weighted F1 is the official SemEval score; macro F1 is reported next to it.

### Checkpoints (`src/bilstm_checkpoint.py`)

A checkpoint holds the model weights, the word and character vocabularies, the length limits, every config setting (including the preprocessing `mode`), the seed, the best epoch, the dev metrics of that epoch, the preprocessing options, a fingerprint (SHA-256) of `configs/hinglish_variants.json`, train/dev sizes, truncation counts and library versions. It contains **no test tweets, test labels or test-derived numbers** (the training script never loads the test split; a test scans the file for any mention of the test split). It is loaded with `torch.load(weights_only=True)`, so it can contain only tensors and plain data and cannot run code.

```python
from src.bilstm_checkpoint import load_checkpoint

ck = load_checkpoint("results/checkpoints/bilstm_bilstm_normalized_seed42.pt")
ck.meta["dev_metrics"]["weighted_f1"]      # the score of the saved epoch on dev
ck.config.mode, ck.meta["seed"], ck.meta["best_epoch"]

ck.predict_texts(dev["text_proc"])                      # already preprocessed text -> labels
ck.predict_texts(["Yaaar bohot accha tha !!"], preprocess=True)   # original tweets: the checkpoint's own mode is applied first
```

Loading warns if the Hinglish dictionary changed since training, because new text would then be preprocessed differently from the training text. The one-time final test run can use these checkpoints instead of retraining (evaluate the chosen checkpoints on the test split once, after the models are frozen).

## Things to keep in mind

- **Dev scores are slightly optimistic**: the best epoch is chosen on dev, as the project plan says. The final unbiased numbers need the single test run, which this script does not do.
- **Mean and spread over seeds**: report the mean and standard deviation (printed and saved). With 3 seeds the standard deviation is itself rough. A difference between `raw` and `normalized` smaller than the seed spread (and the roughly 0.009 sampling error of F1 on 3,000 dev tweets noted in `docs/baselines_tfidf.md`) should not be called an improvement.
- **Nothing here shows that normalization helps.** It only provides the tool to measure it.
- **Reference point:** the TF-IDF baselines score about 0.61 weighted F1 on dev (`docs/baselines_tfidf.md`).

## Limitations and pending work

- The full experiments have not been run on the real data yet. Only small smoke runs (500 tweets, 2 epochs) were done, and their scores mean nothing. The tests use a small synthetic dataset in the real CoNLL format.
- Reproducibility is verified on CPU (identical results across separate processes). On a GPU, some CUDA operations (for example embedding backward) can vary slightly between runs, so rely on the mean and spread over seeds.
- Output protection covers what this script writes. It cannot protect a file that someone edits or deletes by hand, and it only recognises log rows that carry the `run=<name>;` prefix, so rows written by an older version of the script are not detected.
- Embeddings are trained from scratch; pretrained Hinglish or multilingual vectors could help but are not used.
- No hyperparameter search beyond the defaults; no class weighting.
- The test-set run is not implemented; add it only after the final models are frozen.
- Only CPU was used for testing; with a GPU the same command runs on `cuda` automatically (`device=auto`).
