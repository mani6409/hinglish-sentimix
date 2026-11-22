# Assessment preparation: Hinglish sentiment analysis

This file explains the code in this repository so you can answer questions about it in your own words.
Everything here is taken from the code, the configs, the `docs/` folder and the `results/` folder. Where the
repository does not say *why* something was done, this file says so and lists it under
[Open questions for you](#12-open-questions-for-you) instead of guessing.

How to use it: read sections 1 to 3 first (the big picture). Then, for each module in section 4, try to answer
the **Question** and the **Follow-up questions** out loud *before* reading the explanation.

---

## 1. The project in one minute

- **Task:** SemEval-2020 Task 9 (SentiMix), Hindi-English part. Each tweet is labelled **negative**,
  **neutral** or **positive**. This is three-class sentiment classification of **code-mixed** text.
- **Code-mixed / Hinglish:** Hindi and English mixed in one tweet, with Hindi usually typed in Roman letters
  and spelled inconsistently (`acha`, `accha`, `achha`).
- **Data:** official CoNLL files in `data/raw/.../Hinglish/`. 14,000 train, 3,000 dev, 3,000 test tweets. Every token
  has a language tag (`Hin`, `Eng`, `O` = other, `EMT` = emoticon). Train label shares: neutral 37.6%,
  positive 33.1%, negative 29.3%. That is fairly balanced, with neutral the largest class.
- **Official metric:** **weighted F1** (per-class F1 averaged by class size). Macro F1 and accuracy are also
  reported.
- **Models compared:** majority class → TF-IDF + logistic regression → BiLSTM (word + character CNN) →
  fine-tuned transformers (mBERT, XLM-R, MuRIL, IndicBERT v2) → zero-shot LLM (Qwen2.5-0.5B-Instruct).
- **Language and libraries:** Python; pandas and NumPy for data, scikit-learn for TF-IDF, logistic regression
  and metrics; PyTorch for the BiLSTM; Hugging Face `transformers` for the transformers and the LLM; matplotlib
  for figures; pytest for tests.
- **What this project is NOT:** there is **no web API, no frontend and no deployed app**. It is a research and
  experiment pipeline driven from the command line (`python -m src.<script>`). There is also **no stopword
  removal, no stemming or lemmatisation, and no class weighting** in the loss. If asked, say so plainly.

### Main results (from `results/final_model_comparison.csv`)

| System | Dev weighted F1 | Test weighted F1 (mean ± sd over 3 seeds) |
|---|---|---|
| Majority class (always "neutral") | 0.2055 | – |
| TF-IDF char 2-5 + logistic regression | 0.6127 | not run on test |
| BiLSTM word+char, clean text | 0.6118 ± 0.0015 | 0.6681 ± 0.0023 |
| mBERT, clean, lr 2e-5 | 0.6099 ± 0.0046 | 0.6866 ± 0.0081 |
| **XLM-R base, clean, lr 1e-5** | **0.6269 ± 0.0029** | **0.7006 ± 0.0041** |
| MuRIL, raw, lr 2e-5 | 0.6271 ± 0.0049 | not run on test |
| IndicBERT v2, raw, lr 2e-5 (1 seed) | 0.6305 | not run on test |
| Zero-shot Qwen2.5-0.5B (200-tweet dev subset only) | 0.4768 | not run on test |

Things to be able to say about this table:
- XLM-R is the best model on test. On dev, XLM-R beats mBERT on every seed (by 0.014 to 0.020), but the paired
  bootstrap interval for one pair of runs (−0.030 to +0.003) includes 0. So the dev difference is suggestive,
  not certain.
- **Test scores are higher than dev scores for every model.** For the BiLSTM this was investigated
  (`results/bilstm_test_analysis/bilstm_final_analysis.md`, section 4). The difference in class mix explains
  almost none of it, and the gap is officially **partly unexplained**.
- The zero-shot row uses only 200 dev tweets with a very small model, so it cannot be compared directly with the
  other rows.

---

## 2. Overall architecture

```
data/raw/*.txt (CoNLL)                configs/*.json (settings, Hinglish spelling dictionary)
        │                                        │
        ▼                                        ▼
  src/data.py  ──load_split──►  DataFrame: uid, label, text, tokens, langs
        │
        ▼
  src/preprocess.py  ── mode = raw | clean | normalized ──►  adds column text_proc
        │
        ├──► src/baselines_tfidf.py      TF-IDF features → logistic regression   (reads raw `text`)
        ├──► src/bilstm.py + src/train_bilstm.py           word ids + char ids → BiLSTM
        ├──► src/transformer_utils.py + src/train_transformer.py   subwords → mBERT / XLM-R
        ├──► src/indian_transformer.py (train_muril / train_indicbert)   subwords → MuRIL / IndicBERT
        └──► src/zero_shot_llm.py        prompt → LLM → parse answer       (reads raw `text`)
                         │
                         ▼
     every model writes the same file format:  uid,label,pred   (results/predictions/*_dev.csv)
                         │
                         ▼
  src/metrics.py  compute_metrics (weighted F1 = official), confusion
                         │
        ┌────────────────┼──────────────────────────┬──────────────────────────┐
        ▼                ▼                          ▼                          ▼
 src/cmi.py +      src/error_analysis*.py    src/transformer_comparison.py   src/evaluate_transformer_test.py
 src/cmi_evaluation.py   (why models fail)  src/run_transformer_experiments.py   + src/eval_protocol_utils.py
 (does mixing matter?)                       (lr search, seeds)             (single frozen TEST run)
                                                                                   │
                                                                                   ▼
                                                                  src/final_comparison.py → one results table
```

Shared helpers used everywhere:
- `src/seed.py`: `set_seed`, which fixes all randomness so runs can be repeated.
- `src/uid_checks.py`: `check_uid_alignment`, which makes sure a merge never drops, repeats or invents rows.

**Key design idea:** every model writes predictions in **one shared format (`uid,label,pred`)**, and every score
comes from **one metric function**. Because of this, the analysis scripts (CMI, error analysis, comparisons)
work for any model without changes, and all models are scored the same way.

---

## 3. Complete input-to-output flow (one tweet)

Take a train tweet as it appears in the CoNLL file:

```
meta	12345	positive
@	O
rahul	Hin
bahut	Hin
accha	Hin
movie	Eng
yaar	Hin
https	O
// 	O
...
```

1. **Parse** (`src/data.py: parse_conll`). The `meta` line starts a new tweet with `uid=12345` and
   `label="positive"`. Each following line adds a token and its language tag. A blank line separates tweets.
   `to_frame` joins the tokens with spaces into `text`: `"@ rahul bahut accha movie yaar https // t . co / x"`.
2. **Preprocess** (`src/preprocess.py: preprocess_text`). The mode controls what happens:
   - `raw`: the text is left as it is.
   - `clean`: `"USER bahut accha movie yaar URL"`. The mention became `USER`, the link became `URL`, a
     leading `RT` and Twitter's `…` are removed, `# tag` becomes `tag`, and `didn ’ t` becomes `didn't`.
   - `normalized`: clean, plus `goooood` becomes `good` (runs of 3+ letters cut to 2), plus dictionary
     spellings, so `accha` becomes `achha`.
3. **Model-specific encoding and prediction:**
   - **TF-IDF:** the raw `text` becomes a sparse vector of word or character n-gram weights. Logistic
     regression gives a score per class, and the class with the highest score wins.
   - **BiLSTM:** `TextEncoder.encode` splits on spaces and lowercases words. It maps each word to an id
     (unknown → `<unk>`) and each character to an id. `make_batch` pads the batch. The model combines a word
     embedding with a char-CNN vector for each word, runs a BiLSTM, then max+mean pooling and a linear layer,
     giving 3 logits. `argmax` picks the label.
   - **Transformers:** the model's own tokenizer splits `text_proc` into subwords (truncated at `max_length`).
     The encoder plus classification head gives 3 logits, and `argmax` picks the label.
   - **Zero-shot LLM:** the tweet is put into a prompt. The model answers in free text, and `parse_response`
     accepts the answer only if exactly one of positive/negative/neutral appears.
4. **Write** `uid,label,pred` to `results/predictions/<run>_dev.csv`.
5. **Score** with `compute_metrics(y_true, y_pred)`. This gives weighted F1, macro F1, accuracy and
   per-class precision, recall, F1 and support.

**Training flow** (BiLSTM and transformers): fit on **train**, score on **dev** after every epoch, keep the
**epoch with the best dev weighted F1** (early stopping), and save that checkpoint. The **test** split is
never loaded by any training script. It was used **once**, at the end, by a separate frozen evaluation script.

---

## 4. Module-by-module questions

### 4.1 `src/data.py`: `parse_conll`, `to_frame`, `load_split`

**Question:** What does `load_split` do?

**Explanation:** It reads one official data file (train, dev or test) and returns a pandas DataFrame with one
row per tweet: `uid, label, text, tokens, langs, n_tokens`.

**How it works:**
1. `parse_conll` reads the file line by line. A `meta<TAB>uid<TAB>label` line starts a new tweet. A
   `token<TAB>lang` line adds a token. Blank lines are skipped.
2. A malformed line (token before any header, no tab, empty uid) raises `ValueError` with the file and line
   number. Bad data is never silently skipped.
3. `to_frame` builds the DataFrame and the `text` column (tokens joined by spaces).
4. For `test`, the labels are in a separate CSV (`Uid,Sentiment`). They are joined **by uid**, and an error is
   raised if any test tweet has no label.

**Why it matters:** every model and analysis starts from this function, so all of them see exactly the same
tweets.

**Follow-up questions:**
- *Why join the test labels by uid and not by row order?* Row order can differ between files. Joining by uid
  makes a silent misalignment impossible.
- *What happens with an empty file or a file whose first line is a token?* An empty file gives an empty list.
  A token before any header raises `ValueError`.

### 4.2 `src/preprocess.py`: `clean_text`, `normalize_repeated_chars`, `HinglishNormalizer`, `preprocess_text`

**Question:** What preprocessing do you do, and why?

**Explanation:** There are three stages, and each can be switched off. Three named **presets** are used in
experiments: `raw` (nothing), `clean` (noise removal only) and `normalized` (clean + repeated letters + Hinglish
spellings).

**How it works:**
1. `clean_text` uses regular expressions. It first swaps URLs and mentions for private **temporary marker
   characters**, so later steps cannot damage them, and only at the end turns them into `URL` / `USER`. It also
   removes a leading `RT` and Twitter's truncation `…`, keeps hashtag words without `#`, merges several mentions
   in a row into one, and rejoins split contractions (`didn ’ t` → `didn't`). The regexes handle the dataset's
   *already tokenised* form (`https // t . co / x`, `@ name`) as well as normal tweets.
2. `normalize_repeated_chars` cuts runs of **3 or more of the same letter** down to **2** (`goooood` → `good`).
   It cuts to 2, not 1, because many real words have double letters (`good`, `yaar`, `aaj`). It touches letters
   only: `!!!` and repeated emojis are kept as signs of emphasis.
3. `HinglishNormalizer` loads `configs/hinglish_variants.json` (canonical → list of variants). `build_variant_map`
   checks that no variant maps to two canonical forms and that no canonical form is also a variant (otherwise the
   result would depend on order). Matching is **whole-word, case-insensitive**, longest variant first, and the
   original capitalisation is kept (`Acha` → `Achha`).
4. `preprocess_text` runs the enabled stages in order. `load_processed_split` adds the result as `text_proc`.

**Why it matters:** tweets are noisy, and Romanised Hindi has no standard spelling. Cleaning removes noise
that carries no sentiment (links, user names). Normalisation tries to make spelling variants share one
representation.

**Key design rules (from the module docstring and `docs/preprocessing.md`):**
- **Nothing is learned from data**, so applying the same function to train, dev and test cannot leak
  information between splits.
- **Sentiment-bearing material is kept:** negations (`nahi`, `not`), emojis, `!`, hashtag words, capitals.
- A variant is added to the dictionary **only if it is not a common English word** (`he`, `me`, `per` are left
  out even though they are often Hindi). Normalisation cannot see language tags, so it cannot tell Hindi `he`
  from English `he`.

**Follow-up questions:**
- *Did normalisation help?* No. BiLSTM test weighted F1 was clean 0.668 > raw 0.660 > normalized 0.657. The
  differences are small compared with seed variation, and the README states "Normalization did not help."
- *Why are contractions rejoined?* `didn ’ t` would otherwise leave a stray `t` that tokenisers may drop, and
  the negation would be lost.

**Empty or invalid input:** `None` and `NaN` become `""` (`_as_text`). Empty or whitespace-only text returns
`""`. An unknown option value (for example `url="drop"`) raises `ValueError`.

### 4.3 `src/metrics.py`: `compute_metrics`, `confusion`

**Question:** How do you evaluate a model?

**Explanation:** `compute_metrics(y_true, y_pred)` returns weighted F1 (official), macro F1, accuracy, and
precision, recall, F1 and support for each class. `confusion` returns the 3×3 confusion matrix (rows = true,
columns = predicted). Both use a fixed label order: `["negative", "neutral", "positive"]`.

**How it works:** it calls sklearn's `precision_recall_fscore_support` with `labels=LABELS` and
`zero_division=0` (a class that is never predicted gets precision 0 instead of an error). `_check_labels` first
rejects lists of different lengths, empty input and unknown labels.

**Why it matters:** one metric function for every model means every comparison is fair.

**Follow-up questions:**
- *Weighted vs macro F1?* Macro gives every class equal weight. Weighted weights each class by its number of true
  examples, so the larger neutral class counts more. Weighted F1 is the official SemEval metric.
- *Why is weighted recall equal to accuracy?* Weighted recall = Σ (support_c / N) × (TP_c / support_c) = Σ TP_c / N =
  accuracy.

### 4.4 `src/seed.py`: `set_seed`

**Question:** How do you make results reproducible?

**Explanation and how it works:** `set_seed(seed)` seeds Python's `random`, NumPy and PyTorch (CPU and all GPUs),
and makes cuDNN deterministic. The training scripts also seed the DataLoader's shuffling generator. Every result
is reported over **three seeds (42, 43, 44)** as mean ± sample standard deviation.

**Follow-up questions:**
- *Are GPU runs bit-identical?* Not always. Some CUDA kernels are non-deterministic (`docs/transformer_pipeline.md`).
  That is why results are averaged over seeds, and why the frozen test evaluation allows a small tolerance when
  it reproduces dev predictions.
- *Why three seeds and not one?* One seed can be lucky or unlucky. The sd shows how much of a difference between
  models is just noise.

### 4.5 `src/cmi.py` and `src/cmi_evaluation.py`: code-mixing index

**Question:** What is CMI and what did you find?

**Explanation:** The Code-Mixing Index (Das and Gambäck, 2014) measures how mixed a tweet is:
`CMI = 100 × (1 − max(w_i) / (n − u))`, where `n` = tokens, `u` = language-independent tokens (`O`, `EMT`) and
`max(w_i)` = tokens of the most frequent language. It is 0 for one language and gets larger as the mixing gets
more even. If every token is language-independent, it returns 0.

**How `cmi_evaluation` works:**
1. Bucket edges are the 1/3 and 2/3 quantiles of CMI on **train only** (23.81 and 34.78), giving low, medium and
   high buckets.
2. Each dev prediction file is joined to the CMI table by uid (with `check_uid_alignment`).
3. Weighted F1 is computed per bucket, with a **bootstrap** 95% interval from 1,000 resamples with replacement.
   Each file gets its own random generator seeded from 42 and its file name, so adding a new file never changes
   the other files' intervals.
4. Bucket differences (low − high, ...) get intervals from the difference of the resampled scores.

**Finding (`docs/cmi_analysis.md`):** **no clear relationship.** 35 of 36 pairwise intervals contain 0. A possible
(untested) reason: every tweet in this dataset is mixed, so "low" still means moderately mixed.

**Follow-up questions:**
- *Why take bucket edges from train and not dev?* So no dev information shapes the analysis.
- *What does a bootstrap interval that contains 0 mean?* The data cannot rule out "no difference". That is not
  proof that there is no difference ("absence of evidence is not evidence of absence").

### 4.6 `src/baselines_tfidf.py`: TF-IDF + logistic regression

**Question:** What is your baseline and how is it built?

**Explanation:** First a **majority-class floor** (always predict the most common train label, "neutral"):
dev weighted F1 0.2055. Then three TF-IDF feature sets with logistic regression: word 1-2 grams, character 2-5
grams (`char_wb`), and both together.

**How it works:**
1. `fit_features` **fits** the vectorizer(s) on train (vocabulary + IDF) and only **transforms** dev.
2. For each feature set, try `C ∈ {0.1, 1, 10}` (C = inverse regularisation strength) and keep the C with the
   best **dev** weighted F1.
3. Save dev predictions, write a scores JSON, and append to `results/experiment_log.csv`.

Settings: `min_df=2` (drop features seen in only one tweet), `sublinear_tf=True` (1 + log count),
`lowercase=True`, and for words `token_pattern=\S+` (split on spaces only, so emojis and punctuation are kept).

**Why it matters:** it shows how far simple n-gram counts get (about 0.61). The neural models must beat this.
Note that on dev, the BiLSTM (0.612) does **not** beat TF-IDF (0.613).

**Follow-up questions:**
- *What is TF-IDF?* Term frequency × inverse document frequency. A word gets a high weight if it is frequent in
  this tweet but rare across tweets.
- *Why character n-grams for Hinglish?* Different spellings of one Hindi word share many character pieces, so
  they still get similar features. (This is the general idea. The repository does not record why the range 2-5
  was chosen; see section 12.)

### 4.7 `src/bilstm.py`: data encoding, model, checkpoints

**Question:** Explain your BiLSTM architecture.

**Explanation:** For each word, a **word embedding** (100-d) is concatenated with a **character-CNN** vector
(32 filters × kernel sizes 2, 3, 4 = 96-d). The sequence goes through a **bidirectional LSTM** (128 hidden units
per direction). Its outputs are pooled with **max and mean over the real words** (padding is masked out),
passed through dropout and a linear layer, and give 3 logits.

**How it works (the important pieces):**
- `Vocab`: token↔id table. Id 0 = `<pad>`, 1 = `<unk>`, then reserved placeholders `USER` = 2 and `URL` = 3,
  then tokens by frequency (ties alphabetical, so the result is deterministic). `min_freq=2`: words seen only
  once in train become `<unk>`.
- `TextEncoder.fit` builds the vocabularies **from train only**. `encode` lowercases words (not the
  `USER`/`URL` placeholders, so they never clash with the real word "user") and cuts tweets to `max_words` and
  words to `max_chars`. **An empty tweet becomes a single `<unk>`**, so the model always gets at least one word.
- `make_batch`: pads each batch only to its own longest tweet and word (dynamic padding) and returns
  `words (B,T)`, `chars (B,T,C)`, `lengths`, `labels`.
- `CharCNN`: embeds characters, runs a 1-D convolution for each kernel size, and takes the max over positions.
  Windows that overlap padding are excluded, so a word gets the same vector whatever the batch's padding.
- `BiLSTMClassifier.forward`: uses `pack_padded_sequence` so the LSTM skips padding, then masked pooling
  (padding set to −∞ for max, excluded from the mean).
- `save_checkpoint` / `load_checkpoint`: one `.pt` file with weights, config, vocabularies and metadata. It is
  written atomically and loaded with `weights_only=True` (a checkpoint file cannot run code). Loading warns if
  the Hinglish dictionary has changed since training.

**Why a char CNN and not a char BiLSTM** (`docs/bilstm.md`): every word of every tweet is encoded (about 26 per
tweet), so a CNN is much cheaper. Its n-gram filters also suit short words and spelling variants.

**Follow-up questions:**
- *Why bidirectional?* The meaning of a word depends on words before *and* after it (for example a negation
  after the verb in Hindi word order: "accha nahi hai").
- *Why max + mean pooling instead of the last hidden state?* Max picks up the strongest signal anywhere in the
  tweet, and mean summarises all of it. The `"last"` option exists in the code too. The repository does not
  record a comparison that justified `maxmean`; see section 12.

### 4.8 `src/train_bilstm.py`: `derive_shared_limits`, `train_one_seed`, `run_experiment`

**Question:** How is the BiLSTM trained?

**How `train_one_seed` works:**
1. `set_seed(seed)`, then fit the encoder on train and build the train and dev datasets.
2. Adam (lr 1e-3, weight decay 1e-5), `CrossEntropyLoss`, batch size 64, gradient clipping at 5.0.
3. Each epoch: train, then score dev. If dev weighted F1 improved, keep a copy of the weights. Otherwise
   increase the patience counter and stop after **4** epochs without improvement (max 20 epochs).
4. Reload the best weights, predict dev again, and save a checkpoint.

**`derive_shared_limits`:** length limits (max words per tweet, max characters per word) are measured on
**train** under each of the three modes, and the **maximum** is used for all of them (56 words, 18 chars). Then
raw, clean and normalized are truncated exactly the same way, which makes the comparison fair.

**`run_experiment` output protection:** before training, it stops if any output of this run already exists.
It writes into a hidden staging folder and publishes only after all seeds finish. With `--overwrite`, old files
are moved to `archive/`, never deleted. The experiment log is append-only.

**Follow-up questions:**
- *Why is the dev score "slightly optimistic"?* The best epoch was **chosen** on dev, so dev is no longer an
  unbiased estimate. That is why there is a separate one-time test evaluation.
- *What does gradient clipping do?* It rescales the gradient when its norm exceeds a limit, which prevents
  exploding gradients in RNNs.

### 4.9 `src/transformer_utils.py` and `src/train_transformer.py`: fine-tuning mBERT / XLM-R

**Question:** How do you fine-tune a transformer?

**Explanation:** One script works for any Hugging Face encoder. The model is a setting (`--model mbert|xlmr|...`),
not a separate code path. `AutoModelForSequenceClassification` adds a new randomly initialised 3-class layer on
top of the pretrained encoder, and the **whole network** is trained.

**How it works:**
1. **Max length:** `derive_max_length` measures subword lengths of the **train** tweets with this model's
   tokenizer and rounds the longest up to a multiple of 8, capped at the model limit (512). Result: mBERT 80,
   XLM-R 96. It is measured per tokenizer because the same tweet splits into different numbers of subwords.
2. **Data:** `EncodedTweets` tokenises once, without padding. `make_collate` pads each batch to its own longest
   tweet (dynamic padding), and the attention mask tells the model to ignore padding.
3. **Optimiser:** AdamW with weight decay 0.01, except on biases and LayerNorm (standard BERT practice).
4. **Schedule:** linear warm-up over the first 10% of steps, then linear decay to 0.
5. **Mixed precision (fp16)** on GPU, with `GradScaler` so small gradients do not underflow. Optional gradient
   accumulation. Gradient clipping at 1.0.
6. Score dev after each epoch, keep the best epoch, early-stop with patience 2.

**Experiment protocol** (`src/run_transformer_experiments.py`, `docs/transformer_results.md`):
- **Learning-rate search** with seed 42 for each lr in {1e-5, 2e-5, 3e-5}, **6 epochs**. Chosen: mBERT 2e-5,
  XLM-R 1e-5. Ties go to the smaller lr.
- **Why 6 epochs, not the default 4:** with 4 epochs, every run had its best dev score in the *last* epoch, so
  the models were still improving.
- Then seeds 43 and 44 at the chosen lr. Seed 42 helped choose the lr, so its score is a little optimistic. The
  docs also report the mean of seeds 43 and 44 alone.
- **Text mode `clean`:** URLs and mentions become placeholders, but spellings are left alone "because subword
  tokenizers handle variants" (`docs/transformer_pipeline.md`).

**Follow-up questions:**
- *Why warm-up?* At the start, the new classification layer is random and its gradients are large. A small
  learning rate at first protects the pretrained weights.
- *Why could a 4-epoch run not simply be continued to 6?* The learning-rate schedule depends on the total number
  of steps, so the runs had to be redone from scratch.

### 4.10 `src/indian_transformer.py`, `src/train_muril.py`, `src/train_indicbert.py`

**Question:** How were MuRIL and IndicBERT trained, and how is it different from mBERT/XLM-R?

**Explanation:** This is a separate, simpler loop written before `train_transformer.py` reached `main`. It uses
the same optimiser recipe (AdamW, warm-up and decay, fp16 on GPU), but:
- **Fixed padding** to `max_len=96` for every tweet, instead of dynamic padding. 96 covers MuRIL's longest train
  tweet (91 subwords).
- **Raw text** (default `--text-mode raw`), 3 epochs, lr 2e-5, **no early stopping**. The best dev epoch's
  predictions are kept.
- IndicBERT: the original `ai4bharat/indic-bert` is gated on the Hugging Face Hub, so the reported run used
  **`ai4bharat/IndicBERTv2-MLM-only`** via `--checkpoint`. Only seed 42 exists.
- `load_texts` **refuses** the test split on purpose.

**Follow-up questions:**
- *Can you compare MuRIL's dev score directly with XLM-R's?* Only with care: different text mode (raw vs
  clean), padding, epochs and lr protocol. Neither MuRIL nor IndicBERT was evaluated on test.
- *Why might an Indian-language model help on Hinglish?* MuRIL's pretraining is reported to include
  transliterated (Roman-script) Indian text. The repository's doc says this comes from memory of the model card
  and should be verified before citing.

### 4.11 `src/zero_shot_llm.py`: zero-shot LLM

**Question:** How does the zero-shot classifier work?

**Explanation:** No training at all. An instruction-tuned LLM gets an instruction plus the tweet and answers in
text. Dev labels are used only afterwards, for scoring.

**How it works:**
1. `build_prompt` fills a template. **v1** is a plain instruction. **v2** explains what Hinglish is, gives
   example words (`nahi`, `bakwas`, `accha`) and defines neutral. There are no labelled examples, otherwise it
   would be few-shot.
2. A backend (local Hugging Face model with greedy decoding, or the Anthropic API with temperature 0) returns
   text.
3. `parse_response`: the answer is **valid only if exactly one** of positive/negative/neutral appears as a word.
   Empty answers, answers with no label and answers with two different labels are invalid and get the fallback
   label `neutral`. They are **counted and reported**.
4. `classify` catches backend errors (network, quota), so one failure does not stop a long run. `run` saves
   every 10 answers and skips uids already done, so an interrupted run can be resumed.

**Result:** Qwen2.5-0.5B-Instruct on 200 random dev tweets: v1 answered "neutral" for 197/200. v2 reached weighted
F1 0.477. This shows that prompt wording matters a lot for a tiny model. It says nothing about large LLMs.

**Follow-up questions:**
- *Why map invalid answers to neutral instead of dropping them?* So the prediction file covers every tweet and the
  score is not inflated by skipping hard cases. A valid-answers-only score is also reported.
- *Why was v2 the primary prompt?* It was chosen **before** any scores were seen, so the result was not picked on
  the evaluation tweets.

### 4.12 `src/evaluate_transformer_test.py` and `src/eval_protocol_utils.py`: the one-time test evaluation

**Question:** How did you make sure the test set was used fairly?

**Explanation:** The test set was used **once**, by a script that refuses to run without `--confirm-final-test`,
on six **frozen** checkpoints (mBERT and XLM-R × seeds 42, 43, 44).

**How it works:**
1. **Pre-flight:** every checkpoint file and every frozen dev prediction file is **MD5-verified** against
   `configs/transformer_frozen_checkpoints.json`. Each checkpoint's `run_info.json` must match the run the
   manifest names (model, seed, lr, mode, epochs, max_length).
2. **Dev reproduction:** each checkpoint predicts dev again and must agree with its frozen dev predictions
   (≥ 99% agreement, weighted F1 within 0.005). This proves the inference code is the one the models were
   selected with.
3. **Stage A:** parse the **unlabeled** test file and predict with all six checkpoints. Write the predictions,
   read them back and validate them.
4. **Stage B:** only now is the labels file hashed and read **for the first time**. Labels are joined **by uid**,
   and metrics are computed **from the saved files**.
5. All inputs are hashed again at the end, outputs are published with one atomic rename, and an existing output
   folder is never overwritten.

`--validate-only` runs steps 1 and 2 without opening either test file.

**Follow-up questions:**
- *Why save predictions before reading the labels?* Then the predictions provably could not have been influenced
  by the labels.
- *Why is a small dev disagreement allowed?* GPU fp16 kernels are not bit-exact, so a few predictions can flip.

### 4.13 Analysis and reporting scripts

- `src/error_analysis.py` (TF-IDF only) and `src/error_analysis_models.py` (all models): find every dev error and
  add **automatic flags**. The TF-IDF script flags negation word, spelling variant, truncated, URL, repeated
  letters and emoji. The all-models script flags negation, slang/informal, sarcasm cue (gold negative but the text
  looks positive), spelling variant, very short and high CMI. Both compare error rates with and without each flag
  (Bonferroni threshold / chi-square) and give a sample of errors to read by hand. **The flags are heuristics,
  not proof of the cause.** In the all-models script, the seeds of one setting are merged by majority vote.
  - Main finding (`docs/baselines_tfidf.md`): almost all errors involve **neutral**. Negative↔positive swaps are
    rare. Annotator agreement in the original dataset was only 55% (Patwa et al.), which fits a fuzzy neutral
    boundary.
- `src/label_agreement.py`: Cohen's kappa between two human readers' error categories.
- `src/transformer_comparison.py`: mean and sd per model, per-seed differences, and a **paired bootstrap**
  (both models scored on the *same* resampled tweets).
- `src/final_comparison.py`: builds the results table in section 1 from saved files (nothing is retrained).
- `src/log_metrics.py`: recomputes scores in `experiment_log.csv` from the prediction files and stops on any
  disagreement.
- `src/eda_part1.py`, `src/eda_part2.py`: label distribution, lengths, language shares, CMI distribution, and
  overlap checks between splits.

**Follow-up questions:**
- *Paired vs unpaired bootstrap?* Paired resamples the same tweets for both models, which removes the "which
  tweets were drawn" noise and makes the comparison sharper. Unpaired is used for CMI buckets because the buckets
  hold *different* tweets.
- *What is a Wilson interval?* A confidence interval for a proportion that stays inside [0, 1] and works for small
  counts.

---

## 5. Important model and preprocessing decisions

| Decision | Reason given in the repository |
|---|---|
| Weighted F1 as the main metric | It is the official SemEval-2020 Task 9 metric (`src/metrics.py`). |
| Everything learned (vocabularies, IDF, lengths, CMI edges) comes from **train only** | No information from dev or test leaks into training. |
| Model and epoch selection on **dev**, test used **once** | Gives an unbiased final estimate (protocol docs). |
| Three preprocessing modes kept as presets | To measure the effect of cleaning and normalisation as an **ablation**. |
| Transformers use `clean`, not `normalized` | "Subword tokenizers handle variants" (`docs/transformer_pipeline.md`). |
| Char CNN instead of char BiLSTM | Cheaper for ~26 words per tweet; n-gram filters suit spelling variants (`docs/bilstm.md`). |
| Shared length limits across modes | Same truncation for raw, clean and normalized makes the comparison fair (`docs/bilstm.md`). |
| `max_length` per tokenizer, from train | Same tweet = different subword counts per model; 56 words can be up to ~90 subwords. |
| 6 epochs for transformers | With 4, every run peaked in the last epoch (`docs/transformer_results.md`). |
| 3 seeds, mean ± sample sd | Shows how much of a difference is seed noise. |
| Dictionary excludes common English words | Text-level normalisation cannot tell Hindi `he` from English `he`. |
| IndicBERT v2 instead of v1 | v1 is gated on the Hugging Face Hub (`docs/indian_transformer_results.md`). |
| Zero-shot with a 0.5B model | No API key or GPU was available (`docs/zero_shot_results.md`). |

---

## 6. Limitations

- **Dev–test gap is partly unexplained.** Test scores are 0.05 to 0.08 higher than dev for every model. Class
  mix explains almost none of it.
- **Only 3 seeds**, so standard deviations are rough. One 3,000-tweet test set has about ±0.01 F1 sampling error.
- **Dev scores are optimistic** (best epoch and lr chosen on dev, and seed 42 used for the lr choice).
- **The neutral class is the main source of error.** Low annotator agreement (55%) limits any model.
- **Normalisation dictionary is small and hand-made.** It cannot use context or language tags, and it did not
  improve results.
- **MuRIL, IndicBERT, TF-IDF and zero-shot have no test evaluation.** MuRIL and IndicBERT also used a different
  training loop and raw text, so their comparison with mBERT and XLM-R is not like-for-like. IndicBERT has one
  seed.
- **Zero-shot result is on 200 tweets with a tiny model**, so it cannot be compared with the full-dev rows.
- **No deployment:** there is no inference API or UI. Prediction on new text is possible only from Python (for
  example `LoadedCheckpoint.predict_texts` for the BiLSTM).
- **CMI analysis has low power:** with ~1,000 tweets per bucket, only differences of about 0.04 or more could be
  detected.

---

## 7. Common bugs and edge cases (and how the code handles them)

| Situation | What happens |
|---|---|
| Empty or whitespace tweet | `clean_text` → `""`; BiLSTM `encode` → one `<unk>` word; the transformer still gets `[CLS] [SEP]` and predicts something. |
| `None` / `NaN` text | `_as_text` turns it into `""`. |
| Word never seen in train | BiLSTM: `<unk>` id 1 (characters still help through the char CNN). Transformers: split into known subwords. |
| Tweet longer than the limit | Truncated (beginning kept); truncation counts are logged. |
| Label typo or unknown label | `compute_metrics`, `SentimentDataset` and `EncodedTweets` raise `ValueError`. |
| `y_true` and `y_pred` of different lengths, or empty | `compute_metrics` raises `ValueError`. |
| A merge drops or repeats rows | `check_uid_alignment` raises; merges use `validate="one_to_one"`. |
| LLM answers "positive or negative" or refuses | Marked invalid, fallback `neutral`, counted in `n_invalid`. |
| Running training twice with the same run name | Stops before training (`FileExistsError`) unless `--overwrite`; old files go to `archive/`. |
| Crash halfway through a multi-seed run | Existing results untouched; partial work kept in `failed_runs/`. |
| Hinglish dictionary changed after training a BiLSTM | `load_checkpoint` warns (SHA-256 fingerprint mismatch). |
| `device=cuda` with no GPU | `transformer_utils.resolve_device` raises instead of silently using the CPU. |
| Normaliser matching inside a word (`nhi` inside `nhin'`) | Prevented by look-arounds that forbid word characters and apostrophes on either side. |

---

## 8. How to test and demonstrate the project

Run every command from the repository root, with the virtual environment active.

```powershell
.venv\Scripts\python.exe -m pytest -q
```

Tests use small synthetic CoNLL files and tiny random models (`tests/synthetic_conll.py`), so they run on a CPU
without downloading anything.

A short live demo (fast, CPU only):

```powershell
.venv\Scripts\python.exe -c "from src.preprocess import preprocess_text as p; print(p('RT @ abc Movie bohot goooood thi yaar !!! https // t . co / xyz', clean=True, repeats=True, hinglish=True))"
```

```powershell
.venv\Scripts\python.exe -c "from src.cmi import cmi; print(cmi(['Hin','Hin','Eng','O']), cmi(['Eng','Eng']))"
```

```powershell
.venv\Scripts\python.exe -c "from src.metrics import compute_metrics as m; print(m(['positive','neutral','negative'], ['positive','neutral','neutral'])['weighted_f1'])"
```

```powershell
.venv\Scripts\python.exe -m src.final_comparison
```

A quick training smoke run (nothing is saved, so it is safe):

```powershell
.venv\Scripts\python.exe -m src.train_bilstm --mode clean --limit 300 --set epochs=1 --no-checkpoints
```

Expected outputs (checked): the preprocessing command prints `USER Movie bahut good thi yaar !!! URL`. `RT` is
gone, the mention and link became placeholders, `bohot` became `bahut`, `goooood` became `good`, and `!!!` was kept.
The CMI command prints `33.33… 0.0`, because the Hin, Hin, Eng tweet gives 100 × (1 − 2/3) and an all-English tweet
gives 0. The metrics command prints `0.5555…`.

---

## 9. Concepts to revise before the assessment

1. **Precision, recall, F1; macro vs weighted averaging; confusion matrix.** Be able to compute them by hand for
   a 3-class example.
2. **TF-IDF** formula, n-grams, and `fit` on train vs `transform` on dev.
3. **Logistic regression** (multiclass), the regularisation parameter C.
4. **Word embeddings, 1-D CNN over characters, max-over-time pooling.**
5. **LSTM / BiLSTM**: gates, why bidirectional, packed sequences and padding masks.
6. **Cross-entropy loss, softmax, logits, argmax.**
7. **Adam vs AdamW**, weight decay, gradient clipping, dropout, early stopping.
8. **Transformers:** subword tokenisation (WordPiece for mBERT, SentencePiece for XLM-R), `[CLS]`, attention mask,
   fine-tuning vs feature extraction.
9. **Learning-rate warm-up and linear decay**; mixed precision (fp16) and loss scaling.
10. **Train / dev / test split discipline**, data leakage, why dev scores after model selection are optimistic.
11. **Bootstrap confidence intervals** (paired vs unpaired), Wilson interval, Cohen's kappa, Bonferroni correction.
12. **Code-mixing index** and what code-mixing / transliteration mean for NLP.
13. **Zero-shot vs few-shot prompting**, greedy decoding / temperature 0.

---

## 10. What the evaluator might ask, and short answers

- *What is the hardest part of Hinglish?* There is no standard spelling for Romanised Hindi, and two languages
  share one script. The same Roman word can be Hindi or English (`he`, `me`), and tweets are noisy (mentions,
  links, truncation).
- *Which model is best and by how much?* XLM-R base: test weighted F1 0.7006 ± 0.0041 vs mBERT 0.6866 ± 0.0081 and
  BiLSTM (clean) 0.6681 ± 0.0023.
- *Did your preprocessing help?* For the BiLSTM, `clean` was marginally best and `normalized` did not help. The
  differences are within seed noise.
- *How do you know there is no data leakage?* Everything learned is fitted on train. Selection uses dev. Training
  scripts never load test. Test was run once, with predictions saved before the labels were read.
- *What would you do next?* Ideas that follow from the documented limitations: test-set runs for MuRIL/IndicBERT
  through the shared pipeline, more seeds, a larger LLM for zero-/few-shot, and investigating the dev–test gap.
  (These are suggestions. They are not in the code.)

---

## 11. Files to know by heart

| File | One line |
|---|---|
| `src/data.py` | Reads CoNLL files into one row per tweet. |
| `src/preprocess.py` | raw / clean / normalized text. |
| `src/metrics.py` | The single metric function (weighted F1). |
| `src/bilstm.py` | BiLSTM vocab, encoding, model, checkpoints. |
| `src/train_bilstm.py` | BiLSTM training, early stopping, safe outputs. |
| `src/transformer_utils.py` | Shared transformer pieces (tokeniser, lengths, batching, predict). |
| `src/train_transformer.py` | Fine-tuning loop for any transformer. |
| `src/evaluate_transformer_test.py` | One-time frozen test evaluation. |
| `configs/hinglish_variants.json` | Spelling dictionary for normalisation. |
| `configs/transformer_frozen_checkpoints.json` | Hashes and settings of the six frozen checkpoints. |
| `results/final_model_comparison.csv` | All results in one table. |

---

## 12. Open questions for you

The repository does not record the reason for these choices. Find out the answer (ask the teammate who wrote it,
or check old issues or PRs) instead of guessing in the assessment:

1. **TF-IDF n-gram ranges** (word 1-2, char 2-5) and `min_df=2`: why these values? Only C was tuned.
2. **BiLSTM hyperparameters** (word_dim 100, hidden 128, dropout 0.4, kernels 2/3/4, patience 4, lr 1e-3): were they
   tuned, or taken as common defaults? `docs/bilstm.md` explains the architecture but not these numbers.
3. **`maxmean` pooling** as the default: was it compared with `last`, `mean` and `max` on dev?
4. **MuRIL/IndicBERT on raw text with 3 epochs:** the shared pipeline uses clean text and 6 epochs. Was raw chosen
   on purpose, or was it the default of the earlier script?
5. **Why no class weighting** in the loss? The classes are only mildly imbalanced (29 / 38 / 33%), which may be the
   reason, but it is not written down.
6. **IndicBERT** has a single seed (42). Was the plan to run 43 and 44?
7. The README says the team's target Python is **3.11**, but the BiLSTM results were produced on Python 3.12.5
   (CPU) and the transformer test evaluation on Python 3.13.2 with torch 2.14.1 on an RTX 4050 Laptop GPU
   (`results/transformer_final_test/integrity_manifest.json`). Be ready to explain the mismatch.
