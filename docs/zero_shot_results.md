# Zero-shot LLM sentiment classification (issue #8)

Code: `src/zero_shot_llm.py`. Tests: `tests/test_zero_shot_llm.py` (16, fake backends, no model needed).

## Model used (and what this result is NOT)
- **Model:** `Qwen/Qwen2.5-0.5B-Instruct` (Hugging Face), revision `7ae557604adf67be50417f59c2c2f167def9a775`, run locally on the CPU of an ordinary laptop (Intel i5-1334U, no GPU), greedy decoding (deterministic), run on 2026-11-21.
- **Why this model:** no API key or GPU was available. A 0.5-billion-parameter model is tiny and cheap. **These numbers say how a very small open model behaves; they do not say how a strong LLM (for example a Claude, GPT or Llama-70B class model) would do on Hinglish.** The report must not present them as "the LLM result" for the project without that qualification. The Anthropic backend (`--backend anthropic`) is implemented and reads the key only from the environment variable `ANTHROPIC_API_KEY`, but it has **not been run** (no key), so it has only been tested with a fake client in the unit tests.
- No training and no fine-tuning: the model gets an instruction and the tweet only. Dev labels are used only afterwards to score answers. No credentials are stored in the repository.

## Prompts
Two zero-shot prompts, no labelled examples (full text in `PROMPTS` in `src/zero_shot_llm.py` and in `results/zero_shot_run_meta*.json`):
- **v1:** plain instruction: classify as positive, negative or neutral, reply with one word.
- **v2:** says the tweet is Hinglish (Hindi in Roman letters, inconsistent spelling, a few glossed example words such as nahi = no), defines neutral (no clear opinion, news, questions, plain statements), and asks for one word without explanation. The glossed example words are vocabulary hints, not labelled tweets.

**v2 was chosen as the primary prompt before any scores were seen**, so the primary result was not picked on the evaluation tweets. v1 is reported as a comparison. Only these two prompts were tried.

## Output mapping and invalid answers
`parse_response` accepts an answer only if exactly one of positive/negative/neutral appears as a whole word (case and punctuation ignored). Empty answers, answers with no label word, and answers containing two different labels ("positive or negative") are *invalid*; backend errors (network, quota) are also recorded as invalid and do not stop the run. Invalid answers get the fallback label `neutral` so the predictions file stays complete, and they are counted separately (`n_invalid`, reasons in the meta file); a score on valid answers only is also stored. In both real runs below the number of invalid answers was **0 of 200**, so this handling was exercised only by the unit tests, not by real model output.

## Evaluation
A random subset of **200 dev tweets** (fixed seed 42; 50 negative, 84 neutral, 66 positive), because CPU inference is slow. The same 200 tweets were used for both prompts and for the baseline comparison. Subset size means the scores are noisy; no confidence interval was computed, and differences of a few points should not be interpreted.

| | Weighted F1 | Macro F1 | Accuracy | Invalid | Recall neg / neu / pos | Predicted neg / neu / pos |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B, **prompt v2** (primary) | 0.4768 | 0.4515 | 0.495 | 0/200 | 0.20 / 0.73 / 0.42 | 28 / 134 / 38 |
| Qwen2.5-0.5B, prompt v1 | 0.2547 | 0.2042 | 0.415 | 0/200 | 0.00 / 0.98 / 0.02 | 0 / 197 / 3 |

For reference, existing supervised models on the **same 200 tweets** (weighted F1, from the existing dev prediction files): TF-IDF char 2-5 0.6745, TF-IDF word 1-2 0.6654, TF-IDF word+char 0.6453, BiLSTM clean/normalized/raw seed 42: 0.6541 / 0.6365 / 0.6501. (The always-neutral baseline scores 0.205 on the full dev set.)

### Reading the results
- The small model is clearly below the supervised baselines (about 0.48 vs 0.64-0.67 weighted F1 on these tweets).
- With the plain prompt v1 it answers "neutral" for 197 of 200 tweets, which is almost the always-neutral floor. The prompt that describes Hinglish and defines neutral (v2) makes it use all three labels and raises weighted F1 by 0.22. That shows prompt wording matters a lot for a small model; it does not show which part of v2 helped, because several things changed at once.
- Even with v2 the model rarely says negative (28 predictions for 50 true negatives; recall 0.20) and leans to neutral (134 predictions for 84 true neutrals).

## Latency and cost
Mean time per tweet on this CPU: 3.6 s (v2) and 4.2 s (v1); wall time 12 and 14 minutes for 200 tweets, including loading the model. Machine load varied, so treat these as rough. Cost: no API cost (local model; the 1 GB download is the only resource). Extrapolating, the full 3,000-tweet dev set would take roughly three hours per prompt on this CPU (estimate, not run). An API model would need token counts: the Anthropic backend records them in `api_token_usage` in the meta file.

## Limitations
- Tiny model, one run, 200 tweets, two prompts; no repetition (greedy decoding is deterministic, so a rerun gives the same answers).
- Not a statement about stronger LLMs.
- The test set was not used.

## Reproduce
```bash
python -m src.zero_shot_llm --backend hf --model Qwen/Qwen2.5-0.5B-Instruct --n 200 --prompt v2
python -m src.zero_shot_llm --backend hf --model Qwen/Qwen2.5-0.5B-Instruct --n 200 --prompt v1 --out-suffix _v1
# with an API model (key in the environment, not in the repo):
#   set ANTHROPIC_API_KEY=...   then   python -m src.zero_shot_llm --backend anthropic --model <model id> --n 200
```
Runs resume if interrupted. Committed files: `results/zero_shot_dev_predictions.csv` (primary prompt v2, `uid,label,pred`), `results/zero_shot_run_meta.json` (v2: model revision, prompt text, counts, metrics) and `results/zero_shot_run_meta_v1.json` (the v1 comparison row). The per-tweet raw model answers (`zero_shot_dev_raw_responses*.csv`) and the v1 predictions file are produced by the commands above but are not committed; rerun to regenerate them. The primary predictions file is read automatically by `src.error_analysis_models` the next time it is run (it covers only 200 tweets, which that script handles).
