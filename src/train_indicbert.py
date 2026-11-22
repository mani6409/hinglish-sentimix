"""Fine-tune IndicBERT (ai4bharat/indic-bert) for Hinglish sentiment. Issue #7 (optional part).

Uses exactly the same training code as MuRIL (src/indian_transformer.py); only the checkpoint differs.
    python -m src.train_indicbert --inspect-only   # compatibility / tokenizer report
    python -m src.train_indicbert                  # writes results/indicbert_dev_predictions.csv
IndicBERT is an ALBERT model; its tokenizer needs the `sentencepiece` package (see requirements.txt).
"""
from src.indian_transformer import main_for

CHECKPOINT = "ai4bharat/indic-bert"

if __name__ == "__main__":
    main_for("indicbert", CHECKPOINT)
