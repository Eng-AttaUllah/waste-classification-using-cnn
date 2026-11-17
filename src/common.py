"""Shared constants and helpers for the waste classification project."""
from pathlib import Path

# ---------------------------------------------------------------- paths ----
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"          # downloaded Kaggle dataset
SPLIT_DIR = DATA_DIR / "splits"     # train / val / test folders
OUTPUT_DIR = PROJECT_DIR / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
FIG_DIR = OUTPUT_DIR / "figures"
METRIC_DIR = OUTPUT_DIR / "metrics"

for _d in (DATA_DIR, RAW_DIR, SPLIT_DIR, OUTPUT_DIR, MODEL_DIR, FIG_DIR, METRIC_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------- constants ----
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42
MODEL_PATH = MODEL_DIR / "waste_cnn.keras"
HISTORY_PATH = METRIC_DIR / "history.json"

# Ratios for train / validation / test
TRAIN_RATIO, VAL_RATIO = 0.70, 0.15  # test = 1 - train - val = 0.15
