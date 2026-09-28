"""
Central filesystem paths, resolved relative to the project root so the pipeline
works no matter which directory it is invoked from.
"""

from pathlib import Path

# src/prem_prediction/paths.py  ->  project root is three levels up
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Data files
MATCHES_FILE = DATA_DIR / "historical_matches.csv"
STANDINGS_FILE = DATA_DIR / "historical_standings.csv"
TRAINING_FILE = DATA_DIR / "out.csv"
SQUAD_VALUE_FILE = DATA_DIR / "squad_values.csv"

# Model artifacts
MODEL_FILE = MODELS_DIR / "best_model.h5"
SCALER_FILE = MODELS_DIR / "scaler.joblib"

# Outputs (predictions + visualisations)
PREDICTIONS_FILE = OUTPUTS_DIR / "predictions.json"
GIF_FILE = OUTPUTS_DIR / "title_race.gif"
COMPARISON_GIF_FILE = OUTPUTS_DIR / "model_vs_kalshi.gif"
DASHBOARD_FILE = OUTPUTS_DIR / "dashboard.html"

# Kalshi market comparison cache (data/)
KALSHI_FILE = DATA_DIR / "kalshi_current.csv"


def ensure_dirs() -> None:
    """Create the data/models/outputs directories if they do not yet exist."""
    for d in (DATA_DIR, MODELS_DIR, OUTPUTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
