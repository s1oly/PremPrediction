"""
Generate Kalshi-style Premier League title probabilities for the season in
progress, one distribution per completed gameweek.

Usage:
    python -m prem_prediction.season_predict

Writes outputs/predictions.json and prints a summary. Requires a trained model
(models/best_model.h5) and scaler (models/scaler.joblib).
"""

import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from .config import current_season_start_year, season_label
from .data_acquisition import FDCUKDataLoader
from .model_train import FEATURES, MAX_GAMEWEEK
from .paths import (
    MATCHES_FILE,
    MODEL_FILE,
    PREDICTIONS_FILE,
    RESIDUAL_STD_FILE,
    SCALER_FILE,
    STANDINGS_FILE,
    ensure_dirs,
)
from .prepare_model_data import prepare

N_SIMS = 20000  # Monte Carlo draws per gameweek for the title-probability estimate

PREDICT_SEASON = current_season_start_year()  # e.g. 2026 for the 2026-27 season


def pad_sequence(X: np.ndarray, max_len: int = MAX_GAMEWEEK) -> np.ndarray:
    """Pad a (T, F) array to (max_len, F) with zeros."""
    pad_len = max_len - X.shape[0]
    if pad_len < 0:
        raise ValueError(f"Sequence longer than max_len ({max_len})")
    if pad_len == 0:
        return X
    return np.vstack([X, np.zeros((pad_len, X.shape[1]))])


def get_team_points(model, scaler, team_df: pd.DataFrame) -> float:
    """Predicted final-season points for a single team's partial-season sequence."""
    team_df = team_df.sort_values("gameweek")
    X = scaler.transform(team_df[FEATURES])
    X_padded = pad_sequence(X, MAX_GAMEWEEK)
    return float(model.predict(np.expand_dims(X_padded, axis=0), verbose=0)[0][0])


def predict_gameweek(model, scaler, season_df, gameweek, resid_std, n_sims=N_SIMS) -> dict[str, float]:
    """
    Predict every team's final points as of `gameweek`, then derive the title
    probability as P(a team's final points are the league's highest) by Monte Carlo:
    sample each team's final total from N(predicted, std[gameweek]), clipped to the
    realistic band [points already earned, maximum still obtainable], and count how
    often each team tops the simulated table. std[gameweek] is the model's own
    final-points error at that stage (wide early, tight late), so the probabilities
    sharpen as the season progresses.
    """
    gw_df = season_df[season_df["gameweek"] <= gameweek]
    teams = list(gw_df["teamName"].unique())

    std = float(resid_std.get(str(gameweek), resid_std.get("overall", 8.0)))
    std = max(std, 1e-6)

    preds, floors, ceils = [], [], []
    for t in teams:
        tdf = gw_df[gw_df["teamName"] == t]
        latest = tdf.sort_values("gameweek").iloc[-1]
        preds.append(get_team_points(model, scaler, tdf))
        floors.append(float(latest["points"]))                 # can't finish below current points
        ceils.append(float(latest["max_obtainable_points"]))   # or above the max still obtainable

    preds = np.array(preds); floors = np.array(floors); ceils = np.array(ceils)
    # Sample final points for every team and tally who tops the table each sim.
    samples = np.random.normal(preds[:, None], std, size=(len(teams), n_sims))
    samples = np.clip(samples, floors[:, None], ceils[:, None])
    winners = np.argmax(samples, axis=0)  # index of the top team in each simulation
    counts = np.bincount(winners, minlength=len(teams))
    probs = counts / float(n_sims)
    return dict(zip(teams, probs))


def build_current_season_frame() -> pd.DataFrame:
    """
    Download the in-progress season and process it with full Elo continuity by
    chaining onto the saved historical matches/standings.
    """
    loader = FDCUKDataLoader()
    print(f"Downloading {season_label(PREDICT_SEASON)} season data...")
    cur_matches = loader.download_season_matches(PREDICT_SEASON)
    if cur_matches is None or cur_matches.empty:
        raise RuntimeError(f"Could not download {season_label(PREDICT_SEASON)} data.")
    print(f"  {len(cur_matches)} fixtures downloaded.")

    hist_matches = pd.read_csv(MATCHES_FILE)
    hist_standings = pd.read_csv(STANDINGS_FILE)

    all_matches = pd.concat([hist_matches, cur_matches], ignore_index=True)
    cur_standings = loader.compute_standings_from_matches(cur_matches, PREDICT_SEASON)
    all_standings = pd.concat([hist_standings, cur_standings], ignore_index=True)

    processed = prepare(all_standings, all_matches)
    return processed[processed["season"] == PREDICT_SEASON].copy()


def predict_all_gameweeks(model, scaler, season_df, resid_std) -> dict[int, dict[str, float]]:
    """Return {gameweek: {team: title probability}} for every completed gameweek."""
    max_gw = int(season_df["gameweek"].max())
    predictions = {}
    for gw in range(1, max_gw + 1):
        if season_df[season_df["gameweek"] == gw].empty:
            continue
        predictions[gw] = predict_gameweek(model, scaler, season_df, gw, resid_std)
    return predictions


def save_predictions(predictions: dict[int, dict[str, float]], season: int) -> None:
    """Write predictions to outputs/predictions.json for the visualisation layer."""
    ensure_dirs()
    payload = {
        "season": season,
        "season_label": season_label(season),
        "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gameweeks": {
            str(gw): {team: round(float(p), 6) for team, p in probs.items()}
            for gw, probs in predictions.items()
        },
    }
    with open(PREDICTIONS_FILE, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nPredictions written to {PREDICTIONS_FILE}")


def load_residual_std() -> dict:
    """Load the per-gameweek final-points error std; empty dict falls back to a default."""
    if RESIDUAL_STD_FILE.exists():
        with open(RESIDUAL_STD_FILE) as f:
            return json.load(f)
    return {}


def main():
    model = load_model(MODEL_FILE, compile=False)
    scaler = joblib.load(SCALER_FILE)
    resid_std = load_residual_std()

    season_df = build_current_season_frame()
    max_gw = int(season_df["gameweek"].max())
    print(f"\nPredicting gameweeks 1-{max_gw} for {season_label(PREDICT_SEASON)}.\n")

    predictions = predict_all_gameweeks(model, scaler, season_df, resid_std)
    save_predictions(predictions, PREDICT_SEASON)

    latest = predictions[max_gw]
    print("\n" + "=" * 44)
    print(f"CURRENT TITLE PROBABILITIES (after GW {max_gw})")
    print("=" * 44)
    for team, prob in sorted(latest.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(prob * 40)
        print(f"  {team:<22} {prob * 100:5.1f}%  {bar}")


if __name__ == "__main__":
    main()
