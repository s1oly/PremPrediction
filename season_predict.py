"""
Generates Kalshi-style Premier League title probability predictions for the
current 2025-26 season, one output block per completed gameweek.

Usage:
    python season_predict.py

Outputs a formatted table to stdout, e.g.:

    === Gameweek 1 ===
    Liverpool          34.2%
    Arsenal            18.7%
    ...

Requires:
    best_model.h5   — trained LSTM (from model_train.py)
    scaler.joblib   — fitted StandardScaler (from model_train.py)
"""

import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

from data_aquisition import FDCUKDataLoader
from prepare_model_data import convert_form, add_elo_feature, add_relative_features
from model_train import FEATURES, MAX_GAMEWEEK

PREDICT_SEASON = 2025  # 2025-26 season


def pad_sequence(X: np.ndarray, max_len: int = MAX_GAMEWEEK) -> np.ndarray:
    """Pad a (T, F) array to (max_len, F) with zeros."""
    pad_len = max_len - X.shape[0]
    if pad_len < 0:
        raise ValueError(f"Sequence longer than max_len ({max_len})")
    if pad_len == 0:
        return X
    return np.vstack([X, np.zeros((pad_len, X.shape[1]))])


def get_team_logit(model, scaler, team_df: pd.DataFrame) -> float:
    """Return the raw logit (pre-sigmoid) for a single team's partial-season sequence."""
    team_df = team_df.sort_values("gameweek")
    X = scaler.transform(team_df[FEATURES])
    X_padded = pad_sequence(X, MAX_GAMEWEEK)
    X_input = np.expand_dims(X_padded, axis=0)
    return float(model.predict(X_input, verbose=0)[0][0])


def predict_gameweek(
    model, scaler, season_df: pd.DataFrame, gameweek: int,
    temperature: float = 1.0,
) -> dict[str, float]:
    """
    Collect raw logits for all teams up to gameweek, then apply softmax across
    all 20 teams to produce a proper probability distribution (sums to 1.0).

    temperature < 1 sharpens the distribution (more confident);
    temperature > 1 flattens it (more uncertain). Default 1.0 = standard softmax.
    """
    gw_df = season_df[season_df["gameweek"] <= gameweek]
    teams = list(gw_df["teamName"].unique())

    logits = np.array([
        get_team_logit(model, scaler, gw_df[gw_df["teamName"] == t])
        for t in teams
    ])

    # Softmax across all 20 teams — the model now treats this as a
    # 20-way ranking problem, not 20 independent binary predictions.
    logits_scaled = logits / temperature
    exp_logits = np.exp(logits_scaled - logits_scaled.max())  # numerically stable
    probs = exp_logits / exp_logits.sum()

    return dict(zip(teams, probs))


def format_kalshi(probs: dict[str, float], gameweek: int) -> str:
    """Format a single gameweek's predictions in Kalshi-style output."""
    sorted_teams = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    lines = [f"\n=== Gameweek {gameweek} ==="]
    for team, prob in sorted_teams:
        lines.append(f"  {team:<22} {prob * 100:5.1f}%")
    return "\n".join(lines)


def prepare_current_season(all_history_df: pd.DataFrame) -> pd.DataFrame:
    """
    Download and prepare the current season (PREDICT_SEASON).
    ELO is computed relative to the previous season in all_history_df.
    """
    loader = FDCUKDataLoader()
    print(f"Downloading 2025-26 season data...")
    cur_df = loader.get_season(PREDICT_SEASON)
    if cur_df is None or cur_df.empty:
        raise RuntimeError("Could not download 2025-26 season data from football-data.co.uk")
    print(f"  {len(cur_df)} rows downloaded.")

    # Merge with history so ELO can look back at 2024-25 standings
    combined = pd.concat([all_history_df, cur_df], ignore_index=True)
    combined = convert_form(combined)
    combined = add_elo_feature(combined)
    combined["games_remaining"] = 38 - combined["gameweek"]
    combined["won_league"] = 0  # placeholder
    combined = add_relative_features(combined)

    return combined[combined["season"] == PREDICT_SEASON].copy()


def main():
    model = load_model("best_model.h5", compile=False)
    scaler = joblib.load("scaler.joblib")

    # Load historical data for ELO back-reference
    try:
        history_df = pd.read_csv("historical_standings.csv")
    except FileNotFoundError:
        print("historical_standings.csv not found — run build_dataset.py first.")
        return

    season_df = prepare_current_season(history_df)

    max_gw = int(season_df["gameweek"].max())
    print(f"\nPredicting gameweeks 1–{max_gw} for the 2025-26 Premier League season.\n")

    all_output = []
    for gw in range(1, max_gw + 1):
        # Only include teams that have played at least this many games
        teams_at_gw = season_df[season_df["gameweek"] == gw]["teamName"].unique()
        if len(teams_at_gw) == 0:
            continue

        probs = predict_gameweek(model, scaler, season_df, gw)
        all_output.append(format_kalshi(probs, gw))

    print("\n".join(all_output))

    # Also print the latest gameweek as a clean summary
    latest_probs = predict_gameweek(model, scaler, season_df, max_gw)
    print("\n" + "=" * 40)
    print(f"CURRENT STANDINGS (after GW {max_gw})")
    print("=" * 40)
    for team, prob in sorted(latest_probs.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(prob * 40)
        print(f"  {team:<22} {prob * 100:5.1f}%  {bar}")


if __name__ == "__main__":
    main()
