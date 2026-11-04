import json

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.losses import Huber
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from .paths import MODEL_FILE, RESIDUAL_STD_FILE, SCALER_FILE, TRAINING_FILE, ensure_dirs

FEATURES = [
    "position", "form", "won", "draw", "lost",
    "points", "goalsFor", "goalsAgainst", "goalDifference",
    "games_remaining", "elo", "prev_position",
    "points_per_game", "points_gap_from_leader", "max_obtainable_points",
    "shot_att", "shot_def",  # chances created / conceded (match_stats.py)
    "sos_played", "sos_remaining",  # strength of schedule faced / still to come (schedule_strength.py)
    "squad_value_z", "trophy_score",  # pre-season priors, learned directly (prepare_model_data.py)
]
# Regression target: a team's final-season points. Title probability is derived
# from predicted final points by simulation at prediction time (season_predict).
TARGET = "final_points"
MAX_GAMEWEEK = 38

# Gameweek cutoffs at which partial sequences are generated during training, so
# the model learns to make confident predictions early, not just at GW38.
# The early cutoffs (1-3) teach it to lean on the pre-season priors (prev_position,
# elo) before results accumulate — without them a GW1 sequence is out-of-distribution
# and the model falls back to a near-uniform guess.
PARTIAL_CUTOFFS = [1, 2, 3, 5, 10, 15, 20, 25, 30, 35, 38]


def build_sequences(df: pd.DataFrame, features: list[str], max_gw: int = MAX_GAMEWEEK):
    """
    Build training sequences for each (season, team) at multiple gameweek cutoffs.
    For each cutoff the real rows are kept and the remainder is zero-padded so the
    Masking layer ignores them.

    Returns X (N, max_gw, n_features), y (N,) final-points targets, and cutoffs (N,)
    the gameweek each sequence was truncated at (for per-gameweek error calibration).
    """
    X_list, y_list, cut_list = [], [], []
    n_features = len(features)

    for (_, __), team_df in df.groupby(["season", "teamName"]):
        team_df = team_df.sort_values("gameweek").reset_index(drop=True)
        if len(team_df) < max_gw:
            continue
        target = float(team_df[TARGET].iloc[0])
        full = team_df[features].values[:max_gw]

        for cutoff in PARTIAL_CUTOFFS:
            partial = full[:cutoff]
            pad = np.zeros((max_gw - cutoff, n_features))
            X_list.append(np.vstack([partial, pad]))
            y_list.append(target)
            cut_list.append(cutoff)

    return np.array(X_list), np.array(y_list), np.array(cut_list)


def build_model(timesteps: int, n_features: int) -> Sequential:
    """Two-layer masked LSTM with a linear head that predicts final-season points."""
    model = Sequential([
        Masking(mask_value=0.0, input_shape=(timesteps, n_features)),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(1),  # linear: predicted final points
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=Huber(delta=5.0),  # robust regression on points (less swayed by outliers)
        metrics=["mae"],
    )
    return model


def calibrate_residual_std(model, X, y, cutoffs) -> dict:
    """
    Std of the final-points prediction error at each gameweek cutoff. Prediction turns
    point forecasts into title probabilities by sampling N(pred, std[gameweek]), so this
    is the model's own honest uncertainty — wide early in the season, tight near the end.
    """
    pred = model.predict(X, verbose=0).ravel()
    resid = y - pred
    out = {}
    for c in sorted(set(int(x) for x in cutoffs)):
        r = resid[cutoffs == c]
        out[str(c)] = float(np.std(r)) if len(r) > 1 else float(np.std(resid))
    out["overall"] = float(np.std(resid))
    return out


def main():
    ensure_dirs()
    df = pd.read_csv(TRAINING_FILE)

    scaler = StandardScaler()
    df[FEATURES] = scaler.fit_transform(df[FEATURES])
    joblib.dump(scaler, SCALER_FILE)

    X, y, cutoffs = build_sequences(df, FEATURES)
    print(f"Sequences: {X.shape}, target final_points range: {y.min():.0f}-{y.max():.0f}")

    model = build_model(X.shape[1], X.shape[2])
    model.summary()

    callbacks = [
        ModelCheckpoint(str(MODEL_FILE), save_best_only=True, monitor="val_loss"),
        EarlyStopping(patience=8, restore_best_weights=True, monitor="val_loss"),
    ]

    model.fit(
        X, y,
        validation_split=0.15,
        epochs=60,
        batch_size=32,
        verbose=2,  # one line per epoch: loss / val_loss / mae (easy to watch)
        callbacks=callbacks,
    )

    resid_std = calibrate_residual_std(model, X, y, cutoffs)
    with open(RESIDUAL_STD_FILE, "w") as f:
        json.dump(resid_std, f, indent=2)
    print(f"Residual std by gameweek -> {RESIDUAL_STD_FILE} "
          f"(GW1 ±{resid_std.get('1', 0):.1f} pts, GW35 ±{resid_std.get('35', 0):.1f} pts)")
    print(f"Training complete. Model -> {MODEL_FILE}, scaler -> {SCALER_FILE}")


if __name__ == "__main__":
    main()
