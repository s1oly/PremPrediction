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

# The final-points forecast used for the title simulation is a blend of the LSTM
# prediction and a simple current-pace projection (points + ppg * games_remaining).
# The blend is gameweek-dependent: a current-pace projection is meaningless after 1
# game but authoritative after 25, whereas the LSTM's value is its pre-season priors
# early on. So the pace weight RAMPS UP with games played — pace_w = gw / (gw + TAU) —
# keeping the LSTM (priors) in charge at GW1-4 and handing over to pace by mid-season.
# This fixed both failure modes held-out CV exposed: the LSTM's mid-season hot-streak
# overshoot (GW15-30) and a fixed blend's GW1 nonsense. TAU=6 centres the handover
# around GW6 and matches the ~0.4 average LSTM weight the fixed-weight sweep preferred.
BLEND_TAU = 9.0

# Regression-to-the-mean on the pace projection. A team's points-per-game is shrunk
# toward the league average with PACE_PRIOR_GAMES pseudo-games, so a hot 5-game run is
# not extrapolated linearly across the whole season. This is what football actually does
# (extreme early paces regress), and it damps the week-to-week swings the raw projection
# caused — strong shrinkage early (few real games), fading as the sample grows.
PACE_PRIOR_GAMES = 12.0
LEAGUE_MEAN_PPG = 1.37  # ~52 pts over 38 games, the long-run league average


def blend_final_points(lstm_pred, points, gameweek, max_obtainable):
    """
    Blend the LSTM final-points forecast with a regressed current-pace projection,
    weighting the pace term by gw/(gw+TAU) so it earns trust as games accumulate.
    Accepts scalars or numpy arrays. `gameweek` is games played; games_remaining and
    points-per-game come from the current table, so the anchor is leak-free.
    """
    import numpy as _np
    points = _np.asarray(points, dtype=float)
    gw = _np.maximum(_np.asarray(gameweek, dtype=float), 1.0)
    games_remaining = _np.maximum((_np.asarray(max_obtainable, dtype=float) - points) / 3.0, 0.0)
    # Shrink points-per-game toward the league mean (regression to the mean).
    ppg = (points + PACE_PRIOR_GAMES * LEAGUE_MEAN_PPG) / (gw + PACE_PRIOR_GAMES)
    pace_proj = points + ppg * games_remaining
    pace_w = gw / (gw + BLEND_TAU)
    return (1.0 - pace_w) * _np.asarray(lstm_pred, dtype=float) + pace_w * pace_proj

# Gameweek cutoffs at which partial sequences are generated during training, so
# the model learns to make confident predictions early, not just at GW38.
# The early cutoffs (1-3) teach it to lean on the pre-season priors (prev_position,
# elo) before results accumulate — without them a GW1 sequence is out-of-distribution
# and the model falls back to a near-uniform guess.
PARTIAL_CUTOFFS = [1, 2, 3, 5, 10, 15, 20, 25, 30, 35, 38]


def build_sequences(df: pd.DataFrame, features: list[str], max_gw: int = MAX_GAMEWEEK,
                    anchor_df: pd.DataFrame | None = None):
    """
    Build training sequences for each (season, team) at multiple gameweek cutoffs.
    For each cutoff the real rows are kept and the remainder is zero-padded so the
    Masking layer ignores them.

    Returns X (N, max_gw, n_features), y (N,) final-points targets, cutoffs (N,) the
    gameweek each sequence was truncated at, and anchors (N, 2) the UNSCALED
    [points, max_obtainable_points] at that cutoff (for the pace-blend calibration).
    Pass `anchor_df` (the same rows, unscaled) when `df`'s feature columns are scaled.
    """
    X_list, y_list, cut_list, anc_list = [], [], [], []
    n_features = len(features)
    anchor_df = anchor_df if anchor_df is not None else df
    anchor_groups = {k: g.sort_values("gameweek").reset_index(drop=True)
                     for k, g in anchor_df.groupby(["season", "teamName"])}

    for key, team_df in df.groupby(["season", "teamName"]):
        team_df = team_df.sort_values("gameweek").reset_index(drop=True)
        if len(team_df) < max_gw:
            continue
        target = float(team_df[TARGET].iloc[0])
        full = team_df[features].values[:max_gw]
        anc = anchor_groups[key][["points", "max_obtainable_points"]].values[:max_gw]

        for cutoff in PARTIAL_CUTOFFS:
            partial = full[:cutoff]
            pad = np.zeros((max_gw - cutoff, n_features))
            X_list.append(np.vstack([partial, pad]))
            y_list.append(target)
            cut_list.append(cutoff)
            anc_list.append(anc[cutoff - 1])  # points / max obtainable after `cutoff` games

    return np.array(X_list), np.array(y_list), np.array(cut_list), np.array(anc_list)


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


def calibrate_residual_std(model, X, y, cutoffs, anchors) -> dict:
    """
    Std of the BLENDED final-points forecast's error at each gameweek cutoff. The
    simulation samples N(blend, std[gameweek]); calibrating on the blended forecast's
    own error (not the raw LSTM's) is what makes it honest — early-season pace
    extrapolation carries large error, so std is wide at GW5 and tight near GW38.
    """
    lstm = model.predict(X, verbose=0).ravel()
    pred = blend_final_points(lstm, anchors[:, 0], cutoffs, anchors[:, 1])
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

    scaler = StandardScaler().fit(df[FEATURES])
    joblib.dump(scaler, SCALER_FILE)
    df_scaled = df.copy()
    df_scaled[FEATURES] = scaler.transform(df[FEATURES])

    X, y, cutoffs, anchors = build_sequences(df_scaled, FEATURES, anchor_df=df)
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

    resid_std = calibrate_residual_std(model, X, y, cutoffs, anchors)
    with open(RESIDUAL_STD_FILE, "w") as f:
        json.dump(resid_std, f, indent=2)
    print(f"Residual std by gameweek -> {RESIDUAL_STD_FILE} "
          f"(GW1 ±{resid_std.get('1', 0):.1f} pts, GW35 ±{resid_std.get('35', 0):.1f} pts)")
    print(f"Training complete. Model -> {MODEL_FILE}, scaler -> {SCALER_FILE}")


if __name__ == "__main__":
    main()
