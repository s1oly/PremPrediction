import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from .paths import MODEL_FILE, SCALER_FILE, TRAINING_FILE, ensure_dirs

FEATURES = [
    "position", "form", "won", "draw", "lost",
    "points", "goalsFor", "goalsAgainst", "goalDifference",
    "games_remaining", "elo", "prev_position",
    "points_per_game", "points_gap_from_leader", "max_obtainable_points",
]
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

    Returns X of shape (N, max_gw, n_features) and y of shape (N,).
    """
    X_list, y_list = [], []
    n_features = len(features)

    for (_, __), team_df in df.groupby(["season", "teamName"]):
        team_df = team_df.sort_values("gameweek").reset_index(drop=True)
        if len(team_df) < max_gw:
            continue
        label = int(team_df["won_league"].iloc[0])
        full = team_df[features].values[:max_gw]

        for cutoff in PARTIAL_CUTOFFS:
            partial = full[:cutoff]
            pad = np.zeros((max_gw - cutoff, n_features))
            X_list.append(np.vstack([partial, pad]))
            y_list.append(label)

    return np.array(X_list), np.array(y_list)


def build_model(timesteps: int, n_features: int) -> Sequential:
    """The two-layer masked LSTM with a raw-logit head (softmax applied at predict time)."""
    model = Sequential([
        Masking(mask_value=0.0, input_shape=(timesteps, n_features)),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(1),  # raw logit — softmax across the 20 teams happens at prediction time
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=BinaryCrossentropy(from_logits=True),  # numerically stable sigmoid+BCE
        metrics=["accuracy"],
    )
    return model


def main():
    ensure_dirs()
    df = pd.read_csv(TRAINING_FILE)

    scaler = StandardScaler()
    df[FEATURES] = scaler.fit_transform(df[FEATURES])
    joblib.dump(scaler, SCALER_FILE)

    X, y = build_sequences(df, FEATURES)
    print(f"Sequences: {X.shape}, positive labels: {y.sum()}/{len(y)}")

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
        verbose=2,  # one line per epoch: loss / val_loss / accuracy (easy to watch)
        callbacks=callbacks,
    )
    print(f"Training complete. Model -> {MODEL_FILE}, scaler -> {SCALER_FILE}")


if __name__ == "__main__":
    main()
