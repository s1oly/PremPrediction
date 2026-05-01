import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, LSTM, Masking
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

FEATURES = [
    "position", "form", "won", "draw", "lost",
    "points", "goalsFor", "goalsAgainst", "goalDifference",
    "games_remaining", "elo",
    "points_per_game", "points_gap_from_leader", "max_obtainable_points",
]
MAX_GAMEWEEK = 38

# Gameweek cutoffs at which partial sequences are generated during training.
# The model sees every team at GW5, GW10, ... GW38, teaching it to make
# confident early-season predictions, not just end-of-season ones.
PARTIAL_CUTOFFS = [5, 10, 15, 20, 25, 30, 35, 38]


def from_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def build_sequences(df: pd.DataFrame, features: list[str], max_gw: int = MAX_GAMEWEEK):
    """
    Build training sequences for each (season, team) at multiple gameweek
    cutoffs.  For each cutoff the real rows are kept and the remainder is
    zero-padded so the Masking layer knows to ignore them.

    This teaches the model to discriminate teams at every stage of the season,
    not just at the final whistle.

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


def main():
    df = from_csv("out.csv")

    # Fit scaler on training features only
    scaler = StandardScaler()
    df[FEATURES] = scaler.fit_transform(df[FEATURES])
    joblib.dump(scaler, "scaler.joblib")

    X, y = build_sequences(df, FEATURES)
    print(f"Sequences: {X.shape}, positive labels: {y.sum()}/{len(y)}")

    model = Sequential([
        Masking(mask_value=0.0, input_shape=(X.shape[1], X.shape[2])),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        # Raw logit output — no sigmoid here.
        # Softmax is applied ACROSS all 20 teams at prediction time,
        # so the model learns a relative ranking, not 20 independent probabilities.
        Dense(1),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        # from_logits=True: keras applies sigmoid internally (numerically stable,
        # mathematically identical to sigmoid + standard BCE).
        loss=BinaryCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.summary()

    callbacks = [
        ModelCheckpoint("best_model.h5", save_best_only=True, monitor="val_loss"),
        EarlyStopping(patience=8, restore_best_weights=True, monitor="val_loss"),
    ]

    model.fit(
        X, y,
        validation_split=0.15,
        epochs=60,
        batch_size=32,
        verbose=1,
        callbacks=callbacks,
    )
    print("Training complete. Model saved to best_model.h5, scaler to scaler.joblib")


if __name__ == "__main__":
    main()
