import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, LSTM, Masking
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

FEATURES = [
    "position", "form", "won", "draw", "lost",
    "points", "goalsFor", "goalsAgainst", "goalDifference",
    "games_remaining", "elo",
]
MAX_GAMEWEEK = 38


def from_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def build_sequences(df: pd.DataFrame, features: list[str], max_gw: int = MAX_GAMEWEEK):
    """
    Build fixed-length (max_gw × n_features) sequences for each (season, team).
    Only teams with exactly max_gw rows are included (complete seasons).
    Returns X of shape (N, max_gw, n_features) and y of shape (N,).
    """
    X_list, y_list = [], []

    for (_, __), team_df in df.groupby(["season", "teamName"]):
        team_df = team_df.sort_values("gameweek")
        if len(team_df) < max_gw:
            continue
        X_list.append(team_df[features].values[:max_gw])
        y_list.append(int(team_df["won_league"].iloc[0]))

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
        Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
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
