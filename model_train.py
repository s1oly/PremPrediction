import requests
import pandas as pd
import numpy as np
import csv
import time
from datetime import datetime
import ast
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Masking, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import joblib


def from_csv(csv = None):
    df = pd.read_csv(csv)
    return df

def build_sequence(df, features, max_gameweek = 38):
    X_seq = []
    y_seq = []

    grouped = df.groupby(["season", "teamID"])
    for (_,_), team_df in grouped:
        team_df = team_df.sort_values("gameweek")
        if len(team_df) < max_gameweek:
            continue
        X_seq.append(team_df[features].values)
        y_seq.append(team_df["won_league"].iloc[0])
    return np.array(X_seq), np.array(y_seq)

def predict_team_probablity(model, team_df, features):
    team_df = team_df.sort_values("gameweek")
    X_partial = team_df[features].values
    X_partial = np.expand_dims(X_partial, axis = 0)
    return model.predict(X_partial)[0][0]


def predict_league_probs(df, model, season, gameweek, features):
    probs = []
    for team_id in df[df["season"] == season][team_id].unique():
        team_df = df[(df["season" == season]) & (df["teamID" == team_id]) & (df["gameweek" <= gameweek])]
        prob = predict_team_probablity(model, team_df, features)
        probs.append()

def main():
    FEATURES = ["position", "form", "won", "draw", "lost", "points", "goalsFor", "goalsAgainst", "goalDifference", "games_remaining"]
    df = from_csv("out.csv")
    X = df[FEATURES]
    Y = df["won_league"]

    scalar = StandardScaler()
    X_scaled = scalar.fit_transform(X)
    df[FEATURES] = X_scaled

    X_seq, Y_seq = build_sequence(df, FEATURES)

    model = Sequential([
        LSTM(64, return_sequences = True, input_shape = (X_seq.shape[1], X_seq.shape[2])),
        Masking(mask_value = 0.0),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(1, activation = "sigmoid")
    ])

    model.compile(
        optimizer = Adam(learning_rate = 0.001),
        loss = "binary_crossentropy",
        metrics = ["accuracy"]
    )

    checkpoint = ModelCheckpoint("best_model.h5", save_best_only = True, monitor = "val_loss")

    history = model.fit(X_seq, Y_seq, validation_split = 0.2, epochs = 30, batch_size = 16, verbose = 1, callbacks = [checkpoint])

    joblib.dump(scalar, "scalar.joblib")

  
if __name__ == '__main__':
    main()

