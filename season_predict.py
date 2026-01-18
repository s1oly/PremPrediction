import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib


import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib

MAX_WEEK = 38

# ----------------- Functions -----------------
def from_csv(csv_file):
    return pd.read_csv(csv_file)

def pad_sequence(X, max_len=38):
    pad_len = max_len - X.shape[0]
    if pad_len < 0:
        raise ValueError("Too many gameweeks")
    return np.vstack([X, np.zeros((pad_len, X.shape[1]))])

def predict_team_probability(model, scaler, team_df, features):
    # Scale features
    team_df = team_df.sort_values("gameweek")
    X = scaler.transform(team_df[features])
    
    X_padded = pad_sequence(X, MAX_WEEK)
    X_padded = np.expand_dims(X_padded, axis=0) # (1, timesteps, features)
    
    prob = model.predict(X_padded, verbose=0)[0][0]
    return prob

def predict_league_probs(df, model, scaler, season, gameweek, features):
    probs = {}
    season_df = df[df["season"] == season]
    for team_id in season_df["teamID"].unique():
        team_df = season_df[
            (season_df["teamID"] == team_id) & (season_df["gameweek"] <= gameweek)
        ]
        prob = predict_team_probability(model, scaler, team_df, features)
        probs[team_id] = prob
    return probs

def main():
    FEATURES = [
        "position", "form", "won", "draw", "lost", 
        "points", "goalsFor", "goalsAgainst", 
        "goalDifference", "games_remaining"
    ]

    # Load model and scaler
    model = load_model("best_model.h5", compile = False)
    scaler = joblib.load("scalar.joblib")

    # Load new data
    predict_df = from_csv("test.csv")

    print("Columns:", predict_df.columns.tolist())
    print("Unique seasons:", predict_df["season"].unique())
    print("Rows total:", len(predict_df))


    # Predict probabilities for all teams up to a certain gameweek
    season = 2025
    gameweek = 22
    league_probs = predict_league_probs(predict_df, model, scaler, season, gameweek, FEATURES)




    # Print results
    for team_id, prob in league_probs.items():
        print(f"Team {team_id}: Probability of winning league = {prob:.4f}")

if __name__ == "__main__":
    main()
