"""
Honest leave-one-season-out style evaluation.

`val_loss` during training is per-team sigmoid BCE on a leaky random split — it is
not the thing we ship. We ship a softmax *ranking* of the 20 teams per gameweek, so
this module scores exactly that, on held-out whole seasons (grouped k-fold, no
sequence leaks across the split):

  * champ_logloss — −log(probability the model gave the eventual champion). Lower is
                    better; the uniform baseline is ln(20) ≈ 3.00.
  * champ_prob    — the probability mass the model put on the actual champion.
  * top1          — fraction of held-out seasons where the champion was ranked #1.
  * leader_top1   — same, for the trivial "current league leader wins" baseline.
                    If the model can't beat this, it isn't earning its keep.
  * brier         — multiclass Brier score of the title distribution.

Each metric is reported per evaluation gameweek, so you can see how sharp the model
is pre-season (GW1), at the winter break (~GW19), and late (GW35).

Usage:
    python -m prem_prediction.evaluate            # 5-fold, default gameweeks
    python -m prem_prediction.evaluate --folds 8
"""

import argparse
import json
import math

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping

from .model_train import FEATURES, MAX_GAMEWEEK, build_model, build_sequences
from .paths import OUTPUTS_DIR, TRAINING_FILE, ensure_dirs

EVAL_GAMEWEEKS = [1, 5, 10, 15, 19, 25, 30, 35, 38]


def _pad(X: np.ndarray) -> np.ndarray:
    if X.shape[0] >= MAX_GAMEWEEK:
        return X[:MAX_GAMEWEEK]
    return np.vstack([X, np.zeros((MAX_GAMEWEEK - X.shape[0], X.shape[1]))])


def softmax_title_probs(model, scaler, season_df: pd.DataFrame, gameweek: int) -> dict[str, float]:
    """Softmax title probabilities across all teams at `gameweek` (one batched predict)."""
    gw_df = season_df[season_df["gameweek"] <= gameweek]
    teams = list(gw_df["teamName"].unique())
    seqs = []
    for t in teams:
        tdf = gw_df[gw_df["teamName"] == t].sort_values("gameweek")
        seqs.append(_pad(scaler.transform(tdf[FEATURES])))
    logits = model.predict(np.stack(seqs), verbose=0).ravel()
    exp = np.exp(logits - logits.max())
    probs = exp / exp.sum()
    return dict(zip(teams, probs))


def _season_folds(seasons: list[int], k: int, seed: int) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    shuffled = list(seasons)
    rng.shuffle(shuffled)
    return [sorted(shuffled[i::k]) for i in range(k)]


def run_cv(k: int = 5, eval_gws=EVAL_GAMEWEEKS, epochs: int = 40, seed: int = 42) -> pd.DataFrame:
    df = pd.read_csv(TRAINING_FILE)
    seasons = [
        s for s in sorted(df["season"].unique())
        if (df[(df["season"] == s) & (df["won_league"] == 1)].shape[0] > 0)
    ]
    folds = _season_folds(seasons, k, seed)
    print(f"Evaluating {len(seasons)} seasons in {k} grouped folds "
          f"({FEATURES.count('elo')} elo feature incl. squad value)...")

    records = []
    for fi, test_seasons in enumerate(folds, 1):
        train_df = df[~df["season"].isin(test_seasons)].copy()
        scaler = StandardScaler()
        train_df[FEATURES] = scaler.fit_transform(train_df[FEATURES])
        X, y = build_sequences(train_df, FEATURES)
        model = build_model(X.shape[1], X.shape[2])
        model.fit(
            X, y, validation_split=0.15, epochs=epochs, batch_size=32, verbose=0,
            callbacks=[EarlyStopping(patience=6, restore_best_weights=True, monitor="val_loss")],
        )

        for s in test_seasons:
            sdf = df[df["season"] == s]  # unscaled — softmax_title_probs scales internally
            champ = sdf[sdf["won_league"] == 1]["teamName"].iloc[0]
            for gw in eval_gws:
                gw_rows = sdf[sdf["gameweek"] == gw]
                if gw_rows.empty:
                    continue
                probs = softmax_title_probs(model, scaler, sdf, gw)
                p_ch = probs.get(champ, 1e-12)
                ranked = sorted(probs, key=probs.get, reverse=True)
                leader = gw_rows.sort_values("position")["teamName"].iloc[0]
                records.append({
                    "season": s, "gw": gw, "champion": champ,
                    "champ_prob": p_ch,
                    "logloss": -math.log(max(p_ch, 1e-12)),
                    "brier": sum((p - (1.0 if t == champ else 0.0)) ** 2 for t, p in probs.items()),
                    "top1": int(ranked[0] == champ),
                    "leader_top1": int(leader == champ),
                })
        print(f"  fold {fi}/{k}: held out {test_seasons}")

    return pd.DataFrame(records)


def summarise(records: pd.DataFrame) -> pd.DataFrame:
    agg = records.groupby("gw").agg(
        seasons=("season", "nunique"),
        model_top1=("top1", "mean"),
        leader_top1=("leader_top1", "mean"),
        champ_logloss=("logloss", "mean"),
        champ_prob=("champ_prob", "mean"),
        brier=("brier", "mean"),
    ).reset_index()
    return agg


def main():
    ap = argparse.ArgumentParser(description="Grouped season CV of the title-ranking model.")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    records = run_cv(k=args.folds, epochs=args.epochs, seed=args.seed)
    agg = summarise(records)

    uniform = math.log(20)
    print("\n" + "=" * 78)
    print("HELD-OUT TITLE-RANKING METRICS  (uniform-guess logloss = %.2f)" % uniform)
    print("=" * 78)
    print(f"{'GW':>3} {'seasons':>7} {'model_top1':>11} {'leader_top1':>12} "
          f"{'champ_LL':>9} {'champ_prob':>11} {'brier':>7}")
    for _, r in agg.iterrows():
        print(f"{int(r.gw):>3} {int(r.seasons):>7} {r.model_top1:>11.2f} {r.leader_top1:>12.2f} "
              f"{r.champ_logloss:>9.3f} {r.champ_prob*100:>10.1f}% {r.brier:>7.3f}")

    overall = {
        "model_top1_mean": float(records["top1"].mean()),
        "leader_top1_mean": float(records["leader_top1"].mean()),
        "champ_logloss_mean": float(records["logloss"].mean()),
        "uniform_logloss": uniform,
    }
    print("\nOverall (all gameweeks): model top-1 %.2f vs leader %.2f | mean champ logloss %.3f"
          % (overall["model_top1_mean"], overall["leader_top1_mean"], overall["champ_logloss_mean"]))

    ensure_dirs()
    out = OUTPUTS_DIR / "evaluation.json"
    with open(out, "w") as f:
        json.dump({"per_gameweek": agg.to_dict("records"), "overall": overall}, f, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
