"""
Kalshi title-market comparison — the model against the prediction market, live.

Kalshi runs a Premier League "winner" market: one binary YES market per club, under
the event KXPREMIERLEAGUE-<yy> (e.g. KXPREMIERLEAGUE-27 for 2026-27). The public REST
API is free — no key, no per-call cost — so we can pull the whole season's daily price
history and line it up against the model's own per-gameweek probabilities.

Two honesty choices:

* **Normalisation.** The 20 clubs are separate YES markets, so their prices do NOT sum
  to 100% (there is a small over/under-round — ~97% early in 2026-27). Before comparing
  to the model's softmax, each day's prices are divided by their sum so both sides are
  proper distributions over the 20 teams.
* **Alignment.** The model emits a probability per *gameweek*; Kalshi trades by *date*.
  A gameweek is matched to the date by which every club has played that many games (the
  end of the round), and Kalshi is sampled at the last quote on or before that date — so
  each point compares like with like, using only information both sides had by then.

The settled-season archive lives behind a different endpoint that does not serve a price
series, so this comparison is for the *in-progress* season only.
"""

import datetime as dt
import json
import time
import urllib.request

import pandas as pd

from .config import current_season_start_year
from .data_acquisition import FDCUKDataLoader
from .paths import KALSHI_FILE, ensure_dirs

KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"
SERIES = "KXPREMIERLEAGUE"

# Kalshi market code (ticker suffix) -> football-data.co.uk team name.
CODE_TO_FDCUK = {
    "ARS": "Arsenal", "AVL": "Aston Villa", "BOU": "Bournemouth", "BRE": "Brentford",
    "BRI": "Brighton", "CFC": "Chelsea", "COV": "Coventry", "CRY": "Crystal Palace",
    "EVE": "Everton", "FUL": "Fulham", "HUL": "Hull", "IPS": "Ipswich",
    "LEE": "Leeds", "LFC": "Liverpool", "MCI": "Man City", "MUN": "Man United",
    "NEW": "Newcastle", "NFO": "Nott'm Forest", "SUN": "Sunderland", "TOT": "Tottenham",
}


def event_ticker(start_year: int) -> str:
    """2026-27 season (start_year 2026) -> 'KXPREMIERLEAGUE-27'."""
    return f"{SERIES}-{(start_year + 1) % 100:02d}"


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=25) as r:
        return json.load(r)


def fetch_kalshi_daily(start_year: int | None = None) -> dict[str, dict[str, float]]:
    """
    {date_iso: {team: normalised_prob}} — the daily Kalshi title distribution for the
    season, each day renormalised to sum to 1. Empty dict if the market is unavailable.
    """
    start_year = start_year if start_year is not None else current_season_start_year()
    event = event_ticker(start_year)
    markets = _get(f"{KALSHI_API}/markets?event_ticker={event}&limit=60").get("markets", [])
    if not markets:
        return {}

    start = int(dt.datetime(start_year, 7, 1, tzinfo=dt.timezone.utc).timestamp())
    end = int(dt.datetime.now(dt.timezone.utc).timestamp())

    raw: dict[str, dict[str, float]] = {}  # date -> {team: yes_price}
    for mk in markets:
        code = mk["ticker"].split("-")[-1]
        team = CODE_TO_FDCUK.get(code, mk.get("yes_sub_title", code))
        url = (f"{KALSHI_API}/series/{SERIES}/markets/{mk['ticker']}/candlesticks"
               f"?start_ts={start}&end_ts={end}&period_interval=1440")
        try:
            candles = _get(url).get("candlesticks", [])
        except Exception as e:
            print(f"  Kalshi {mk['ticker']}: candlesticks failed ({e})")
            candles = []
        for x in candles:
            ts = x.get("end_period_ts")
            pr = x.get("price", {})
            v = pr.get("close_dollars") or pr.get("mean_dollars")
            if ts and v is not None:
                d = dt.datetime.fromtimestamp(ts, dt.timezone.utc).date().isoformat()
                raw.setdefault(d, {})[team] = float(v)
        time.sleep(0.12)  # be polite

    norm: dict[str, dict[str, float]] = {}
    for d, probs in raw.items():
        tot = sum(probs.values())
        if tot > 0:
            norm[d] = {t: v / tot for t, v in probs.items()}
    return norm


def gameweek_dates(start_year: int | None = None) -> dict[int, dt.date]:
    """
    {gameweek: date} where the date is when the LAST team completed that many games
    (i.e. the round is done for everyone) — the honest 'as of' point for that gameweek.
    """
    start_year = start_year if start_year is not None else current_season_start_year()
    matches = FDCUKDataLoader().download_season_matches(start_year)
    if matches is None or matches.empty:
        return {}
    matches = matches.copy()
    matches["Date"] = pd.to_datetime(
        matches["Date"], format="mixed", dayfirst=True, errors="coerce"
    )
    matches = matches.dropna(subset=["Date"]).sort_values("Date")

    played: dict[str, int] = {}
    reach: dict[tuple[str, int], dt.date] = {}
    for _, m in matches.iterrows():
        for t in (str(m["HomeTeam"]).strip(), str(m["AwayTeam"]).strip()):
            played[t] = played.get(t, 0) + 1
            reach[(t, played[t])] = m["Date"].date()
    if not played:
        return {}

    complete_n = min(played.values())  # every team has played at least this many
    return {
        n: max(reach[(t, n)] for t in played if (t, n) in reach)
        for n in range(1, complete_n + 1)
    }


def kalshi_by_gameweek(start_year: int | None = None) -> pd.DataFrame:
    """Long frame [gameweek, teamName, kalshi_prob], Kalshi sampled at each gameweek's date."""
    daily = fetch_kalshi_daily(start_year)
    gwd = gameweek_dates(start_year)
    cols = ["gameweek", "teamName", "kalshi_prob"]
    if not daily or not gwd:
        return pd.DataFrame(columns=cols)

    dates = sorted(daily)
    rows = []
    for gw, date in gwd.items():
        dstr = date.isoformat()
        pick = None
        for d in dates:
            if d <= dstr:
                pick = d
            else:
                break
        pick = pick or dates[0]  # gameweek before the market opened -> earliest quote
        for team, p in daily[pick].items():
            rows.append({"gameweek": gw, "teamName": team, "kalshi_prob": round(p, 6)})
    return pd.DataFrame(rows, columns=cols)


def build_kalshi_cache(start_year: int | None = None) -> pd.DataFrame:
    """Fetch and cache the aligned Kalshi series to data/kalshi_current.csv."""
    df = kalshi_by_gameweek(start_year)
    ensure_dirs()
    if not df.empty:
        df.to_csv(KALSHI_FILE, index=False)
        print(f"Kalshi comparison saved to {KALSHI_FILE} "
              f"({df['gameweek'].nunique()} gameweeks, {df['teamName'].nunique()} teams)")
    else:
        print("No Kalshi data fetched (market unavailable or season not started).")
    return df


def main():
    build_kalshi_cache()


if __name__ == "__main__":
    main()
