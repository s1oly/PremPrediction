"""
Squad market value scraper (Transfermarkt).

Total squad market value is the single strongest pre-season prior for where a club
finishes, and it captures what league tables cannot: transfer spending and the gap
in raw squad quality between the elite and everyone else. Prediction markets move on
exactly this information, so feeding it to the model makes pre-gameweek-1 odds far
more realistic.

Values are only reliably available from ~2005 onwards, so training is restricted to
those seasons (see MIN_VALUE_SEASON). The engineered feature is a *share* — a team's
value divided by the league total that season — which is era-robust: it cancels out
decades of transfer-fee inflation and encodes squad strength relative to peers.

Usage:
    python -m prem_prediction.squad_value                 # scrape all covered seasons
    python -m prem_prediction.squad_value --season 2026   # one season
"""

import argparse
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

from .config import current_season_start_year
from .paths import SQUAD_VALUE_FILE, ensure_dirs

MIN_VALUE_SEASON = 2005  # earliest season with reliable Transfermarkt data

_BASE = "https://www.transfermarkt.com/premier-league/startseite/wettbewerb/GB1/plus/?saison_id={year}"
_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"),
    "Accept-Language": "en-US,en;q=0.9",
}

# Transfermarkt club name -> football-data.co.uk convention. Only the ones that
# differ after the generic suffix-stripping in _normalise() need listing here.
_NAME_MAP = {
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Tottenham Hotspur": "Tottenham",
    "Newcastle United": "Newcastle",
    "Brighton & Hove Albion": "Brighton",
    "AFC Bournemouth": "Bournemouth",
    "Nottingham Forest": "Nott'm Forest",
    "Wolverhampton Wanderers": "Wolves",
    "West Ham United": "West Ham",
    "West Bromwich Albion": "West Brom",
    "Queens Park Rangers": "QPR",
    "Leicester City": "Leicester",
    "Ipswich Town": "Ipswich",
    "Norwich City": "Norwich",
    "Hull City": "Hull",
    "Cardiff City": "Cardiff",
    "Stoke City": "Stoke",
    "Swansea City": "Swansea",
    "Blackburn Rovers": "Blackburn",
    "Bolton Wanderers": "Bolton",
    "Wigan Athletic": "Wigan",
    "Birmingham City": "Birmingham",
    "Leeds United": "Leeds",
    "Huddersfield Town": "Huddersfield",
    "Charlton Athletic": "Charlton",
    "Derby County": "Derby",
    "Luton Town": "Luton",
    "Coventry City": "Coventry",
    "Sheffield United": "Sheffield United",
    "Sheffield Wednesday": "Sheffield Weds",
    "Bradford City": "Bradford",
    "Wimbledon FC": "Wimbledon",
    "Oldham Athletic": "Oldham",
    "Barnsley FC": "Barnsley",
}

_STRIP_SUFFIXES = (" FC", " AFC")


def _normalise(tm_name: str) -> str:
    """Map a Transfermarkt club name to the football-data.co.uk convention."""
    name = tm_name.strip()
    if name in _NAME_MAP:
        return _NAME_MAP[name]
    for suf in _STRIP_SUFFIXES:
        if name.endswith(suf):
            name = name[: -len(suf)].strip()
    if name.startswith("AFC "):
        name = name[4:].strip()
    return _NAME_MAP.get(name, name)


def _parse_value(text: str) -> float | None:
    """Parse a Transfermarkt value string like '€1.36bn' / '€427.90m' to euros."""
    s = text.replace("€", "").strip()
    mult = 1.0
    if s.endswith("bn"):
        mult, s = 1e9, s[:-2]
    elif s.endswith("m"):
        mult, s = 1e6, s[:-1]
    elif s.endswith("k"):
        mult, s = 1e3, s[:-1]
    try:
        return float(s.replace(",", "")) * mult
    except ValueError:
        return None


def fetch_season_values(year: int, session: requests.Session | None = None) -> dict[str, float]:
    """Return {fdcuk_team_name: total_squad_value_eur} for a season, or {} on failure."""
    sess = session or requests.Session()
    try:
        resp = sess.get(_BASE.format(year=year), headers=_HEADERS, timeout=25)
        resp.raise_for_status()
    except Exception as e:
        print(f"  Season {year}: fetch failed ({e})")
        return {}

    soup = BeautifulSoup(resp.text, "lxml")
    table = soup.select_one("table.items")
    if table is None:
        print(f"  Season {year}: no value table found")
        return {}

    values = {}
    for tr in table.select("tbody > tr"):
        a = tr.select_one("td.hauptlink a[href*='/verein/']")
        if not a:
            continue
        name = _normalise(a.get("title") or a.text)
        cells = tr.select("td.rechts")
        val = _parse_value(cells[-1].text) if cells else None
        if val:
            values[name] = val
    return values


def build_squad_values(start: int = MIN_VALUE_SEASON, end: int | None = None) -> pd.DataFrame:
    """Scrape every season in [start, end] and write data/squad_values.csv."""
    ensure_dirs()
    end = end if end is not None else current_season_start_year()
    session = requests.Session()
    rows = []
    for year in range(start, end + 1):
        print(f"  Transfermarkt {year}-{(year + 1) % 100:02d}...", end=" ", flush=True)
        vals = fetch_season_values(year, session)
        print(f"{len(vals)} clubs")
        for team, val in vals.items():
            rows.append({"season": year, "teamName": team, "value_eur": val})
        time.sleep(2.0)  # be polite

    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(SQUAD_VALUE_FILE, index=False)
        print(f"\nSaved {len(df)} rows across {df['season'].nunique()} seasons to {SQUAD_VALUE_FILE}")
    else:
        print("No squad values scraped.")
    return df


def main():
    ap = argparse.ArgumentParser(description="Scrape Transfermarkt squad values.")
    ap.add_argument("--season", type=int, help="Scrape a single season (updates the CSV in place).")
    ap.add_argument("--start", type=int, default=MIN_VALUE_SEASON)
    ap.add_argument("--end", type=int, default=None)
    args = ap.parse_args()

    if args.season is not None:
        vals = fetch_season_values(args.season)
        print(f"{args.season}: {len(vals)} clubs")
        new = pd.DataFrame(
            [{"season": args.season, "teamName": t, "value_eur": v} for t, v in vals.items()]
        )
        if SQUAD_VALUE_FILE.exists():
            existing = pd.read_csv(SQUAD_VALUE_FILE)
            existing = existing[existing["season"] != args.season]
            new = pd.concat([existing, new], ignore_index=True)
        new.sort_values(["season", "teamName"]).to_csv(SQUAD_VALUE_FILE, index=False)
        print(f"Updated {SQUAD_VALUE_FILE}")
    else:
        build_squad_values(args.start, args.end)


if __name__ == "__main__":
    main()
