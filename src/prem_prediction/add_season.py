"""
Add a newly completed Premier League season to the training dataset and update
trophy_data.py with that season's trophy winners.

Usage (interactive):
    python -m prem_prediction.add_season --season 2025

Usage (non-interactive):
    python -m prem_prediction.add_season --season 2025 \\
        --pl Arsenal --fa "Man City" --lc "Man City" \\
        --cl None --el "Aston Villa" --ecl "Crystal Palace"

After running, retrain:  python -m prem_prediction.model_train
"""

import argparse
import re
import sys
import textwrap
from pathlib import Path

import pandas as pd

from .config import season_label
from .data_acquisition import FDCUKDataLoader
from .paths import MATCHES_FILE, STANDINGS_FILE, TRAINING_FILE, ensure_dirs
from .prepare_model_data import prepare

TROPHY_FILE = Path(__file__).resolve().parent / "trophy_data.py"


def _ask(prompt: str, allow_none: bool = True) -> str | None:
    raw = input(prompt).strip()
    if allow_none and raw.lower() in ("none", "n/a", ""):
        return None
    return raw


def _patch_trophy_dict(source: str, dict_name: str, season: int, value: str | None) -> str:
    """Insert `season: "value",` (or `season: None,`) into the named dict in source."""
    if value is None:
        entry = f"    {season}: None,              # {season}-{(season + 1) % 100:02d}\n"
    else:
        pad = " " * max(1, 18 - len(value))
        entry = f'    {season}: "{value}",{pad}# {season}-{(season + 1) % 100:02d}\n'

    pattern = re.compile(rf"({re.escape(dict_name)}\s*=\s*\{{.*?\}})", re.DOTALL)
    match = pattern.search(source)
    if not match:
        print(f"  WARNING: could not find {dict_name} in trophy_data.py — skipping.")
        return source

    dict_text = match.group(1)
    if f"    {season}:" in dict_text:
        print(f"  {dict_name} already has season {season} — skipping.")
        return source
    updated = dict_text.rstrip().rstrip("}") + entry + "}"
    return source[:match.start()] + updated + source[match.end():]


def update_trophy_file(season, pl, fa, lc, cl, el, ecl) -> None:
    source = TROPHY_FILE.read_text()
    updates = [("PL_CHAMPIONS", pl), ("FA_CUP_WINNERS", fa), ("LEAGUE_CUP_WINNERS", lc)]
    if cl:
        updates.append(("CL_WINNERS_ENGLISH", cl))
    if el:
        updates.append(("EL_WINNERS_ENGLISH", el))
    if ecl:
        updates.append(("ECL_WINNERS_ENGLISH", ecl))
    for dict_name, value in updates:
        source = _patch_trophy_dict(source, dict_name, season, value)
    TROPHY_FILE.write_text(source)
    print(f"  trophy_data.py updated with season {season} entries.")


def main():
    parser = argparse.ArgumentParser(description="Add a completed season to the pipeline.")
    parser.add_argument("--season", type=int, required=True, help="Start year (e.g. 2025)")
    parser.add_argument("--pl", default=None)
    parser.add_argument("--fa", default=None)
    parser.add_argument("--lc", default=None)
    parser.add_argument("--cl", default=None)
    parser.add_argument("--el", default=None)
    parser.add_argument("--ecl", default=None)
    args = parser.parse_args()

    ensure_dirs()
    season = args.season
    label = season_label(season)
    print(f"\n=== Adding season {label} ===\n")

    def resolve(val, prompt):
        if val is not None:
            return None if val.lower() == "none" else val
        return _ask(prompt)

    pl = resolve(args.pl, f"PL Champion [{label}]: ")
    fa = resolve(args.fa, f"FA Cup winner [{label}]: ")
    lc = resolve(args.lc, f"League Cup winner [{label}]: ")
    cl = resolve(args.cl, "CL winner (English club, or None): ")
    el = resolve(args.el, "EL winner (English club, or None): ")
    ecl = resolve(args.ecl, "Conference League winner (English club, or None): ")

    print(f"\nTrophies for {label}: PL={pl} FA={fa} LC={lc} CL={cl} EL={el} ECL={ecl}")
    if input("\nWrite to trophy_data.py? [Y/n]: ").strip().lower() not in ("", "y", "yes"):
        print("Aborted.")
        sys.exit(0)
    update_trophy_file(season, pl, fa, lc, cl, el, ecl)

    loader = FDCUKDataLoader()
    print(f"\nDownloading {label} matches...")
    new_matches = loader.download_season_matches(season)
    if new_matches is None or new_matches.empty:
        print(f"ERROR: could not download season {season}.")
        sys.exit(1)
    print(f"  {len(new_matches)} fixtures downloaded.")

    try:
        matches = pd.read_csv(MATCHES_FILE)
        matches = matches[matches["season"] != season]
        matches = pd.concat([matches, new_matches], ignore_index=True)
    except FileNotFoundError:
        matches = new_matches
    matches = matches.sort_values(["season"]).reset_index(drop=True)
    matches.to_csv(MATCHES_FILE, index=False)
    print(f"  Saved {len(matches)} fixtures to {MATCHES_FILE}")

    standings = loader.standings_from_all_matches(matches)
    standings.to_csv(STANDINGS_FILE, index=False)
    out = prepare(standings, matches)
    out.to_csv(TRAINING_FILE, index=False)
    print(f"  Rebuilt {STANDINGS_FILE} and {TRAINING_FILE}")

    print(textwrap.dedent(f"""
    Done. Season {label} added.

    Next steps:
        python -m prem_prediction.model_train      # retrain on the expanded data
        python -m prem_prediction.season_predict   # predict the current season
    """))


if __name__ == "__main__":
    main()
