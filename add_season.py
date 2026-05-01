"""
Add a newly completed Premier League season to the training dataset and
update trophy_data.py with that season's trophy winners.

Usage (interactive):
    python add_season.py --season 2025

    The script will prompt for each trophy winner.  Team names must match
    the football-data.co.uk convention (e.g. "Man City", "Nott'm Forest").

Usage (non-interactive, supply all flags):
    python add_season.py --season 2025 \\
        --pl Liverpool --fa "Crystal Palace" --lc Liverpool \\
        --cl None --el Tottenham --ecl None

After running:
    1. historical_standings.csv is updated with the new season.
    2. out.csv is rebuilt with ELO and labels.
    3. trophy_data.py is patched with the new entries.
    4. Re-train: python model_train.py
"""

import argparse
import ast
import re
import sys
import textwrap

import pandas as pd

from data_aquisition import FDCUKDataLoader
from prepare_model_data import prepare

STANDINGS_FILE = "historical_standings.csv"
TRAINING_FILE  = "out.csv"
TROPHY_FILE    = "trophy_data.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ask(prompt: str, allow_none: bool = True) -> str | None:
    raw = input(prompt).strip()
    if allow_none and raw.lower() in ("none", "n/a", ""):
        return None
    return raw


def _patch_trophy_dict(source: str, dict_name: str, season: int, value: str | None) -> str:
    """
    Insert `season: "value",` (or `season: None,`) into the named dict in source.
    Finds the last entry and appends after it.
    """
    if value is None:
        entry = f"    {season}: None,              # {season}-{(season+1) % 100:02d}\n"
    else:
        entry = f"    {season}: \"{value}\",{' ' * max(1, 18 - len(value))}# {season}-{(season+1) % 100:02d}\n"

    # Find the closing brace of the target dict
    pattern = re.compile(
        rf"({re.escape(dict_name)}\s*=\s*\{{.*?\}})",
        re.DOTALL,
    )
    match = pattern.search(source)
    if not match:
        print(f"  WARNING: could not find {dict_name} in {TROPHY_FILE} — skipping patch.")
        return source

    dict_text = match.group(1)
    # Insert before the closing brace
    updated_dict = dict_text.rstrip().rstrip("}") + entry + "}\n"
    return source[:match.start()] + updated_dict + source[match.end():]


def update_trophy_file(season: int, pl: str | None, fa: str | None, lc: str | None,
                       cl: str | None, el: str | None, ecl: str | None) -> None:
    with open(TROPHY_FILE, "r") as f:
        source = f.read()

    updates = [
        ("PL_CHAMPIONS",     pl),
        ("FA_CUP_WINNERS",   fa),
        ("LEAGUE_CUP_WINNERS", lc),
    ]
    if cl:
        updates.append(("CL_WINNERS_ENGLISH", cl))
    if el:
        updates.append(("EL_WINNERS_ENGLISH", el))
    if ecl:
        updates.append(("ECL_WINNERS_ENGLISH", ecl))

    for dict_name, value in updates:
        source = _patch_trophy_dict(source, dict_name, season, value)

    with open(TROPHY_FILE, "w") as f:
        f.write(source)
    print(f"  trophy_data.py updated with season {season} entries.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Add a completed season to the training pipeline.")
    parser.add_argument("--season", type=int, required=True,
                        help="Start year of the season to add (e.g. 2025 for 2025-26)")
    parser.add_argument("--pl",  default=None, help="PL champion team name")
    parser.add_argument("--fa",  default=None, help="FA Cup winner team name")
    parser.add_argument("--lc",  default=None, help="League Cup winner team name")
    parser.add_argument("--cl",  default=None, help="CL winner (English club) or 'None'")
    parser.add_argument("--el",  default=None, help="EL winner (English club) or 'None'")
    parser.add_argument("--ecl", default=None, help="Conference League winner (English) or 'None'")
    args = parser.parse_args()

    season = args.season
    end_yr = (season + 1) % 100
    season_label = f"{season}-{end_yr:02d}"
    print(f"\n=== Adding season {season_label} ===\n")

    # --- Trophy data ---
    def resolve(val, prompt):
        if val is not None:
            return None if val.lower() == "none" else val
        return _ask(prompt)

    pl  = resolve(args.pl,  f"PL Champion [{season_label}]: ")
    fa  = resolve(args.fa,  f"FA Cup winner [{season_label}]: ")
    lc  = resolve(args.lc,  f"League Cup winner [{season_label}]: ")
    cl  = resolve(args.cl,  f"CL winner English club (or blank/None): ")
    el  = resolve(args.el,  f"EL winner English club (or blank/None): ")
    ecl = resolve(args.ecl, f"Conference League winner English club (or blank/None): ")

    print(f"\nTrophies for {season_label}:")
    print(f"  PL:  {pl}")
    print(f"  FA:  {fa}")
    print(f"  LC:  {lc}")
    print(f"  CL:  {cl}")
    print(f"  EL:  {el}")
    print(f"  ECL: {ecl}")
    confirm = input("\nWrite to trophy_data.py? [Y/n]: ").strip().lower()
    if confirm not in ("", "y", "yes"):
        print("Aborted.")
        sys.exit(0)

    update_trophy_file(season, pl, fa, lc, cl, el, ecl)

    # --- Standings data ---
    loader = FDCUKDataLoader()
    print(f"\nDownloading {season_label} standings from football-data.co.uk...")
    new_df = loader.get_season(season)
    if new_df is None or new_df.empty:
        print(f"ERROR: could not download season {season}. Check the season code and try again.")
        sys.exit(1)
    print(f"  {len(new_df)} rows downloaded.")

    # Load existing standings
    try:
        existing = pd.read_csv(STANDINGS_FILE)
        if season in existing["season"].values:
            print(f"  Season {season} already exists in {STANDINGS_FILE} — overwriting it.")
            existing = existing[existing["season"] != season]
        combined_raw = pd.concat([existing, new_df], ignore_index=True)
    except FileNotFoundError:
        print(f"  {STANDINGS_FILE} not found — creating it fresh.")
        combined_raw = new_df

    combined_raw = combined_raw.sort_values(["season", "teamName", "gameweek"]).reset_index(drop=True)
    combined_raw.to_csv(STANDINGS_FILE, index=False)
    print(f"  Saved {len(combined_raw)} rows to {STANDINGS_FILE}")

    # Rebuild out.csv
    print("\nRebuilding training data with ELO and labels...")
    out = prepare(combined_raw)
    out.to_csv(TRAINING_FILE, index=False)
    n_seqs = out.groupby(["season", "teamName"]).filter(lambda g: len(g) >= 38).groupby(["season", "teamName"]).ngroups
    print(f"  Saved {len(out)} rows to {TRAINING_FILE}")
    print(f"  Complete training sequences: {n_seqs}")

    print(textwrap.dedent(f"""
    Done.  Season {season_label} has been added.

    Next steps:
        python model_train.py      # retrain the model on the expanded dataset
        python season_predict.py   # generate predictions for the current season
    """))


if __name__ == "__main__":
    main()
