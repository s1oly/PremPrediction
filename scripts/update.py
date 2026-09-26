#!/usr/bin/env python
"""
Thin wrapper so the gameweek refresh can be run as a plain script:

    python scripts/update.py

Equivalent to `python -m prem_prediction.update`. Requires the package to be
importable (run `pip install -e .` once, or run from the repo root).
"""

import os
import sys

# Allow running without an editable install by adding ./src to the path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prem_prediction.update import main

if __name__ == "__main__":
    main()
