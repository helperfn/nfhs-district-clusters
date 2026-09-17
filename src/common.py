"""
Shared paths, constants and small helpers used by every script in the project.

Keeping these in one place means:
  * every script writes to the same folders (no scattered output),
  * the random seed is identical everywhere, so results are reproducible
    (the rubric and the viva both care about reproducibility).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- paths -----
# Path(__file__).resolve().parents[1] = the project root (nfhs-clusters/),
# no matter which folder the script is launched from. This is what makes the
# project work unchanged on Windows and macOS.
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROC = ROOT / "data" / "processed"
CONFIG = ROOT / "config"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

for _p in (DATA_RAW, DATA_INTERIM, DATA_PROC, CONFIG, MODELS, REPORTS, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------- reproducibility -
SEED = 42
np.random.seed(SEED)

# ------------------------------------------------------------- text helpers --
def canon(name: str) -> str:
    """
    Canonical form of an indicator name: lower-case, letters and digits only.

    NFHS factsheets are typed by hand, so the same indicator appears as
    'Male Blood sugar level high...' and 'MaleBlood sugar level high...'.
    Stripping spaces and punctuation makes both collapse to one key, which is
    how we detect and merge misspelt duplicates without guessing.
    """
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def save_json(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def banner(title: str) -> None:
    """Consistent section headers so long console output stays readable."""
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
