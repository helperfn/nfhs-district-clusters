"""
PHASE 1 -- download the raw NFHS district data and describe what we got.

Primary source (district-level NFHS-5 *and* NFHS-4, matched to Census 2011
codes, all states/UTs):
    https://github.com/SaiSiddhardhaKalla/NFHS      -> India.csv

Backup / cross-check source (consolidated factsheets):
    https://github.com/jvargh7/nfhs5_factsheets     -> districts.csv

Nothing here is edited: data/raw/ is treated as read-only for the rest of the
project, so the pipeline can always be re-run from the original bytes.

Run:  python src/download.py
"""
from __future__ import annotations

import sys
import urllib.request

import pandas as pd

from common import DATA_RAW, banner

SOURCES = {
    # name -> (url, required?)
    "India.csv": (
        "https://raw.githubusercontent.com/SaiSiddhardhaKalla/NFHS/main/India.csv",
        True,
    ),
    "nfhs5_factsheets_districts.csv": (
        "https://raw.githubusercontent.com/jvargh7/nfhs5_factsheets/main/data%20for%20analysis/districts.csv",
        False,  # only a cross-check; the pipeline still works if this 404s
    ),
}


def fetch(name: str, url: str, required: bool) -> bool:
    dest = DATA_RAW / name
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [skip] {name} already downloaded ({dest.stat().st_size/1e6:.1f} MB)")
        return True
    try:
        print(f"  [get ] {url}")
        urllib.request.urlretrieve(url, dest)
        print(f"         saved -> {dest} ({dest.stat().st_size/1e6:.1f} MB)")
        return True
    except Exception as exc:  # noqa: BLE001 - we want the reason printed, not a traceback
        if dest.exists():
            dest.unlink()
        msg = f"  [fail] {name}: {exc}"
        if required:
            print(msg)
            sys.exit("Primary source could not be downloaded - check your connection.")
        print(msg + "  (optional source, continuing)")
        return False


def describe(path) -> None:
    """Print the summary Phase 1 asks for: districts, states, indicators, format."""
    df = pd.read_csv(path, low_memory=False, dtype=str)
    banner(f"RAW FILE: {path.name}")
    print(f"rows x cols        : {df.shape[0]:,} x {df.shape[1]}")
    print(f"columns            : {list(df.columns)}")

    # One row per (district, indicator) with a value column per survey round
    # => the file is in LONG format and must be pivoted to wide in Phase 2.
    print("\nformat             : LONG (one row per district x indicator)")
    print("                     value columns 'NFHS 5' (2019-21) and 'NFHS 4' (2015-16)")

    df["State"] = df["State"].str.strip()
    df["District"] = df["District"].str.strip()
    print(f"\nstates / UTs       : {df['State'].nunique()}")
    print(f"districts          : {df.groupby(['State', 'District']).ngroups}")
    print(f"indicators (raw)   : {df['Indicator'].nunique()}")
    print(f"thematic categories: {df['Category'].nunique()}")

    print("\nsample indicators:")
    for ind in df["Indicator"].drop_duplicates().head(8):
        print("   -", ind)

    print("\nrows per district (should be ~104):")
    print(df.groupby(["State", "District"]).size().value_counts().to_string())

    print(
        "\nNOTE: this mirror already parsed the factsheet PDFs, so bracketed\n"
        "      '(value)' estimates and suppressed '*' cells arrive as plain numbers\n"
        "      or blanks. clean.py still implements the bracket/asterisk parser so\n"
        "      the pipeline also works on the original IIPS factsheet exports."
    )


def main() -> None:
    banner("PHASE 1 - DOWNLOAD")
    ok = {name: fetch(name, url, req) for name, (url, req) in SOURCES.items()}
    describe(DATA_RAW / "India.csv")
    if not ok.get("nfhs5_factsheets_districts.csv"):
        print("\n(cross-check file unavailable - Phase 2 will proceed on the primary file)")


if __name__ == "__main__":
    main()
