"""
PHASE 2 -- clean the raw factsheet data into a modelling-ready wide table.

WHICH SOURCE IS PRIMARY, AND WHY
--------------------------------
Values and reliability flags come from the **jvargh7 factsheet mirror**
(`nfhs5_factsheets_districts.csv`). It is the better source for modelling:

  * values are already numeric, with no bracketed `(45.2)` or `*` cells to parse;
  * the reliability information NFHS prints is preserved in its own column -
    "based on 25-49 unweighted cases" and "percentage not shown; fewer than 25
    unweighted cases" - for *every* indicator;
  * suppressed estimates are already blank rather than silently filled in.

`India.csv` (SaiSiddhardhaKalla mirror) is kept for the two things it alone has:
Census 2011 district codes, and a clean set of indicator names.

An earlier version of this file used India.csv for values and joined the flags on
by indicator NAME. That was wrong in a way worth recording: the two mirrors spell
indicators differently, so only 82% of cells matched, and the misses were not
random - 17 indicators (including child anaemia and the whole blood-pressure /
blood-sugar block) matched nothing at all, so small-sample estimates were removed
for some indicators and kept for others. Same data, inconsistent treatment.

HOW THE TWO MIRRORS ARE ALIGNED
-------------------------------
Not by name - the factsheet mirror's names come straight out of the PDFs and are
unusable for matching: gender is carried only by the factsheet item number
(items 86-88 are women's blood sugar, 89-91 men's), and a few names are garbled
by the PDF text extraction.

Instead we align by **value fingerprint**: for each numbered factsheet item, find
the India.csv indicator whose values are closest across all shared districts.
Both mirrors were parsed from the same PDFs, so the true match agrees to about
0.001, while the runner-up is off by a thousand times more. The alignment is
therefore verified from the data rather than assumed, and the script asserts that
every match is unambiguous.

THE REST OF THE PIPELINE
------------------------
  1. pivot long -> wide (one row per district, one column per indicator)
  2. report missingness, drop indicators >10% missing and districts >20% missing
  3. impute the rest with KNNImputer(k=5) on standardised values
  4. winsorise at the 1st/99th percentile
  5. drop near-duplicate indicators (|r| > 0.95)
  6. write data/processed/{features,outcomes,ids}.csv and config/indicators.csv

Features and outcomes are imputed SEPARATELY (features from features, outcomes
from outcomes). If outcome columns helped fill in feature columns, the Phase 7
held-out test would be contaminated and would flatter the clusters.

Run:  python src/clean.py
"""
from __future__ import annotations

import difflib
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import (CONFIG, DATA_INTERIM, DATA_PROC, DATA_RAW, MODELS, REPORTS,
                    SEED, banner, canon)
from indicator_meta import DOMAIN_LABELS, INDICATORS

MAX_MISSING_PER_INDICATOR = 0.10   # drop an indicator missing in >10% of districts
MAX_MISSING_PER_DISTRICT = 0.20    # drop a district missing >20% of indicators
CORR_DUPLICATE_THRESHOLD = 0.95    # |r| above this = near-duplicate indicator
ALIGN_MARGIN = 10.0                # the best value-match must beat the runner-up by this factor


# --------------------------------------------------------------------------
# 2a. load both mirrors
# --------------------------------------------------------------------------
def district_key(state: pd.Series, district: pd.Series, state_map: dict | None = None) -> pd.Series:
    """Join key that survives spelling differences: 'kerala|ernakulam'."""
    s = state.map(state_map) if state_map else state.map(canon)
    return s + "|" + district.map(canon)


def load_reference() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    India.csv, used for (a) the value fingerprints that align the two mirrors and
    (b) Census 2011 district codes.
    """
    df = pd.read_csv(DATA_RAW / "India.csv", low_memory=False, dtype=str)
    for col in ("State", "District", "Indicator"):
        df[col] = df[col].astype(str).str.strip()
    df = df[~df["District"].str.lower().str.contains("not available|unknown", regex=True)]

    df["value"] = pd.to_numeric(df["NFHS 5"], errors="coerce")
    df["key"] = district_key(df["State"], df["District"])

    wide = df.pivot_table(index="key", columns="Indicator", values="value", aggfunc="mean")

    # Census 2011 code = state code * 1000 + district code. Districts created
    # after 2011 have no code here, which is expected and noted in the README.
    st = pd.to_numeric(df["ST_CEN_CD"], errors="coerce")
    dt = pd.to_numeric(df["DT_CEN_CD"], errors="coerce")
    codes = (df.assign(census_code=st * 1000 + dt)
               .groupby("key")["census_code"]
               .agg(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan))
    return wide, codes.to_frame()


def load_primary() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    The factsheet mirror: values, NFHS-4 values, reliability flags.

    Indicators are keyed by their factsheet ITEM NUMBER, not their name, because
    the names lose the gender of the NCD indicators and some are garbled.
    """
    fs = pd.read_csv(DATA_RAW / "nfhs5_factsheets_districts.csv", low_memory=False)
    # Normally the item number leads: '86. Blood sugar level - high...'. In two
    # rows the PDF glued a section heading in front of it ('...blood pressure (%)
    # 98.Ever undergone a screening test...'), so fall back to the LAST 'NN.' in
    # the string, which is the item's own number.
    item = fs["Indicator"].str.extract(r"^\s*(\d+)\s*\.")[0]
    fallback = fs["Indicator"].str.findall(r"(\d+)\s*\.\s*[A-Za-z]").str[-1]
    fs["item"] = item.fillna(fallback).astype(float)
    assert fs["item"].notna().all(), "some factsheet rows have no item number"

    # state names differ slightly between mirrors ('NCT Delhi' vs 'NCT of Delhi')
    reference_states = sorted({canon(s) for s in
                               pd.read_csv(DATA_RAW / "India.csv", low_memory=False,
                                           dtype=str, usecols=["State"])["State"].str.strip().unique()})
    state_map = {}
    for s in fs["state"].unique():
        match = difflib.get_close_matches(canon(s), reference_states, n=1, cutoff=0.6)
        state_map[s] = match[0] if match else canon(s)
    fs["key"] = district_key(fs["state"], fs["district"], state_map)

    flag_text = fs["Flag_NFHS5"].astype(str)
    fs["low_reliability"] = flag_text.str.contains("25-49", na=False).astype(int)
    fs["suppressed"] = flag_text.str.contains("not shown", na=False)

    values5 = fs.pivot_table(index="key", columns="item", values="NFHS5", aggfunc="mean")
    values4 = fs.pivot_table(index="key", columns="item", values="NFHS4", aggfunc="mean")
    flags = fs.pivot_table(index="key", columns="item", values="low_reliability", aggfunc="max")

    names = (fs.drop_duplicates("key")[["key", "state", "district"]]
               .set_index("key").sort_index())

    print(f"  factsheet mirror: {values5.shape[0]} districts x {values5.shape[1]} items")
    print(f"  low-reliability cells (25-49 unweighted cases): {int(fs['low_reliability'].sum()):,}")
    print(f"  suppressed cells (already blank)              : {int(fs['suppressed'].sum()):,}")
    print(f"  cells missing without a suppression flag      : "
          f"{int((fs['NFHS5'].isna() & ~fs['suppressed']).sum()):,}")
    return values5, values4, flags, names


# --------------------------------------------------------------------------
# 2b. align the two mirrors by value fingerprint
# --------------------------------------------------------------------------
def align_items(primary: pd.DataFrame, reference: pd.DataFrame, cfg: pd.DataFrame) -> dict:
    """
    Map factsheet item number -> our short_name, by matching VALUES.

    For each item, compute the mean absolute difference against every India.csv
    indicator over the districts both mirrors share, and take the closest. Both
    files were parsed from the same PDFs, so the correct match agrees to ~0.001
    while the next-best is off by orders of magnitude - which is what lets us
    assert the alignment rather than trust it.
    """
    shared = primary.index.intersection(reference.index)
    print(f"\n  aligning by value fingerprint over {len(shared)} shared districts")
    prim, ref = primary.loc[shared], reference.loc[shared]

    name_to_short = {canon(f): s for f, s, *_ in INDICATORS}
    mapping, report = {}, []
    for item in prim.columns:
        diffs = ref.sub(prim[item], axis=0).abs().mean().dropna().sort_values()
        if len(diffs) < 2:
            continue
        best, runner_up = diffs.index[0], diffs.iloc[1]
        short = name_to_short.get(canon(best))
        if short is None:
            continue
        # unambiguous = the winner is far closer than the runner-up
        assert diffs.iloc[0] * ALIGN_MARGIN < runner_up, (
            f"item {int(item)} is ambiguous: {best} ({diffs.iloc[0]:.4f}) vs "
            f"{diffs.index[1]} ({runner_up:.4f})")
        mapping[item] = short
        report.append(dict(item=int(item), short_name=short, full_name=best,
                           mean_abs_diff=diffs.iloc[0], runner_up_diff=runner_up))

    rep = pd.DataFrame(report).sort_values("item")
    rep.to_csv(REPORTS / "source_alignment.csv", index=False)

    expected = set(cfg["short_name"])
    matched = set(mapping.values())
    print(f"  matched {len(mapping)} factsheet items to {len(matched)} indicators")
    print(f"  worst alignment error: {rep['mean_abs_diff'].max():.4f} "
          f"(runner-up for that item: {rep.loc[rep['mean_abs_diff'].idxmax(), 'runner_up_diff']:.2f})")
    if expected - matched:
        print(f"  NOT FOUND in the factsheet mirror: {sorted(expected - matched)}")
    # items that mapped to the same indicator (duplicate factsheet rows)
    dupes = rep["short_name"].value_counts()
    for short in dupes[dupes > 1].index:
        print(f"  note: {list(rep.loc[rep['short_name'] == short, 'item'])} both map to {short}")
    return mapping


# --------------------------------------------------------------------------
# 2c. indicator metadata
# --------------------------------------------------------------------------
def build_indicator_config() -> pd.DataFrame:
    """Write config/indicators.csv from the curated table in indicator_meta.py."""
    cfg = pd.DataFrame([
        dict(code=canon(full), short_name=short, full_name=full, domain=domain,
             domain_label=DOMAIN_LABELS[domain], direction=direction, role=role)
        for full, short, domain, direction, role in INDICATORS
    ])
    cfg.to_csv(CONFIG / "indicators.csv", index=False)
    return cfg


# --------------------------------------------------------------------------
# 2d. missingness, imputation, near-duplicate removal
# --------------------------------------------------------------------------
def report_missing(mat: pd.DataFrame, ids: pd.DataFrame, lines: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop indicators / districts that are too incomplete to model honestly."""
    miss_ind = mat.isna().mean().sort_values(ascending=False)

    print(f"\n  overall missing cells: {mat.isna().to_numpy().mean():.2%}")
    print("  worst indicators:")
    print(miss_ind.head(8).map(lambda x: f"{x:.1%}").to_string())

    drop_cols = miss_ind[miss_ind > MAX_MISSING_PER_INDICATOR].index.tolist()
    if drop_cols:
        lines.append(f"Indicators dropped (> {MAX_MISSING_PER_INDICATOR:.0%} missing): "
                     + ", ".join(f"`{c}` ({miss_ind[c]:.1%})" for c in drop_cols))
        print(f"  dropping {len(drop_cols)} indicator(s): {drop_cols}")
    mat = mat.drop(columns=drop_cols)

    miss_dist = mat.isna().mean(axis=1)
    drop_rows = miss_dist[miss_dist > MAX_MISSING_PER_DISTRICT].index
    if len(drop_rows):
        names = ids.loc[drop_rows, ["state", "district"]].apply(" / ".join, axis=1).tolist()
        lines.append(f"Districts dropped (> {MAX_MISSING_PER_DISTRICT:.0%} of indicators missing): "
                     + "; ".join(names))
        print(f"  dropping {len(drop_rows)} district(s): {names}")
    return mat.drop(index=drop_rows), ids.drop(index=drop_rows)


def knn_impute(mat: pd.DataFrame, name: str) -> tuple[pd.DataFrame, Pipeline]:
    """
    KNN imputation on STANDARDISED values.

    Standardising first matters: KNN measures distance between districts, and
    without scaling an indicator measured in rupees (thousands) would drown out
    percentages (0-100) when choosing the 5 nearest neighbours.
    """
    pipe = Pipeline([("scale", StandardScaler()), ("impute", KNNImputer(n_neighbors=5))])
    z = pipe.fit_transform(mat)
    filled = pipe.named_steps["scale"].inverse_transform(z)   # back to real units
    out = pd.DataFrame(filled, index=mat.index, columns=mat.columns)
    print(f"  {name}: imputed {int(mat.isna().sum().sum()):,} cells with KNN(k=5)")
    return out, pipe


def drop_near_duplicates(mat: pd.DataFrame, cfg: pd.DataFrame, lines: list[str]) -> pd.DataFrame:
    """
    Remove indicators that carry almost the same information (|r| > 0.95).

    Why: two columns that are 0.98 correlated give that concept double weight in
    both the distance calculation and PCA. Which one to keep is decided by the
    curated order in indicator_meta.py, where the headline/total version of an
    indicator is listed before its sub-components.
    """
    order = {short: i for i, short in enumerate(cfg["short_name"])}
    corr = mat.corr().abs()
    dropped = {}
    for i, a in enumerate(corr.columns):
        if a in dropped:
            continue
        for b in corr.columns[i + 1:]:
            if b in dropped:
                continue
            if corr.loc[a, b] > CORR_DUPLICATE_THRESHOLD:
                loser = b if order[a] <= order[b] else a
                keeper = a if loser == b else b
                dropped[loser] = (keeper, corr.loc[a, b])
                if loser == a:
                    break
    if dropped:
        print(f"\n  dropping {len(dropped)} near-duplicate indicator(s) (|r| > {CORR_DUPLICATE_THRESHOLD}):")
        for loser, (keeper, r) in dropped.items():
            print(f"    {loser:38s} ~ {keeper:38s} r={r:.3f}")
            lines.append(f"Near-duplicate dropped: `{loser}` (|r| = {r:.3f} with `{keeper}`, kept)")
    return mat.drop(columns=list(dropped))


# --------------------------------------------------------------------------
def main() -> None:
    banner("PHASE 2 - CLEANING")
    lines: list[str] = []          # collected notes -> reports/cleaning_report.md

    cfg = build_indicator_config()
    print(f"  config/indicators.csv written: {len(cfg)} indicators")
    print(cfg["role"].value_counts().to_string())

    values5, values4, flags, names = load_primary()
    reference, codes = load_reference()
    mapping = align_items(values5, reference, cfg)

    # item numbers -> short names, then collapse any item that maps to the same
    # indicator twice (a handful of factsheet rows are printed in two sections)
    def rename(mat: pd.DataFrame) -> pd.DataFrame:
        out = mat[[c for c in mat.columns if c in mapping]].rename(columns=mapping)
        return out.T.groupby(level=0).mean().T

    wide5, wide4, wide_flags = rename(values5), rename(values4), rename(flags)

    ids = (names.join(codes, how="left").reset_index()
           .rename(columns={"index": "key"}))
    ids["census_code"] = ids["census_code"].astype("Int64")
    print(f"\n  districts: {len(ids)} | with a Census 2011 code: "
          f"{ids['census_code'].notna().sum()}")

    # reindex, not .loc: districts created after NFHS-4 have no 2015-16 row at
    # all, and pivot_table simply omits them. reindex puts them back as blanks.
    wide5 = wide5.reindex(ids["key"]).reset_index(drop=True)
    wide4 = wide4.reindex(ids["key"]).reset_index(drop=True)
    wide_flags = wide_flags.reindex(ids["key"]).reset_index(drop=True)
    ids = ids.drop(columns="key")

    for frame, path in ((wide5, "nfhs5_wide_raw.csv"), (wide4, "nfhs4_wide_raw.csv"),
                        (wide_flags, "nfhs5_reliability_flags.csv")):
        pd.concat([ids[["state", "district"]], frame], axis=1).to_csv(DATA_INTERIM / path, index=False)

    feature_cols = [c for c in cfg.loc[cfg["role"] == "feature", "short_name"] if c in wide5.columns]
    outcome_cols = [c for c in cfg.loc[cfg["role"] == "outcome", "short_name"] if c in wide5.columns]
    X, Y = wide5[feature_cols].copy(), wide5[outcome_cols].copy()

    banner("MISSING DATA - FEATURES")
    X, ids = report_missing(X, ids, lines)
    Y = Y.loc[X.index]
    banner("MISSING DATA - OUTCOMES (held out)")
    Y, _ = report_missing(Y, ids, lines)
    X = X.loc[Y.index]
    ids = ids.loc[Y.index]

    banner("IMPUTATION")
    X_imp, feat_pipe = knn_impute(X, "features")
    Y_imp, _ = knn_impute(Y, "outcomes")

    banner("NEAR-DUPLICATE INDICATORS")
    X_imp = drop_near_duplicates(X_imp, cfg, lines)

    banner("WINSORISING EXTREME VALUES")
    # NFHS district estimates come with sampling error, and a handful of very
    # small districts (Chandigarh, Lakshadweep) sit 15+ standard deviations out
    # on single indicators. Left alone, one district drags a whole cluster onto
    # itself. Clipping at the 1st / 99th percentile keeps every district in the
    # analysis while stopping any one value from dominating.
    X_imp.to_csv(DATA_INTERIM / "features_unclipped.csv", index=False)
    lo, hi = X_imp.quantile(0.01), X_imp.quantile(0.99)
    n_clipped = int(((X_imp < lo) | (X_imp > hi)).to_numpy().sum())
    X_imp = X_imp.clip(lower=lo, upper=hi, axis=1)
    print(f"  clipped {n_clipped:,} cells ({n_clipped / X_imp.size:.1%}) to the 1st/99th percentile")
    lines.append(f"Winsorised {n_clipped:,} feature cells "
                 f"({n_clipped / X_imp.size:.1%}) at the 1st/99th percentile.")

    joblib.dump({"pipeline": feat_pipe, "columns": X.columns.tolist(),
                 "final_columns": X_imp.columns.tolist(),
                 "clip_lo": lo[X_imp.columns], "clip_hi": hi[X_imp.columns]},
                MODELS / "imputer.joblib")

    ids = ids.reset_index(drop=True)
    X_imp = X_imp.reset_index(drop=True)
    Y_imp = Y_imp.reset_index(drop=True)
    ids.to_csv(DATA_PROC / "ids.csv", index=False)
    X_imp.to_csv(DATA_PROC / "features.csv", index=False)
    Y_imp.to_csv(DATA_PROC / "outcomes.csv", index=False)
    X.reset_index(drop=True).to_csv(DATA_INTERIM / "features_before_imputation.csv", index=False)

    banner("PHASE 2 RESULT")
    print(f"  districts           : {len(ids)}")
    print(f"  clustering features : {X_imp.shape[1]}")
    print(f"  held-out outcomes   : {Y_imp.shape[1]} -> {list(Y_imp.columns)}")
    print(f"  saved               : data/processed/{{features,outcomes,ids}}.csv")

    header = [
        "# Phase 2 - cleaning report", "",
        "- Values and reliability flags: **jvargh7 factsheet mirror** (already numeric, "
        "flags for every indicator). India.csv supplies Census 2011 codes and the "
        "indicator names, and the two are aligned by value fingerprint "
        "(see `reports/source_alignment.csv`).",
        f"- Districts kept: **{len(ids)}**",
        f"- Clustering features: **{X_imp.shape[1]}** (from {len(feature_cols)} candidates)",
        f"- Held-out outcomes: **{Y_imp.shape[1]}**",
        "- Missing cells imputed with KNNImputer(k=5) on standardised values, "
        "features and outcomes imputed separately to protect the held-out test.",
        "",
    ]
    (REPORTS / "cleaning_report.md").write_text("\n".join(header + [f"- {l}" for l in lines]),
                                                encoding="utf-8")


if __name__ == "__main__":
    main()
