"""
PHASE 5 -- the traditional groupings that ML clustering has to beat.

These are not straw men. A composite index is how development ranking is
normally done (SDG district index, Aspirational Districts Programme, HDI-style
scores), and "which state is it in" is how funds are usually allocated. If
clustering cannot do better than these, the project's premise is wrong and the
report should say so.

  1. composite index        flip the negative indicators, min-max each to 0-1,
                            average with equal weights -> one score per district,
                            then cut into k equal-sized quantile groups
  2. domain-weighted index  average within each domain first, then across the
                            11 domains. This stops a domain with 15 indicators
                            (NCDs) outvoting a domain with 2 (education).
  3. state grouping         districts grouped by their state/UT
  4. region grouping        states mapped to North / South / East / West /
                            Central / North-East

Every baseline produces the same kind of object as a clustering - a label per
district - so Phase 7 can score them all with exactly the same metrics.

Run:  python src/baselines.py     (after cluster.py, which fixes k)
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common import (CONFIG, DATA_PROC, FIGURES, REPORTS, banner, canon,
                    load_json, save_json)

import difflib

sns.set_theme(style="whitegrid", context="talk")

# Standard six-region grouping of Indian states/UTs (Ministry of Home Affairs
# zonal councils, with the North-East kept separate as is conventional in
# health research because its districts behave very differently).
REGIONS = {
    "North": ["Jammu & Kashmir", "Ladakh", "Himachal Pradesh", "Punjab", "Haryana",
              "Chandigarh", "NCT of Delhi", "Rajasthan", "Uttarakhand"],
    "Central": ["Uttar Pradesh", "Madhya Pradesh", "Chhattisgarh"],
    "East": ["Bihar", "Jharkhand", "Odisha", "West Bengal"],
    "West": ["Gujarat", "Maharashtra", "Goa", "Dadra & Nagar Haveli", "Daman & Diu"],
    "South": ["Andhra Pradesh", "Telangana", "Karnataka", "Kerala", "Tamil Nadu",
              "Puducherry", "Lakshadweep", "Andaman & Nicobar Island"],
    "North-East": ["Assam", "Arunachal Pradesh", "Manipur", "Meghalaya", "Mizoram",
                   "Nagaland", "Tripura", "Sikkim"],
}


def orient(X: pd.DataFrame, cfg: pd.DataFrame) -> pd.DataFrame:
    """
    Flip the indicators where higher is worse, then min-max scale to 0-1.

    After this every column reads the same way: 1 = best district on that
    indicator, 0 = worst. Only then is averaging them meaningful.
    """
    direction = dict(zip(cfg["short_name"], cfg["direction"]))
    good = X.copy()
    for col in good.columns:
        if direction[col] < 0:
            good[col] = -good[col]
    return (good - good.min()) / (good.max() - good.min())


def quantile_groups(score: pd.Series, k: int) -> np.ndarray:
    """
    Cut a continuous score into k equal-sized groups.

    This is the fairest possible comparison: the baseline gets exactly as many
    groups as the clustering, so any difference in the Phase 7 metrics is about
    HOW districts are grouped, not how many groups there are.
    Group 0 = lowest score (worst off).
    """
    return pd.qcut(score.rank(method="first"), q=k, labels=False).to_numpy()


def main() -> None:
    banner("PHASE 5 - TRADITIONAL BASELINES")
    X = pd.read_csv(DATA_PROC / "features.csv")
    ids = pd.read_csv(DATA_PROC / "ids.csv")
    cfg = pd.read_csv(CONFIG / "indicators.csv")
    k = load_json(CONFIG / "k_choice.json")["k"]
    print(f"  using k = {k} (same as the final clustering, so the comparison is fair)")

    good = orient(X, cfg)
    domain_of = dict(zip(cfg["short_name"], cfg["domain"]))

    # ---- 1. equal-weight composite index ---------------------------------
    composite = good.mean(axis=1)

    # ---- 2. domain-weighted composite ------------------------------------
    by_domain = good.T.groupby(good.columns.map(domain_of)).mean().T
    domain_score = by_domain.mean(axis=1)

    out = ids.copy()
    out["composite_score"] = composite
    out["composite_group"] = quantile_groups(composite, k)
    out["domain_score"] = domain_score
    out["domain_group"] = quantile_groups(domain_score, k)

    # ---- 3. state grouping -----------------------------------------------
    out["state_group"] = out["state"].astype("category").cat.codes

    # ---- 4. region grouping ----------------------------------------------
    # Match on canonical spelling, then fall back to closest spelling: the
    # factsheet mirror writes 'NCT Delhi' and 'Jammu Kashmir' where India.csv
    # writes 'NCT of Delhi' and 'Jammu & Kashmir', and it uses the merged
    # 'Dadra Nagar Haveli Daman Diu' UT. All of these must land in a region.
    canon_to_region = {canon(s): r for r, states in REGIONS.items() for s in states}
    def region_of(state: str) -> str:
        key = canon(state)
        if key in canon_to_region:
            return canon_to_region[key]
        match = difflib.get_close_matches(key, list(canon_to_region), n=1, cutoff=0.5)
        return canon_to_region[match[0]] if match else None

    out["region"] = out["state"].map(region_of)
    missing = sorted(out.loc[out["region"].isna(), "state"].unique())
    assert not missing, f"states missing from the region map: {missing}"
    print("  region mapping resolved by closest spelling where needed: "
          + ", ".join(f"{s} -> {region_of(s)}" for s in sorted(out['state'].unique())
                      if canon(s) not in canon_to_region))
    out["region_group"] = out["region"].astype("category").cat.codes

    for name, col in [("composite", "composite_group"), ("domain", "domain_group"),
                      ("state", "state_group"), ("region", "region_group")]:
        print(f"  {name:10s}: {out[col].nunique()} groups, "
              f"sizes {out[col].value_counts().sort_index().tolist()[:8]}"
              + (" ..." if out[col].nunique() > 8 else ""))

    out.to_csv(DATA_PROC / "baselines.csv", index=False)
    by_domain.to_csv(DATA_PROC / "domain_scores_0_1.csv", index=False)
    save_json(REGIONS, CONFIG / "regions.json")

    # ---- the two indices agree almost perfectly ---------------------------
    r = composite.corr(domain_score, method="spearman")
    print(f"\n  Spearman correlation between the two indices: {r:.3f}")
    print("  (they rank districts nearly identically, which is the usual finding:")
    print("   a single score is a single score however you weight it)")

    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    axes[0].scatter(composite, domain_score, s=14, alpha=0.6, color="#2c7fb8")
    axes[0].set_xlabel("equal-weight composite score")
    axes[0].set_ylabel("domain-weighted score")
    axes[0].set_title(f"The two indices agree (Spearman r = {r:.2f})")

    sns.histplot(composite, bins=40, ax=axes[1], color="#2c7fb8")
    for q in np.linspace(0, 1, k + 1)[1:-1]:
        axes[1].axvline(composite.quantile(q), ls="--", lw=1, color="#c7254e")
    axes[1].set_title(f"Composite score, cut into {k} equal-sized groups")
    axes[1].set_xlabel("composite score (1 = best)")
    fig.tight_layout()
    fig.savefig(FIGURES / "14_baseline_index.png", dpi=140)
    plt.close(fig)

    ranked = out.sort_values("composite_score", ascending=False)
    lines = [
        "# Phase 5 - traditional baselines", "",
        f"- Composite index and domain-weighted index both cut into **{k}** equal-sized groups.",
        f"- Spearman correlation between the two indices: **{r:.3f}**.",
        f"- State grouping: **{out['state_group'].nunique()}** states/UTs.",
        f"- Region grouping: **{out['region_group'].nunique()}** regions.", "",
        "## Region mapping", "",
        "| region | states/UTs |", "|---|---|",
        *[f"| {r_} | {', '.join(s for s in states if s in set(out['state']))} |"
          for r_, states in REGIONS.items()],
        "", "## Top 10 districts by composite score", "",
        ranked.head(10)[["state", "district", "composite_score"]].round(3).to_markdown(index=False),
        "", "## Bottom 10 districts by composite score", "",
        ranked.tail(10)[["state", "district", "composite_score"]].round(3).to_markdown(index=False),
    ]
    (REPORTS / "baselines.md").write_text("\n".join(lines), encoding="utf-8")
    print("  saved data/processed/baselines.csv and reports/baselines.md")


if __name__ == "__main__":
    main()
