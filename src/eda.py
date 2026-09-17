"""
PHASE 3 -- exploratory data analysis.

The figures here are not decoration: each one justifies a modelling decision
that comes later.

  missing-value heatmap   -> shows WHERE the gaps are (small-denominator
                             indicators), which is why we dropped some columns
                             instead of imputing everything
  histograms              -> shows the indicators are wide-spread and skewed,
                             so standardising is necessary
  correlation heatmap     -> shows thick blocks of correlated indicators: the
                             direct justification for PCA
  state boxplots          -> shows districts inside one state differ a lot,
                             which is why we cluster districts, not states
  top/bottom tables       -> sanity check that the data behaves as expected

Run:  python src/eda.py
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")           # render to files, never pop up a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common import CONFIG, DATA_INTERIM, DATA_PROC, FIGURES, REPORTS, banner

sns.set_theme(style="whitegrid", context="talk")

# indicators shown in the histogram grid and the state boxplots
KEY_INDICATORS = [
    "women_10yr_schooling", "improved_sanitation", "clean_cooking_fuel",
    "electricity", "institutional_births", "anc_4plus_visits",
    "fully_vaccinated_card_or_recall", "married_before_18",
    "health_insurance", "women_literate", "fp_any_modern_method",
    "adequate_diet_breastfed",
]


def load():
    X = pd.read_csv(DATA_PROC / "features.csv")
    Y = pd.read_csv(DATA_PROC / "outcomes.csv")
    ids = pd.read_csv(DATA_PROC / "ids.csv")
    cfg = pd.read_csv(CONFIG / "indicators.csv")
    raw = pd.read_csv(DATA_INTERIM / "features_before_imputation.csv")
    return X, Y, ids, cfg, raw


def fig_missing(raw: pd.DataFrame) -> None:
    plt.figure(figsize=(16, 9))
    sns.heatmap(raw.isna().T, cbar=False, cmap=["#f0f0f0", "#c0392b"])
    plt.title("Missing values before imputation (red = missing)\n"
              "rows = indicators, columns = districts", fontsize=14)
    plt.xlabel("districts")
    plt.ylabel("")
    plt.yticks(fontsize=6)
    plt.xticks([])
    plt.tight_layout()
    plt.savefig(FIGURES / "01_missing_heatmap.png", dpi=140)
    plt.close()


def fig_histograms(X: pd.DataFrame) -> None:
    cols = [c for c in KEY_INDICATORS if c in X.columns][:12]
    fig, axes = plt.subplots(4, 3, figsize=(16, 14))
    for ax, col in zip(axes.ravel(), cols):
        sns.histplot(X[col], bins=30, ax=ax, color="#2c7fb8")
        ax.set_title(col, fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel("districts", fontsize=9)
        ax.tick_params(labelsize=8)
    fig.suptitle("Distribution of 12 key indicators across districts", fontsize=16)
    fig.tight_layout()
    fig.savefig(FIGURES / "02_histograms.png", dpi=140)
    plt.close(fig)


def fig_correlation(X: pd.DataFrame, cfg: pd.DataFrame) -> pd.DataFrame:
    """Correlation heatmap with columns ORDERED BY DOMAIN, so blocks are visible."""
    order = (cfg[cfg["short_name"].isin(X.columns)]
             .sort_values(["domain", "short_name"])["short_name"].tolist())
    corr = X[order].corr()
    plt.figure(figsize=(18, 15))
    sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                square=True, xticklabels=True, yticklabels=True,
                cbar_kws={"shrink": 0.6, "label": "Pearson r"})
    plt.xticks(fontsize=5, rotation=90)
    plt.yticks(fontsize=5)
    plt.title("Feature correlation, indicators grouped by domain\n"
              "dark blocks = many indicators measuring the same underlying thing "
              "(the reason PCA is used)", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES / "03_correlation_heatmap.png", dpi=140)
    plt.close()
    return corr


def fig_state_boxplots(X: pd.DataFrame, ids: pd.DataFrame) -> pd.DataFrame:
    """Within-state spread: if states were homogeneous, state grouping would do."""
    df = pd.concat([ids[["state"]], X], axis=1)
    big_states = df["state"].value_counts()
    big_states = big_states[big_states >= 10].index[:18]
    sub = df[df["state"].isin(big_states)]

    cols = [c for c in ["improved_sanitation", "women_10yr_schooling",
                        "clean_cooking_fuel", "institutional_births"] if c in X.columns]
    fig, axes = plt.subplots(len(cols), 1, figsize=(16, 5 * len(cols)))
    for ax, col in zip(np.atleast_1d(axes), cols):
        order = sub.groupby("state")[col].median().sort_values().index
        sns.boxplot(data=sub, x="state", y=col, order=order, ax=ax, color="#7fcdbb")
        ax.set_title(f"{col}: district-level spread within each state", fontsize=12)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=90, labelsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "04_state_boxplots.png", dpi=140)
    plt.close(fig)

    # quantify it: how much of the variance is WITHIN states rather than between?
    rows = []
    for col in X.columns:
        grand = X[col].mean()
        between = df.groupby("state")[col].apply(lambda s: len(s) * (s.mean() - grand) ** 2).sum()
        total = ((X[col] - grand) ** 2).sum()
        rows.append({"indicator": col, "within_state_share": 1 - between / total})
    return pd.DataFrame(rows).sort_values("within_state_share", ascending=False)


def fig_top_bottom(X: pd.DataFrame, ids: pd.DataFrame) -> str:
    """Top / bottom 10 districts on a few indicators - a readability sanity check."""
    out = []
    df = pd.concat([ids[["state", "district"]], X], axis=1)
    for col in ["improved_sanitation", "women_10yr_schooling", "clean_cooking_fuel"]:
        if col not in X.columns:
            continue
        s = df[["state", "district", col]].sort_values(col, ascending=False)
        out.append(f"\n### {col}\n")
        out.append("**Top 10**\n")
        out.append(s.head(10).to_markdown(index=False))
        out.append("\n**Bottom 10**\n")
        out.append(s.tail(10).to_markdown(index=False))
    return "\n".join(out)


def main() -> None:
    banner("PHASE 3 - EDA")
    X, Y, ids, cfg, raw = load()

    fig_missing(raw)
    fig_histograms(X)
    corr = fig_correlation(X, cfg)
    within = fig_state_boxplots(X, ids)
    tables = fig_top_bottom(X, ids)

    # ---- numbers quoted in the report -------------------------------------
    abs_corr = corr.abs().where(~np.eye(len(corr), dtype=bool))
    strong = (abs_corr > 0.6).sum().sum() / 2
    mean_abs = abs_corr.stack().mean()
    top_pairs = (abs_corr.stack().sort_values(ascending=False)
                 .drop_duplicates().head(10))

    print(f"  districts x features       : {X.shape}")
    print(f"  mean |correlation|         : {mean_abs:.2f}")
    print(f"  feature pairs with |r|>0.6 : {int(strong)}")
    print(f"  median within-state share of variance: {within['within_state_share'].median():.2f}")

    lines = [
        "# Phase 3 - EDA summary", "",
        f"- **{X.shape[0]} districts x {X.shape[1]} features** after cleaning.",
        f"- Mean absolute correlation between features is **{mean_abs:.2f}**, and "
        f"**{int(strong)} feature pairs** correlate above |r| = 0.6. The indicators are "
        "far from independent, which is exactly the situation PCA is for: a handful of "
        "components can carry most of the information without the double-counting.",
        f"- **{within['within_state_share'].median():.0%}** of the variance in a typical "
        "indicator sits *within* states rather than between them. Districts inside the same "
        "state are not alike, so 'group by state' throws away most of the signal - this is "
        "the motivation for clustering at district level.",
        "- The missing-value heatmap shows gaps are concentrated in small-denominator "
        "indicators (diarrhoea treatment, non-breastfed infant diet), where NFHS suppresses "
        "estimates based on fewer than 25 unweighted cases. Those columns were dropped "
        "rather than imputed.",
        "", "## Most correlated feature pairs", "",
        top_pairs.reset_index().rename(
            columns={"level_0": "indicator A", "level_1": "indicator B", 0: "|r|"}
        ).to_markdown(index=False),
        "", "## Indicators with the most within-state variation", "",
        within.head(10).to_markdown(index=False),
        "", "## Top / bottom districts", tables,
    ]
    (REPORTS / "eda_summary.md").write_text("\n".join(lines), encoding="utf-8")
    within.to_csv(REPORTS / "within_state_variance.csv", index=False)
    print("  figures -> reports/figures/01..04, summary -> reports/eda_summary.md")


if __name__ == "__main__":
    main()
