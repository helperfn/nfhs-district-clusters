"""
PHASE 4 -- standardise the features and run PCA.

Why standardise:
  the features are on different scales (percentages 0-100, sex ratio around
  950, out-of-pocket expenditure in thousands of rupees). Distance-based
  methods - which is all of K-means, Ward, GMM and DBSCAN - would otherwise be
  dominated by whichever column happens to have the biggest numbers.

Why PCA:
  Phase 3 showed the indicators are heavily correlated (115 pairs above
  |r| = 0.6). Correlated columns give the same underlying concept several votes
  in the distance calculation, and in 77 dimensions distances between districts
  all start to look alike ("curse of dimensionality"). PCA rotates the data onto
  uncorrelated axes ordered by how much variance they explain, so we can keep
  the signal in far fewer dimensions.

Outputs: fitted scaler + PCA (joblib), PCA scores at the 80 / 90 / 95% variance
cut-offs, scree and cumulative-variance plots, a loadings heatmap and a
plain-English description of the first five components.

Run:  python src/features.py
"""
from __future__ import annotations

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from common import (CONFIG, DATA_PROC, FIGURES, MODELS, REPORTS, SEED, banner,
                    save_json)

sns.set_theme(style="whitegrid", context="talk")

VARIANCE_TARGETS = (0.80, 0.90, 0.95)
MAIN_TARGET = 0.90          # the "90% rule" used for the headline pipeline


def n_components_for(evr_cumulative: np.ndarray, target: float) -> int:
    """Smallest number of components whose cumulative variance reaches `target`."""
    return int(np.searchsorted(evr_cumulative, target) + 1)


def plot_variance(pca: PCA, ks: dict[float, int]) -> None:
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    axes[0].bar(range(1, len(evr) + 1), evr * 100, color="#2c7fb8")
    axes[0].set_xlim(0, 30)
    axes[0].set_xlabel("component")
    axes[0].set_ylabel("variance explained (%)")
    axes[0].set_title("Scree plot (first 30 components)")

    axes[1].plot(range(1, len(cum) + 1), cum * 100, marker="o", ms=3, color="#253494")
    for target, colour in zip(VARIANCE_TARGETS, ["#d95f0e", "#c7254e", "#31a354"]):
        axes[1].axhline(target * 100, ls="--", lw=1.2, color=colour,
                        label=f"{target:.0%} -> {ks[target]} components")
    axes[1].set_xlim(0, 60)
    axes[1].set_xlabel("number of components")
    axes[1].set_ylabel("cumulative variance explained (%)")
    axes[1].set_title("How many components do we need?")
    axes[1].legend(fontsize=11)

    fig.tight_layout()
    fig.savefig(FIGURES / "05_pca_variance.png", dpi=140)
    plt.close(fig)


def plot_loadings(pca: PCA, cols: list[str], cfg: pd.DataFrame) -> pd.DataFrame:
    """Heatmap of how each original indicator contributes to PC1..PC5."""
    load = pd.DataFrame(pca.components_[:5].T, index=cols,
                        columns=[f"PC{i}" for i in range(1, 6)])
    order = (cfg[cfg["short_name"].isin(cols)]
             .sort_values(["domain", "short_name"])["short_name"].tolist())
    plt.figure(figsize=(10, 20))
    sns.heatmap(load.loc[order], cmap="RdBu_r", center=0, yticklabels=True,
                cbar_kws={"label": "loading"})
    plt.yticks(fontsize=6)
    plt.title("PCA loadings, first 5 components\n(indicators grouped by domain)", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURES / "06_pca_loadings.png", dpi=140)
    plt.close()
    return load


def describe_components(load: pd.DataFrame, pca: PCA, cfg: pd.DataFrame) -> list[str]:
    """
    Turn each component into a plain-English name.

    Method: look at which DOMAINS dominate the strongest loadings, and whether
    the +1-direction ("higher is better") indicators sit on the positive or the
    negative side. That is enough to write an honest label such as
    'overall household development' without hand-waving.
    """
    direction = dict(zip(cfg["short_name"], cfg["direction"]))
    domain = dict(zip(cfg["short_name"], cfg["domain_label"]))
    lines = ["# PCA components in plain English", ""]
    names = {}
    for pc in load.columns:
        s = load[pc]
        top = s.abs().sort_values(ascending=False).head(12).index
        pos = [i for i in top if s[i] > 0]
        neg = [i for i in top if s[i] < 0]
        doms = pd.Series([domain[i] for i in top]).value_counts()
        # does the positive side of this axis mean "better"?
        signed = np.mean([np.sign(s[i]) * direction[i] for i in top])
        polarity = ("higher score = better outcomes" if signed > 0.25
                    else "higher score = worse outcomes" if signed < -0.25
                    else "mixed: not a simple good/bad axis")
        name = f"{doms.index[0]}"
        if len(doms) > 1 and doms.iloc[1] >= doms.iloc[0] - 1:
            name += f" + {doms.index[1]}"
        names[pc] = name
        lines += [
            f"## {pc} - {name}",
            f"*variance explained:* {pca.explained_variance_ratio_[int(pc[2:]) - 1]:.1%}  ",
            f"*polarity:* {polarity}  ",
            f"*dominant domains:* " + ", ".join(f"{d} ({n})" for d, n in doms.items()),
            "",
            "| indicator | loading |", "|---|---|",
            *[f"| {i} | {s[i]:+.2f} |" for i in top[:8]],
            "",
            f"Positive side: {', '.join(pos[:5]) or '-'}  ",
            f"Negative side: {', '.join(neg[:5]) or '-'}",
            "",
        ]
    save_json(names, CONFIG / "pca_component_names.json")
    return lines


def plot_scatter(scores: np.ndarray, ids: pd.DataFrame) -> None:
    """PC1 vs PC2, coloured by state, plus a Karnataka-highlighted version."""
    df = ids.copy()
    df["PC1"], df["PC2"] = scores[:, 0], scores[:, 1]

    plt.figure(figsize=(14, 11))
    big = df["state"].value_counts().head(12).index
    sns.scatterplot(data=df[df["state"].isin(big)], x="PC1", y="PC2", hue="state",
                    palette="tab20", s=45, alpha=0.85)
    sns.scatterplot(data=df[~df["state"].isin(big)], x="PC1", y="PC2",
                    color="#cccccc", s=20, alpha=0.5, label="other states")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    plt.title("Districts in PCA space, coloured by state\n"
              "states overlap heavily - state is not the same thing as a development profile",
              fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURES / "07_pca_scatter_states.png", dpi=140)
    plt.close()

    plt.figure(figsize=(12, 10))
    plt.scatter(df["PC1"], df["PC2"], s=25, color="#d9d9d9", label="all districts")
    k = df[df["state"].str.contains("Karnataka", case=False, na=False)]
    plt.scatter(k["PC1"], k["PC2"], s=70, color="#c7254e", label="Karnataka")
    for _, r in k.iterrows():
        plt.annotate(r["district"], (r["PC1"], r["PC2"]), fontsize=7,
                     xytext=(3, 3), textcoords="offset points")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.title("Karnataka districts highlighted in PCA space", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURES / "08_pca_scatter_karnataka.png", dpi=140)
    plt.close()


def main() -> None:
    banner("PHASE 4 - SCALING AND PCA")
    X = pd.read_csv(DATA_PROC / "features.csv")
    ids = pd.read_csv(DATA_PROC / "ids.csv")
    cfg = pd.read_csv(CONFIG / "indicators.csv")

    scaler = StandardScaler().fit(X)
    Z = scaler.transform(X)
    pd.DataFrame(Z, columns=X.columns).to_csv(DATA_PROC / "features_scaled.csv", index=False)

    pca = PCA(random_state=SEED).fit(Z)
    cum = np.cumsum(pca.explained_variance_ratio_)
    ks = {t: n_components_for(cum, t) for t in VARIANCE_TARGETS}

    print(f"  features                     : {X.shape[1]}")
    for t, k in ks.items():
        print(f"  components for {t:.0%} variance : {k}")
    print(f"  PC1 alone explains           : {pca.explained_variance_ratio_[0]:.1%}")
    print(f"  PC1-PC5 explain              : {cum[4]:.1%}")

    plot_variance(pca, ks)
    load = plot_loadings(pca, X.columns.tolist(), cfg)
    lines = describe_components(load, pca, cfg)

    scores_full = pca.transform(Z)
    plot_scatter(scores_full, ids)

    # Save the PCA scores at each variance target; Phase 7 compares them.
    for t, k in ks.items():
        pd.DataFrame(scores_full[:, :k], columns=[f"PC{i+1}" for i in range(k)]).to_csv(
            DATA_PROC / f"pca_scores_{int(t*100)}.csv", index=False)

    joblib.dump({"scaler": scaler, "pca": pca, "k_by_target": ks,
                 "k_main": ks[MAIN_TARGET], "columns": X.columns.tolist()},
                MODELS / "scaler_pca.joblib")

    header = [
        f"*{X.shape[1]} standardised indicators, {X.shape[0]} districts.*", "",
        "| variance target | components kept |", "|---|---|",
        *[f"| {t:.0%} | {k} |" for t, k in ks.items()], "",
        f"The project uses the **90% rule -> {ks[0.90]} components** for the main pipeline; "
        "80% and 95% are tested in Phase 7 to show the choice is not doing the work.", "",
    ]
    (REPORTS / "pca_components.md").write_text("\n".join(lines[:1] + [""] + header + lines[1:]),
                                               encoding="utf-8")
    print("  saved models/scaler_pca.joblib, reports/pca_components.md, figures 05-08")


if __name__ == "__main__":
    main()
