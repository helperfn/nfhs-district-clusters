"""
PHASE 7 -- evaluation: does clustering actually beat the traditional groupings?

Every grouping (7 clusterings + 4 baselines + PCA variants) is reduced to the
same thing - one label per district - and then scored on five things:

  1. INTERNAL QUALITY        silhouette / Davies-Bouldin / Calinski-Harabasz,
                             all computed in the same 77-feature space so the
                             numbers are comparable across methods.

  2. HELD-OUT OUTCOME TEST   the key fairness test. Stunting, wasting,
     (eta-squared)           underweight, overweight and the five anaemia
                             indicators were never shown to ANY grouping. For
                             each one we run a one-way ANOVA across the groups
                             and record eta-squared = the share of that
                             outcome's variance explained by the grouping.
                             A grouping that carves districts into genuinely
                             different situations should separate outcomes it
                             has never seen. A grouping that only reshuffles
                             districts will not.

  3. PROFILE DISTINCTNESS    how many of the 77 features differ significantly
                             between groups (Kruskal-Wallis, Bonferroni-corrected
                             p < 0.01). Kruskal-Wallis rather than ANOVA because
                             several indicators are skewed.

  4. STABILITY               50 bootstrap samples of 80% of districts. For
                             clusterings: refit and compare with the full-data
                             labels using the Adjusted Rand Index. For the index
                             baselines: rebuild the index on the subsample and
                             measure both ARI of the quantile groups and the
                             Spearman rank correlation of the scores. State and
                             region grouping are deterministic, so their
                             stability is 1.0 by construction - noted, not hidden.

  5. AGREEMENT               ARI between every pair of groupings, as a heatmap.

Plus the evidence the whole project rests on: pairs of districts with nearly
identical composite scores that land in different clusters, with their domain
profiles drawn side by side.

Run:  python src/evaluate.py     (after cluster.py and baselines.py)
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)
from sklearn.mixture import GaussianMixture

from common import (CONFIG, DATA_PROC, FIGURES, MODELS, REPORTS, SEED, banner,
                    load_json)

sns.set_theme(style="whitegrid", context="talk")

N_BOOTSTRAP = 50
BOOTSTRAP_FRACTION = 0.8
SCORE_TOLERANCE = 0.02      # "same composite score" means within this much


# --------------------------------------------------------------------------
# metric helpers
# --------------------------------------------------------------------------
def eta_squared(values: np.ndarray, labels: np.ndarray) -> float:
    """
    Share of a variable's variance explained by a grouping (one-way ANOVA).

        eta^2 = SS_between / SS_total,  between 0 and 1.

    0.30 means the grouping accounts for 30% of the differences between
    districts on that outcome.
    """
    mask = labels >= 0
    v, l = values[mask], labels[mask]
    grand = v.mean()
    ss_total = ((v - grand) ** 2).sum()
    if ss_total == 0:
        return np.nan
    ss_between = sum(((v[l == g].mean() - grand) ** 2) * (l == g).sum() for g in np.unique(l))
    return float(ss_between / ss_total)


def internal_quality(Z: np.ndarray, labels: np.ndarray) -> dict:
    mask = labels >= 0
    if len(np.unique(labels[mask])) < 2:
        return dict(silhouette=np.nan, davies_bouldin=np.nan, calinski_harabasz=np.nan)
    return dict(
        silhouette=silhouette_score(Z[mask], labels[mask]),
        davies_bouldin=davies_bouldin_score(Z[mask], labels[mask]),
        calinski_harabasz=calinski_harabasz_score(Z[mask], labels[mask]),
    )


def distinct_features(X: pd.DataFrame, labels: np.ndarray) -> int:
    """Number of features that differ significantly between groups."""
    mask = labels >= 0
    groups = [g for g in np.unique(labels[mask])]
    if len(groups) < 2:
        return 0
    alpha = 0.01 / X.shape[1]          # Bonferroni: 77 tests, so tighten the bar
    hits = 0
    for col in X.columns:
        samples = [X.loc[mask, col].to_numpy()[labels[mask] == g] for g in groups]
        samples = [s for s in samples if len(s) > 1]
        if len(samples) < 2:
            continue
        try:
            if stats.kruskal(*samples).pvalue < alpha:
                hits += 1
        except ValueError:             # identical values in every group
            continue
    return hits


# --------------------------------------------------------------------------
# stability
# --------------------------------------------------------------------------
def fit_one(spec: dict, Xfit: np.ndarray) -> np.ndarray:
    """Fit the algorithm described by `spec` and return labels."""
    algo, k = spec["algo"], spec["k"]
    if algo == "kmeans":
        return KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(Xfit).labels_
    if algo == "ward":
        return AgglomerativeClustering(n_clusters=k, linkage="ward").fit(Xfit).labels_
    if algo == "gmm":
        cov = "full" if spec["space"] != "raw" else "diag"
        return GaussianMixture(n_components=k, covariance_type=cov, n_init=5,
                               random_state=SEED).fit(Xfit).predict(Xfit)
    if algo == "dbscan":
        from sklearn.cluster import DBSCAN
        return DBSCAN(eps=spec["eps"], min_samples=spec["min_samples"]).fit_predict(Xfit)
    raise ValueError(algo)


def stability(name: str, labels: np.ndarray, spec: dict | None, spaces: dict,
              good: pd.DataFrame, k: int) -> tuple[float, float, str]:
    """
    Returns (mean ARI, sd ARI, note).

    Index baselines are resampled the same way and re-scored, so the comparison
    is like-for-like: how much does the grouping change when 20% of districts
    are removed?
    """
    rng = np.random.default_rng(SEED)
    n = len(labels)
    size = int(BOOTSTRAP_FRACTION * n)
    scores, rank_corrs = [], []

    for _ in range(N_BOOTSTRAP):
        idx = rng.choice(n, size, replace=False)
        if name in ("state", "region"):
            scores.append(1.0)                     # deterministic by definition
            continue
        if name in ("composite_index", "domain_index"):
            sub = good.iloc[idx]
            rescaled = (sub - sub.min()) / (sub.max() - sub.min())
            score = rescaled.mean(axis=1)
            new = pd.qcut(score.rank(method="first"), q=k, labels=False).to_numpy()
            scores.append(adjusted_rand_score(labels[idx], new))
            rank_corrs.append(stats.spearmanr(score, labels[idx]).statistic)
            continue
        new = fit_one(spec, spaces[spec["space"]][idx])
        scores.append(adjusted_rand_score(labels[idx], new))

    note = ""
    if rank_corrs:
        note = f"rank stability (Spearman of score vs full-data group) = {np.mean(rank_corrs):.2f}"
    if name in ("state", "region"):
        note = "deterministic grouping - stability is 1.0 by construction"
    return float(np.nanmean(scores)), float(np.nanstd(scores)), note


# --------------------------------------------------------------------------
# the headline evidence
# --------------------------------------------------------------------------
def same_score_different_problems(ids: pd.DataFrame, base: pd.DataFrame,
                                  labels: np.ndarray, by_domain: pd.DataFrame,
                                  names: dict) -> pd.DataFrame:
    """
    Find districts with (almost) the same composite score that the clustering
    puts in different profiles, and draw their domain profiles side by side.

    This is the project's core claim made visible: one number says two districts
    are equally well off; the profiles say they need different interventions.
    """
    df = ids.copy()
    df["score"] = base["composite_score"]
    df["cluster"] = labels
    df = df.sort_values("score").reset_index()          # 'index' = original row id

    pairs = []
    for i in range(len(df) - 1):
        for j in range(i + 1, min(i + 12, len(df))):
            if df.loc[j, "score"] - df.loc[i, "score"] > SCORE_TOLERANCE:
                break
            if df.loc[i, "cluster"] == df.loc[j, "cluster"]:
                continue
            a, b = df.loc[i, "index"], df.loc[j, "index"]
            gap = float(np.abs(by_domain.loc[a] - by_domain.loc[b]).max())
            pairs.append(dict(
                district_a=f"{df.loc[i, 'district']} ({df.loc[i, 'state']})",
                district_b=f"{df.loc[j, 'district']} ({df.loc[j, 'state']})",
                score_a=df.loc[i, "score"], score_b=df.loc[j, "score"],
                score_gap=abs(df.loc[i, "score"] - df.loc[j, "score"]),
                cluster_a=int(df.loc[i, "cluster"]), cluster_b=int(df.loc[j, "cluster"]),
                biggest_domain_gap=gap, row_a=int(a), row_b=int(b),
            ))
    out = pd.DataFrame(pairs).sort_values("biggest_domain_gap", ascending=False)

    # draw the three starkest examples
    top = out.head(3)
    if len(top):
        fig, axes = plt.subplots(len(top), 1, figsize=(15, 5.5 * len(top)))
        for ax, (_, row) in zip(np.atleast_1d(axes), top.iterrows()):
            dom = by_domain.columns
            width = 0.38
            x = np.arange(len(dom))
            ax.bar(x - width / 2, by_domain.loc[row["row_a"]], width,
                   label=f"{row['district_a']} - profile {names.get(row['cluster_a'], row['cluster_a'])}",
                   color="#2c7fb8")
            ax.bar(x + width / 2, by_domain.loc[row["row_b"]], width,
                   label=f"{row['district_b']} - profile {names.get(row['cluster_b'], row['cluster_b'])}",
                   color="#d95f0e")
            ax.set_xticks(x)
            ax.set_xticklabels(dom, rotation=30, ha="right", fontsize=9)
            ax.set_ylabel("domain score (1 = best)")
            ax.set_title(f"Composite scores {row['score_a']:.3f} vs {row['score_b']:.3f} "
                         f"(gap {row['score_gap']:.3f}) - but different profiles", fontsize=12)
            ax.legend(fontsize=9)
        fig.suptitle("Same score, different problems", fontsize=16)
        fig.tight_layout()
        fig.savefig(FIGURES / "15_same_score_different_problems.png", dpi=140)
        plt.close(fig)
    return out


# --------------------------------------------------------------------------
def main() -> None:
    banner("PHASE 7 - EVALUATION AND COMPARISON")
    X = pd.read_csv(DATA_PROC / "features.csv")
    Z = pd.read_csv(DATA_PROC / "features_scaled.csv").to_numpy()
    Y = pd.read_csv(DATA_PROC / "outcomes.csv")
    ids = pd.read_csv(DATA_PROC / "ids.csv")
    base = pd.read_csv(DATA_PROC / "baselines.csv")
    lab_all = pd.read_csv(DATA_PROC / "cluster_labels_all.csv")
    by_domain = pd.read_csv(DATA_PROC / "domain_scores_0_1.csv")
    cfg = pd.read_csv(CONFIG / "indicators.csv")
    k = load_json(CONFIG / "k_choice.json")["k"]
    P = pd.read_csv(DATA_PROC / "pca_scores_90.csv").to_numpy()
    good_cols = X.columns

    # orientation-corrected 0-1 features, needed for baseline stability
    direction = dict(zip(cfg["short_name"], cfg["direction"]))
    good = X.copy()
    for c in good.columns:
        if direction[c] < 0:
            good[c] = -good[c]
    good = (good - good.min()) / (good.max() - good.min())

    # ---- assemble every grouping -----------------------------------------
    # `spaces` holds each representation of the data; `specs` records how to
    # refit a grouping from scratch, which the bootstrap stability test needs.
    spaces = {"raw": Z, "pca90": P}
    for target in (80, 95):
        spaces[f"pca{target}"] = pd.read_csv(DATA_PROC / f"pca_scores_{target}.csv").to_numpy()

    dbscan_cfg = __import__("joblib").load(MODELS / "cluster_models.joblib")
    groupings: dict[str, np.ndarray] = {n: lab_all[n].to_numpy() for n in lab_all.columns}
    specs: dict[str, dict] = {}
    for n in lab_all.columns:
        algo, space = n.rsplit("_", 1)
        specs[n] = {"algo": algo, "space": "raw" if space == "raw" else "pca90", "k": k}
        if algo == "dbscan":
            specs[n].update(eps=dbscan_cfg["dbscan_eps"],
                            min_samples=dbscan_cfg["dbscan_min_samples"])

    groupings["composite_index"] = base["composite_group"].to_numpy()
    groupings["domain_index"] = base["domain_group"].to_numpy()
    groupings["state"] = base["state_group"].to_numpy()
    groupings["region"] = base["region_group"].to_numpy()

    # PCA-variance variants and the k=3 alternative, for the sensitivity check
    for target in (80, 95):
        specs[f"kmeans_pca{target}"] = {"algo": "kmeans", "space": f"pca{target}", "k": k}
        groupings[f"kmeans_pca{target}"] = fit_one(specs[f"kmeans_pca{target}"],
                                                   spaces[f"pca{target}"])
    specs["kmeans_pca_k3"] = {"algo": "kmeans", "space": "pca90", "k": 3}
    groupings["kmeans_pca_k3"] = fit_one(specs["kmeans_pca_k3"], P)

    # ---- score them -------------------------------------------------------
    Yz = (Y - Y.mean()) / Y.std()
    rows = []
    for name, labels in groupings.items():
        q = internal_quality(Z, labels)
        etas = {f"eta2_{c}": eta_squared(Yz[c].to_numpy(), labels) for c in Y.columns}
        mean_ari, sd_ari, note = stability(name, labels, specs.get(name), spaces, good, k)
        rows.append(dict(
            grouping=name,
            n_groups=int(len(np.unique(labels[labels >= 0]))),
            smallest_group=int(pd.Series(labels[labels >= 0]).value_counts().min()),
            **q,
            mean_outcome_eta2=float(np.nanmean(list(etas.values()))),
            distinct_features=distinct_features(X, labels),
            stability_ari=mean_ari, stability_sd=sd_ari,
            note=note, **etas,
        ))
        print(f"  scored {name}")

    comp = pd.DataFrame(rows).sort_values("mean_outcome_eta2", ascending=False)
    comp.to_csv(REPORTS / "comparison.csv", index=False)

    banner("COMPARISON (sorted by held-out outcome eta-squared)")
    show = comp[["grouping", "n_groups", "silhouette", "davies_bouldin",
                 "mean_outcome_eta2", "distinct_features", "stability_ari"]]
    print(show.round(3).to_string(index=False))

    # ---- agreement heatmap -------------------------------------------------
    names = list(groupings)
    ari = pd.DataFrame(
        [[adjusted_rand_score(groupings[a], groupings[b]) for b in names] for a in names],
        index=names, columns=names)
    ari.to_csv(REPORTS / "ari_agreement.csv")
    plt.figure(figsize=(13, 11))
    sns.heatmap(ari, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0, vmax=1,
                annot_kws={"size": 7}, cbar_kws={"label": "Adjusted Rand Index"})
    plt.title("How much do the groupings agree with each other?\n"
              "1 = identical grouping, 0 = no more alike than chance", fontsize=13)
    plt.xticks(fontsize=8, rotation=90)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "16_ari_agreement.png", dpi=140)
    plt.close()

    # ---- headline comparison chart ----------------------------------------
    main_rows = comp[comp["grouping"].isin(
        ["kmeans_pca", "ward_pca", "gmm_pca", "kmeans_raw", "ward_raw", "gmm_raw",
         "dbscan_pca", "composite_index", "domain_index", "state", "region"])]
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    for ax, col, title in [
        (axes[0], "mean_outcome_eta2", "Held-out outcomes explained\n(mean eta-squared, higher = better)"),
        (axes[1], "distinct_features", "Features differing between groups\n(Kruskal-Wallis, Bonferroni p<0.01)"),
        (axes[2], "stability_ari", "Stability under 80% resampling\n(ARI, higher = better)"),
    ]:
        d = main_rows.sort_values(col, ascending=True)
        colours = ["#c7254e" if g in ("composite_index", "domain_index", "state", "region")
                   else "#2c7fb8" for g in d["grouping"]]
        ax.barh(d["grouping"], d[col], color=colours)
        ax.set_title(title, fontsize=12)
        ax.tick_params(labelsize=9)
    fig.suptitle("Clustering (blue) vs traditional groupings (red)", fontsize=15)
    fig.tight_layout()
    fig.savefig(FIGURES / "17_comparison.png", dpi=140)
    plt.close(fig)

    # ---- final choice -----------------------------------------------------
    # Rule, in the order set out in the project plan:
    #   held-out outcome eta-squared -> stability -> silhouette -> interpretability.
    # With one refinement: differences in eta-squared of less than ETA_TOLERANCE
    # are treated as ties, because with ten outcomes and 705 districts a gap that
    # small is inside the noise. Among the tied leaders we take the most stable
    # model - a grouping nobody can reproduce is not usable policy advice.
    ETA_TOLERANCE = 0.02
    MIN_STABILITY_ARI = 0.60      # the same floor cluster.py used to choose k

    ranked = comp[~comp["grouping"].isin(
        ["composite_index", "domain_index", "state", "region"])].copy()
    ranked = ranked[(ranked["smallest_group"] >= 15) & (ranked["n_groups"] >= 3)]

    # Stability floor. Phase 6 already refused to accept a k whose clustering
    # falls apart when 20% of districts are removed; the same standard has to
    # apply to the choice of algorithm. Without it the rule picks whichever model
    # squeezes out the highest eta-squared even when it is barely reproducible -
    # and an unreproducible grouping also fails the unseen-state test in Phase 9,
    # because "refit without one state" is the same question as "refit without
    # 20% of districts".
    stable = ranked[ranked["stability_ari"] >= MIN_STABILITY_ARI]
    if stable.empty:              # nothing clears the floor: fall back, and say so
        print(f"  WARNING: no clustering reached stability {MIN_STABILITY_ARI}; "
              "selecting on eta-squared alone")
        stable = ranked
    dropped_for_instability = sorted(set(ranked["grouping"]) - set(stable["grouping"]))
    if dropped_for_instability:
        print(f"  excluded for stability < {MIN_STABILITY_ARI}: {dropped_for_instability}")

    tied = stable[stable["mean_outcome_eta2"] >= stable["mean_outcome_eta2"].max() - ETA_TOLERANCE]
    chosen = tied.sort_values(["stability_ari", "silhouette"], ascending=False).iloc[0]
    chosen_spec = specs[chosen["grouping"]]

    # ---- same score, different problems -----------------------------------
    final_labels = groupings[chosen["grouping"]]
    pairs = same_score_different_problems(ids, base, final_labels, by_domain, {})
    pairs.to_csv(REPORTS / "same_score_different_problems.csv", index=False)
    print(f"\n  district pairs within {SCORE_TOLERANCE} composite score but in "
          f"different clusters: {len(pairs)}")

    best_baseline = comp[comp["grouping"].isin(
        ["composite_index", "domain_index", "state", "region"])].iloc[0]

    lines = [
        "# Phase 7 - final model choice", "",
        f"**Chosen model: `{chosen['grouping']}` with k = {chosen['n_groups']}.**", "",
        "Selection order, fixed before the numbers were looked at: a stability floor "
        f"(mean bootstrap ARI >= {MIN_STABILITY_ARI}, the same floor used to choose k), "
        "then held-out outcome eta-squared, then stability, then silhouette, then "
        "interpretability. Groupings with any cluster under 15 districts were excluded."
        + (f" Excluded for instability: {', '.join(dropped_for_instability)}."
           if dropped_for_instability else ""), "",
        "## The numbers that decided it", "",
        show.round(3).to_markdown(index=False), "",
        "## Where the traditional baselines do as well or better", "",
        f"- The strongest baseline is **{best_baseline['grouping']}** "
        f"(mean outcome eta-squared {best_baseline['mean_outcome_eta2']:.3f} vs "
        f"{chosen['mean_outcome_eta2']:.3f} for the chosen clustering).",
        f"- State grouping explains {comp.set_index('grouping').loc['state', 'mean_outcome_eta2']:.3f} "
        "of held-out outcome variance - with 37 groups, far more than the 7 the clustering uses. "
        "Any grouping with more groups has an arithmetic advantage on eta-squared, so this "
        "comparison flatters state grouping and is reported as such.",
        f"- The composite index cut into {k} quantiles explains "
        f"{comp.set_index('grouping').loc['composite_index', 'mean_outcome_eta2']:.3f}. "
        "That is the like-for-like comparison (same number of groups), and it is the one "
        "the project's claim stands or falls on.", "",
        "## Sensitivity to the PCA cut-off", "",
        comp[comp["grouping"].str.startswith("kmeans_pca")][
            ["grouping", "n_groups", "silhouette", "mean_outcome_eta2", "stability_ari"]
        ].round(3).to_markdown(index=False), "",
        "## Same score, different problems", "",
        f"{len(pairs)} pairs of districts sit within {SCORE_TOLERANCE} of each other on the "
        "composite index yet fall in different clusters. The three with the largest domain "
        "gap are plotted in `reports/figures/15_same_score_different_problems.png`.", "",
        pairs.head(10)[["district_a", "district_b", "score_a", "score_b",
                        "cluster_a", "cluster_b", "biggest_domain_gap"]].round(3).to_markdown(index=False),
    ]
    (REPORTS / "final_choice.md").write_text("\n".join(lines), encoding="utf-8")

    # ---- store the final model so Phases 8-10 use exactly this pipeline ---
    import joblib
    from common import save_json

    space = chosen_spec["space"]
    estimator = {"kmeans": KMeans(n_clusters=chosen_spec["k"], n_init=50, random_state=SEED),
                 "ward": AgglomerativeClustering(n_clusters=chosen_spec["k"], linkage="ward"),
                 "gmm": GaussianMixture(n_components=chosen_spec["k"],
                                        covariance_type="full" if space != "raw" else "diag",
                                        n_init=10, random_state=SEED)}[chosen_spec["algo"]]
    estimator.fit(spaces[space])

    pd.DataFrame({"cluster": final_labels}).to_csv(DATA_PROC / "final_labels.csv", index=False)
    joblib.dump({"estimator": estimator, "spec": chosen_spec,
                 "n_components": spaces[space].shape[1] if space != "raw" else None},
                MODELS / "final_model.joblib")
    save_json({"final_grouping": chosen["grouping"], "k": int(chosen["n_groups"]),
               "space": space, "algo": chosen_spec["algo"],
               "n_components": None if space == "raw" else int(spaces[space].shape[1]),
               "pca_scores_file": None if space == "raw" else f"pca_scores_{space[3:]}.csv",
               "selection_rule": f"stability floor (ARI >= {MIN_STABILITY_ARI}) -> "
                                 "held-out outcome eta-squared (ties within 0.02) -> "
                                 "stability -> silhouette; clusters >= 15 districts"},
              CONFIG / "final_model.json")
    banner(f"FINAL MODEL: {chosen['grouping']} (k = {int(chosen['n_groups'])})")
    print("  reports/comparison.csv, reports/final_choice.md, figures 15-17 written")


if __name__ == "__main__":
    main()
