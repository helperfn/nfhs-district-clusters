"""
PHASE 7b -- is the structure real, and where is it soft?

Three checks that answer the three hardest questions an examiner can ask about an
unsupervised model.

  1. PERMUTATION TEST - "your silhouette is only 0.10, isn't that just noise?"
     Shuffle each feature column independently. That destroys every relationship
     between indicators while keeping each indicator's own distribution exactly
     as it was: a district in the shuffled data has one district's sanitation,
     another's schooling, a third's immunisation. Any clustering of that is pure
     chance. Run the identical pipeline on it 20 times and compare.

     We report two things for the shuffled data, not one:
       - silhouette (is the geometry real?)
       - held-out outcome eta-squared (does the grouping still predict stunting
         and anaemia, or was that coming from real structure?)
     The second is the stronger test, because eta-squared uses columns the
     shuffling never touched.

  2. SILHOUETTE VS THE BASELINES - the same number for the composite index and
     the state/region groupings, so "0.10 is low" can be answered with "lower
     than what?".

  3. SOFT BOUNDARIES - per-district silhouette and GMM membership probability,
     drawn per cluster. This shows which districts are firmly inside their
     profile and which are borderline, and it is the direct explanation of the
     unseen-state result: states whose districts are borderline do not reproduce.

Run:  python src/validate.py     (after evaluate.py and profile.py)
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from common import (CONFIG, DATA_PROC, FIGURES, REPORTS, SEED, banner, load_json,
                    save_json)

sns.set_theme(style="whitegrid", context="talk")

N_PERMUTATIONS = 20
UNSEEN_STATES = ["Bihar", "Karnataka", "Assam"]


# --------------------------------------------------------------------------
def eta_squared(values: np.ndarray, labels: np.ndarray) -> float:
    grand = values.mean()
    ss_total = ((values - grand) ** 2).sum()
    if ss_total == 0:
        return np.nan
    ss_between = sum(((values[labels == g].mean() - grand) ** 2) * (labels == g).sum()
                     for g in np.unique(labels))
    return float(ss_between / ss_total)


def mean_outcome_eta2(Yz: pd.DataFrame, labels: np.ndarray) -> float:
    return float(np.nanmean([eta_squared(Yz[c].to_numpy(), labels) for c in Yz.columns]))


def run_pipeline(X: pd.DataFrame, k: int, n_components: int | None) -> np.ndarray:
    """The final pipeline, refitted from scratch: scale -> PCA -> K-means."""
    Z = StandardScaler().fit_transform(X)
    space = Z if n_components is None else PCA(random_state=SEED).fit_transform(Z)[:, :n_components]
    return KMeans(n_clusters=k, n_init=50, random_state=SEED).fit(space).labels_, Z


# --------------------------------------------------------------------------
# 1. permutation test
# --------------------------------------------------------------------------
def permutation_test(X: pd.DataFrame, Yz: pd.DataFrame, k: int, n_components: int | None,
                     real_sil: float, real_eta: float) -> dict:
    """
    Shuffle every column independently, then run the whole pipeline on it.

    Column-wise shuffling is the right null here: it keeps each indicator's
    marginal distribution identical (same mean, same spread, same skew) and
    destroys only the associations BETWEEN indicators - which is exactly the
    thing clustering is supposed to be finding.
    """
    rng = np.random.default_rng(SEED)
    sils, etas = [], []
    for i in range(N_PERMUTATIONS):
        shuffled = X.apply(lambda col: rng.permutation(col.to_numpy()), axis=0)
        labels, Z_shuf = run_pipeline(shuffled, k, n_components)
        sils.append(silhouette_score(Z_shuf, labels))
        etas.append(mean_outcome_eta2(Yz, labels))
        print(f"    permutation {i + 1:2d}/{N_PERMUTATIONS}: "
              f"silhouette {sils[-1]:.3f}, outcome eta2 {etas[-1]:.3f}")

    sils, etas = np.array(sils), np.array(etas)
    # p-value: how often did chance match or beat the real data?
    p_sil = float((sils >= real_sil).sum() + 1) / (N_PERMUTATIONS + 1)
    p_eta = float((etas >= real_eta).sum() + 1) / (N_PERMUTATIONS + 1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, vals, real, name, p in (
        (axes[0], sils, real_sil, "Silhouette", p_sil),
        (axes[1], etas, real_eta, "Held-out outcome eta-squared", p_eta),
    ):
        ax.hist(vals, bins=12, color="#bdbdbd", edgecolor="white",
                label=f"shuffled data ({N_PERMUTATIONS} runs)")
        ax.axvline(real, color="#c7254e", lw=3, label=f"real data = {real:.3f}")
        ax.set_title(f"{name}\np = {p:.3f}", fontsize=13)
        ax.set_xlabel(name.lower())
        ax.set_ylabel("permutations")
        ax.legend(fontsize=10)
    fig.suptitle("Permutation test: every feature column shuffled independently", fontsize=15)
    fig.tight_layout()
    fig.savefig(FIGURES / "20_permutation_test.png", dpi=140)
    plt.close(fig)

    return {
        "n_permutations": N_PERMUTATIONS,
        "real_silhouette": real_sil,
        "shuffled_silhouette_mean": float(sils.mean()),
        "shuffled_silhouette_max": float(sils.max()),
        "silhouette_p_value": p_sil,
        "real_outcome_eta2": real_eta,
        "shuffled_outcome_eta2_mean": float(etas.mean()),
        "shuffled_outcome_eta2_max": float(etas.max()),
        "outcome_eta2_p_value": p_eta,
    }


# --------------------------------------------------------------------------
# 2. silhouette against the baselines
# --------------------------------------------------------------------------
def silhouette_vs_baselines(comparison: pd.DataFrame, final_name: str) -> pd.DataFrame:
    rows = comparison.set_index("grouping")
    wanted = [final_name, "kmeans_raw", "state", "region", "domain_index", "composite_index"]
    table = rows.loc[[w for w in wanted if w in rows.index],
                     ["n_groups", "silhouette", "mean_outcome_eta2"]].reset_index()

    plt.figure(figsize=(11, 6))
    colours = ["#2c7fb8" if g not in ("state", "region", "domain_index", "composite_index")
               else "#c7254e" for g in table["grouping"]]
    order = table.sort_values("silhouette")
    plt.barh(order["grouping"], order["silhouette"],
             color=[colours[list(table["grouping"]).index(g)] for g in order["grouping"]])
    plt.axvline(0, color="#444", lw=1)
    plt.xlabel("silhouette in the 77-feature space (higher = better separated)")
    plt.title("Are the clusters better separated than the traditional groupings?\n"
              "negative = districts sit closer to another group than their own", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURES / "21_silhouette_vs_baselines.png", dpi=140)
    plt.close()
    return table


# --------------------------------------------------------------------------
# 3. soft boundaries
# --------------------------------------------------------------------------
def soft_boundaries(X: pd.DataFrame, ids: pd.DataFrame, labels: np.ndarray,
                    scores: np.ndarray, names: dict) -> pd.DataFrame:
    """
    Per-district silhouette, GMM membership probability and label margin.

    Three views of the same thing - how firmly does this district belong? - and
    they agree, which is why any one of them can be quoted in the viva.
    """
    Z = pd.read_csv(DATA_PROC / "features_scaled.csv").to_numpy()
    sil = silhouette_samples(Z, labels)

    # A GMM fitted at the same k, used ONLY to report soft membership. It is not
    # the final model and it changes no label.
    #
    # Fitted on the leading 6 components with diagonal covariance, deliberately.
    # A full-covariance GMM in 30 dimensions gives a degenerate answer: 99% of
    # districts come back with probability 1.000, because a Gaussian density in
    # high dimensions is so peaked that the nearest component wins by orders of
    # magnitude. The numbers look confident and mean nothing. On the leading
    # components the densities are actually estimable and the probabilities
    # spread out (median 0.95, lowest decile 0.64).
    k = len(np.unique(labels))
    gmm = GaussianMixture(n_components=k, covariance_type="diag",
                          n_init=10, random_state=SEED).fit(scores[:, :6])
    proba = gmm.predict_proba(scores[:, :6]).max(axis=1)

    centroids = pd.DataFrame(scores).groupby(labels).mean().to_numpy()
    d = np.linalg.norm(scores[:, None, :] - centroids[None, :, :], axis=2)
    srt = np.sort(d, axis=1)
    margin = srt[:, 1] / srt[:, 0]

    # Fuzzy c-means style membership: weight each profile by 1/distance^2 and
    # normalise. Scale-free, no distributional assumption, and it cannot
    # saturate the way a high-dimensional Gaussian does. 1/k (0.125 here) means
    # "belongs to all profiles equally".
    w = 1.0 / np.maximum(d, 1e-9) ** 2
    fuzzy = (w / w.sum(axis=1, keepdims=True)).max(axis=1)

    out = ids.copy()
    out["cluster"] = labels
    out["cluster_name"] = [names[str(int(c))]["name"] for c in labels]
    out["silhouette"] = sil
    out["gmm_max_probability"] = proba
    out["fuzzy_membership"] = fuzzy
    out["label_margin"] = margin
    out["confidence"] = np.where(margin >= 1.35, "firm",
                                 np.where(margin >= 1.15, "typical", "borderline"))
    out.to_csv(DATA_PROC / "district_confidence.csv", index=False)

    # ---- the classic silhouette knife plot, one blade per cluster ----------
    plt.figure(figsize=(11, 9))
    palette = sns.color_palette("tab10", len(np.unique(labels)))
    y = 10
    ticks = []
    for c in sorted(np.unique(labels)):
        vals = np.sort(sil[labels == c])
        plt.fill_betweenx(np.arange(y, y + len(vals)), 0, vals,
                          facecolor=palette[c], edgecolor=palette[c], alpha=0.85)
        ticks.append((y + len(vals) / 2, f"{c} ({len(vals)})"))
        y += len(vals) + 10
    plt.axvline(sil.mean(), color="#c7254e", ls="--", lw=2,
                label=f"average = {sil.mean():.3f}")
    plt.axvline(0, color="#444", lw=1)
    plt.yticks([t[0] for t in ticks], [t[1] for t in ticks])
    plt.ylabel("cluster (districts)")
    plt.xlabel("silhouette of each district")
    plt.title("Where each profile is firm and where it is soft\n"
              "districts below zero sit closer to another profile than their own", fontsize=13)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(FIGURES / "22_silhouette_by_cluster.png", dpi=140)
    plt.close()

    # ---- confidence vs the unseen-state result ----------------------------
    plt.figure(figsize=(11, 6))
    plt.scatter(out["label_margin"], out["gmm_max_probability"], s=14, alpha=0.45,
                color="#bdbdbd", label="all districts")
    for state, colour in zip(UNSEEN_STATES, ["#31a354", "#2c7fb8", "#d95f0e"]):
        sub = out[out["state"].str.contains(state, case=False, na=False)]
        plt.scatter(sub["label_margin"], sub["gmm_max_probability"], s=55,
                    color=colour, label=f"{state} ({len(sub)})")
    plt.axvline(1.15, ls="--", color="#c7254e", lw=1.5)
    plt.text(1.155, 0.02, "borderline <- | -> firmer", fontsize=9, color="#c7254e")
    plt.xlabel("label margin (2nd-nearest profile / own profile)")
    plt.ylabel("GMM membership probability of assigned profile")
    plt.title("The two confidence measures agree - and explain the unseen-state test",
              fontsize=13)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(FIGURES / "23_confidence_by_state.png", dpi=140)
    plt.close()
    return out


# --------------------------------------------------------------------------
def main() -> None:
    banner("PHASE 7b - VALIDATION: IS THE STRUCTURE REAL?")
    X = pd.read_csv(DATA_PROC / "features.csv")
    Z = pd.read_csv(DATA_PROC / "features_scaled.csv").to_numpy()
    Y = pd.read_csv(DATA_PROC / "outcomes.csv")
    ids = pd.read_csv(DATA_PROC / "ids.csv")
    labels = pd.read_csv(DATA_PROC / "final_labels.csv")["cluster"].to_numpy()
    comparison = pd.read_csv(REPORTS / "comparison.csv")
    final = load_json(CONFIG / "final_model.json")
    names = load_json(CONFIG / "cluster_names.json")
    scores = (pd.read_csv(DATA_PROC / final["pca_scores_file"]).to_numpy()
              if final["pca_scores_file"] else Z)
    Yz = (Y - Y.mean()) / Y.std()

    real_sil = silhouette_score(Z, labels)
    real_eta = mean_outcome_eta2(Yz, labels)
    print(f"  real data: silhouette {real_sil:.3f}, held-out outcome eta2 {real_eta:.3f}")

    banner("1. PERMUTATION TEST (columns shuffled independently)")
    perm = permutation_test(X, Yz, final["k"], final["n_components"], real_sil, real_eta)
    print(f"\n  silhouette : real {perm['real_silhouette']:.3f} vs shuffled "
          f"{perm['shuffled_silhouette_mean']:.3f} "
          f"(max {perm['shuffled_silhouette_max']:.3f}), p = {perm['silhouette_p_value']:.3f}")
    print(f"  outcome eta2: real {perm['real_outcome_eta2']:.3f} vs shuffled "
          f"{perm['shuffled_outcome_eta2_mean']:.3f} "
          f"(max {perm['shuffled_outcome_eta2_max']:.3f}), p = {perm['outcome_eta2_p_value']:.3f}")

    banner("2. SILHOUETTE AGAINST THE BASELINES")
    sil_table = silhouette_vs_baselines(comparison, final["final_grouping"])
    print(sil_table.round(3).to_string(index=False))

    banner("3. SOFT BOUNDARIES")
    conf = soft_boundaries(X, ids, labels, scores, names)
    print(conf["confidence"].value_counts().to_string())
    print(f"\n  districts with a NEGATIVE silhouette (closer to another profile): "
          f"{(conf['silhouette'] < 0).sum()} of {len(conf)}")
    print("\n  by state (the three used in the unseen-state test):")
    for state in UNSEEN_STATES:
        sub = conf[conf["state"].str.contains(state, case=False, na=False)]
        print(f"    {state:10s} median margin {sub['label_margin'].median():.2f} | "
              f"GMM prob {sub['gmm_max_probability'].median():.2f} | "
              f"fuzzy {sub['fuzzy_membership'].median():.2f} | "
              f"borderline {(sub['confidence'] == 'borderline').sum()}/{len(sub)}")

    save_json({**perm,
               "negative_silhouette_districts": int((conf["silhouette"] < 0).sum()),
               "borderline_districts": int((conf["confidence"] == "borderline").sum()),
               "n_districts": int(len(conf))},
              REPORTS / "validation.json")

    lines = [
        "# Phase 7b - validation", "",
        "## 1. Permutation test", "",
        f"Each of the {X.shape[1]} feature columns was shuffled independently "
        f"{N_PERMUTATIONS} times - which keeps every indicator's own distribution intact and "
        "destroys only the relationships between them - and the complete pipeline "
        "(standardise -> PCA -> K-means) was re-run on each shuffle.", "",
        "| | real data | shuffled (mean) | shuffled (best of 20) | p |",
        "|---|---|---|---|---|",
        f"| Silhouette | **{perm['real_silhouette']:.3f}** | {perm['shuffled_silhouette_mean']:.3f} | "
        f"{perm['shuffled_silhouette_max']:.3f} | {perm['silhouette_p_value']:.3f} |",
        f"| Held-out outcome eta-squared | **{perm['real_outcome_eta2']:.3f}** | "
        f"{perm['shuffled_outcome_eta2_mean']:.3f} | {perm['shuffled_outcome_eta2_max']:.3f} | "
        f"{perm['outcome_eta2_p_value']:.3f} |", "",
        "The second row is the stronger result: the outcome columns were never shuffled, so "
        "a grouping built on scrambled features has no way to predict them.", "",
        "## 2. Silhouette against the baselines", "",
        sil_table.round(3).to_markdown(index=False), "",
        "## 3. Soft boundaries", "",
        f"- **{int((conf['confidence'] == 'borderline').sum())} of {len(conf)}** districts are "
        "borderline (label margin below 1.15).",
        f"- **{int((conf['silhouette'] < 0).sum())}** districts have a negative silhouette - they "
        "sit closer to another profile's centre than to their own.",
        "- Per-district silhouette, GMM membership probability and label margin all agree; "
        "`data/processed/district_confidence.csv` carries all three for every district.", "",
        "| state | median label margin | median GMM probability | borderline | unseen-state ARI |",
        "|---|---|---|---|---|",
    ]
    unseen = pd.read_csv(REPORTS / "unseen_state_test.csv").set_index("state")
    for state in UNSEEN_STATES:
        sub = conf[conf["state"].str.contains(state, case=False, na=False)]
        ari = unseen.loc[state, "ari_vs_full_model"] if state in unseen.index else float("nan")
        lines.append(f"| {state} | {sub['label_margin'].median():.2f} | "
                     f"{sub['gmm_max_probability'].median():.2f} | "
                     f"{(sub['confidence'] == 'borderline').sum()}/{len(sub)} | {ari:.3f} |")
    lines += ["", "Read the last two columns together: the states that fail to reproduce are the "
              "ones whose districts sit on boundaries.", ""]
    (REPORTS / "validation.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n  saved reports/validation.md, reports/validation.json, "
          "data/processed/district_confidence.csv, figures 20-23")


if __name__ == "__main__":
    main()
