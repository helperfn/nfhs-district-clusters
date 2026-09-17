"""
PHASE 6 -- clustering.

Four algorithms, each fitted on two representations of the same districts:
  (a) 'raw'  = the 77 standardised indicators
  (b) 'pca'  = the principal components that carry 90% of the variance

  K-means        fast, assumes roughly round, similar-sized groups. The obvious
                 first choice, and the one the prior work on this dataset used.
  Ward linkage   builds a hierarchy by always merging the pair of clusters that
                 increases within-cluster variance least. No k needed up front,
                 and it produces a dendrogram we can show.
  GMM            soft clustering: each district gets a PROBABILITY of belonging
                 to each profile, and clusters may be elongated rather than
                 round. Full covariance on PCA features; diagonal on the raw
                 features, where full covariance has too many parameters.
  DBSCAN         density-based, included as a deliberate contrast: it can say
                 "this district belongs to no group". On data like this, where
                 districts form one continuous cloud rather than separated
                 blobs, we expect it to label a large share as noise - which is
                 itself a useful finding for the report.

Choosing k (3-10) uses four numbers plus judgement:
  silhouette         higher = better separated (range -1..1)
  Davies-Bouldin     lower  = better
  Calinski-Harabasz  higher = better
  BIC (GMM only)     lower  = better, and it penalises extra parameters
  interpretability   can a policy-maker describe the profile in a sentence?

Run:  python src/cluster.py
"""
from __future__ import annotations

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from common import (CONFIG, DATA_PROC, FIGURES, MODELS, REPORTS, SEED, banner,
                    save_json)

sns.set_theme(style="whitegrid", context="talk")

K_RANGE = range(3, 11)
MIN_CLUSTER_SIZE = 15          # warn if any cluster is smaller than this


# --------------------------------------------------------------------------
def score_all(Xeval: np.ndarray, labels: np.ndarray) -> dict:
    """Internal quality metrics, ignoring DBSCAN noise points (label -1)."""
    mask = labels >= 0
    uniq = np.unique(labels[mask])
    if len(uniq) < 2 or mask.sum() < 10:
        return {"silhouette": np.nan, "davies_bouldin": np.nan,
                "calinski_harabasz": np.nan, "n_clusters": len(uniq),
                "n_noise": int((~mask).sum())}
    return {
        "silhouette": silhouette_score(Xeval[mask], labels[mask]),
        "davies_bouldin": davies_bouldin_score(Xeval[mask], labels[mask]),
        "calinski_harabasz": calinski_harabasz_score(Xeval[mask], labels[mask]),
        "n_clusters": len(uniq),
        "n_noise": int((~mask).sum()),
    }


def fit_kmeans(Xfit, k):
    # n_init=50: K-means depends on its random starting centroids, so we run it
    # 50 times and keep the best. With a fixed seed this is fully reproducible.
    return KMeans(n_clusters=k, n_init=50, random_state=SEED).fit(Xfit)


def fit_ward(Xfit, k):
    return AgglomerativeClustering(n_clusters=k, linkage="ward").fit(Xfit)


def fit_gmm(Xfit, k, cov):
    return GaussianMixture(n_components=k, covariance_type=cov, n_init=10,
                           random_state=SEED).fit(Xfit)


# --------------------------------------------------------------------------
def sweep_k(Z: np.ndarray, P: np.ndarray) -> pd.DataFrame:
    """Fit every algorithm for every k and collect the selection metrics."""
    rows = []
    for space, Xfit in (("raw", Z), ("pca", P)):
        for k in K_RANGE:
            km = fit_kmeans(Xfit, k)
            rows.append(dict(space=space, algo="kmeans", k=k,
                             **score_all(Z, km.labels_), bic=np.nan))

            wd = fit_ward(Xfit, k)
            rows.append(dict(space=space, algo="ward", k=k,
                             **score_all(Z, wd.labels_), bic=np.nan))

            cov = "full" if space == "pca" else "diag"
            try:
                gm = fit_gmm(Xfit, k, cov)
                lab = gm.predict(Xfit)
                rows.append(dict(space=space, algo=f"gmm_{cov}", k=k,
                                 **score_all(Z, lab), bic=gm.bic(Xfit)))
            except Exception as exc:          # singular covariance -> fall back
                print(f"    GMM({cov}) k={k} on {space} failed: {exc}")
    return pd.DataFrame(rows)


def plot_selection(sweep: pd.DataFrame, inertia: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    axes[0, 0].plot(list(inertia), list(inertia.values()), marker="o", color="#253494")
    axes[0, 0].set_title("Elbow: K-means inertia (PCA space)")
    axes[0, 0].set_xlabel("k")
    axes[0, 0].set_ylabel("within-cluster sum of squares")

    for ax, metric, nice in [
        (axes[0, 1], "silhouette", "Silhouette (higher = better)"),
        (axes[1, 0], "davies_bouldin", "Davies-Bouldin (lower = better)"),
        (axes[1, 1], "calinski_harabasz", "Calinski-Harabasz (higher = better)"),
    ]:
        for (space, algo), g in sweep.groupby(["space", "algo"]):
            ax.plot(g["k"], g[metric], marker="o", ms=4, label=f"{algo} / {space}")
        ax.set_title(nice)
        ax.set_xlabel("k")
        ax.legend(fontsize=8)

    fig.suptitle("Choosing the number of clusters (metrics computed in the 77-feature space)",
                 fontsize=15)
    fig.tight_layout()
    fig.savefig(FIGURES / "09_choosing_k.png", dpi=140)
    plt.close(fig)

    gmm = sweep[sweep["algo"].str.startswith("gmm")].dropna(subset=["bic"])
    if len(gmm):
        plt.figure(figsize=(10, 6))
        for (space, algo), g in gmm.groupby(["space", "algo"]):
            plt.plot(g["k"], g["bic"], marker="o", label=f"{algo} / {space}")
        plt.title("GMM: BIC (lower = better)")
        plt.xlabel("k")
        plt.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(FIGURES / "10_gmm_bic.png", dpi=140)
        plt.close()


def plot_dendrogram(P: np.ndarray, ids: pd.DataFrame) -> None:
    Zl = linkage(P, method="ward")
    plt.figure(figsize=(18, 8))
    dendrogram(Zl, truncate_mode="lastp", p=40, leaf_rotation=90, leaf_font_size=9,
               color_threshold=None)
    plt.title("Ward dendrogram (PCA space, last 40 merges)\n"
              "the height of a join = how much variance that merge costs", fontsize=13)
    plt.ylabel("merge distance")
    plt.tight_layout()
    plt.savefig(FIGURES / "11_dendrogram.png", dpi=140)
    plt.close()


def tune_dbscan(P: np.ndarray) -> tuple[float, pd.DataFrame]:
    """
    Pick eps from the k-distance curve: sort every point's distance to its 5th
    nearest neighbour; the 'knee' is where distances start growing quickly, and
    that value is the usual eps choice.
    """
    k = 5
    nn = NearestNeighbors(n_neighbors=k).fit(P)
    d = np.sort(nn.kneighbors(P)[0][:, -1])

    # knee = point furthest from the straight line joining the curve's ends
    x = np.arange(len(d))
    line = d[0] + (d[-1] - d[0]) * x / (len(d) - 1)
    knee = int(np.argmax(line - d)) if (line - d).max() > 0 else int(len(d) * 0.95)
    eps = float(d[knee])

    plt.figure(figsize=(11, 6))
    plt.plot(x, d, color="#253494")
    plt.axhline(eps, ls="--", color="#c7254e", label=f"chosen eps = {eps:.2f}")
    plt.title(f"DBSCAN k-distance plot (distance to {k}th nearest neighbour, PCA space)")
    plt.xlabel("districts sorted by distance")
    plt.ylabel(f"{k}-NN distance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "12_dbscan_kdistance.png", dpi=140)
    plt.close()

    rows = []
    for mult in (0.75, 1.0, 1.25, 1.5, 2.0):
        for min_samples in (5, 10):
            lab = DBSCAN(eps=eps * mult, min_samples=min_samples).fit_predict(P)
            n_clusters = len({l for l in lab if l >= 0})
            rows.append(dict(eps=round(eps * mult, 3), min_samples=min_samples,
                             n_clusters=n_clusters,
                             noise_share=float((lab == -1).mean())))
    return eps, pd.DataFrame(rows)


# --------------------------------------------------------------------------
MIN_STABILITY_ARI = 0.60        # a profile set nobody can reproduce is useless
N_STABILITY_BOOTSTRAPS = 30


def stability_by_k(P: np.ndarray) -> pd.DataFrame:
    """
    How reproducible is a k-cluster solution?

    For each k: cluster the full data, then re-cluster 30 random 80% subsamples
    and measure the Adjusted Rand Index between each subsample's labels and the
    full-data labels for the same districts. ARI = 1 means identical grouping,
    0 means no better than chance. A solution that falls apart when 20% of
    districts are removed is fitting noise, not structure.
    """
    rng = np.random.default_rng(SEED)
    rows = []
    for k in K_RANGE:
        base = fit_kmeans(P, k).labels_
        scores = []
        for b in range(N_STABILITY_BOOTSTRAPS):
            idx = rng.choice(len(P), int(0.8 * len(P)), replace=False)
            lab = KMeans(n_clusters=k, n_init=20, random_state=b).fit(P[idx]).labels_
            scores.append(adjusted_rand_score(base[idx], lab))
        rows.append(dict(k=k, stability_ari=float(np.mean(scores)),
                         stability_sd=float(np.std(scores))))
    return pd.DataFrame(rows)


def choose_k(sweep: pd.DataFrame, P: np.ndarray) -> tuple[int, str, pd.DataFrame]:
    """
    Final k, chosen by a rule fixed in advance and using NO outcome data.

    Why not simply take the best silhouette? Because on this dataset the
    silhouette curve is essentially flat - every k from 3 to 10 scores about
    0.08-0.11 - since districts form one continuous cloud rather than separated
    blobs. Reading a winner off differences of 0.01 would be over-fitting a
    noisy metric. (For the record, the flat maximum sits at k = 3, and k = 3 is
    carried into the Phase 7 comparison table as an alternative.)

    The rule actually used:

      1. every cluster must have at least MIN_CLUSTER_SIZE districts, under both
         K-means and Ward - a profile of three districts is not a policy category;
      2. the solution must be reproducible: mean bootstrap ARI >= MIN_STABILITY_ARI;
      3. take the largest k such that *every* k up to it also passes - i.e. stop
         at the first k where stability breaks down, rather than cherry-picking
         a lone k further out that scrapes over the floor by 0.01.

    Step 3 is a deliberate choice about purpose. The point of this project is to
    say *what kind of problem* a district has, and more profiles carry more of
    that information - but only while the profiles are stable enough to be real.
    So we buy as much detail as stability will pay for, and no more.

    Outcomes (stunting, wasting, anaemia) are NOT consulted here. They are kept
    untouched for the Phase 7 fairness test, which would be meaningless if they
    had helped choose k.
    """
    stab = stability_by_k(P)

    size_ok = []
    for k in K_RANGE:
        if min(pd.Series(fit_kmeans(P, k).labels_).value_counts().min(),
               pd.Series(fit_ward(P, k).labels_).value_counts().min()) >= MIN_CLUSTER_SIZE:
            size_ok.append(k)

    passing = stab[(stab["k"].isin(size_ok)) & (stab["stability_ari"] >= MIN_STABILITY_ARI)]
    # walk up from the smallest k and stop at the first failure
    k = int(stab.loc[stab["stability_ari"].idxmax(), "k"])
    run = sorted(passing["k"].tolist())
    if run:
        k = run[0]
        for cand_k in run[1:]:
            if cand_k == k + 1:
                k = cand_k
            else:
                break

    sil = sweep[(sweep["space"] == "pca") & (sweep["algo"] == "kmeans")].set_index("k")["silhouette"]
    reason = (
        f"silhouette is flat across k (range {sil.min():.3f}-{sil.max():.3f}, best at k={int(sil.idxmax())}), "
        f"so k was chosen on reproducibility instead: k values with all clusters >= {MIN_CLUSTER_SIZE} "
        f"districts = {size_ok}; of those, mean bootstrap ARI >= {MIN_STABILITY_ARI} for "
        f"{run}; largest unbroken run ends at k = {k} "
        f"(ARI {float(stab.loc[stab['k'] == k, 'stability_ari'].iloc[0]):.2f})"
    )
    print("  " + reason)

    plt.figure(figsize=(11, 6))
    plt.errorbar(stab["k"], stab["stability_ari"], yerr=stab["stability_sd"],
                 marker="o", capsize=4, color="#253494", label="bootstrap ARI (mean +/- sd)")
    plt.axhline(MIN_STABILITY_ARI, ls="--", color="#c7254e",
                label=f"stability floor = {MIN_STABILITY_ARI}")
    plt.axvline(k, ls=":", color="#31a354", label=f"chosen k = {k}")
    plt.xlabel("k")
    plt.ylabel("Adjusted Rand Index vs full-data labels")
    plt.title("How reproducible is each k? (K-means, PCA space, 30 x 80% subsamples)")
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(FIGURES / "13_stability_by_k.png", dpi=140)
    plt.close()
    return k, reason, stab


def main() -> None:
    banner("PHASE 6 - CLUSTERING")
    Z = pd.read_csv(DATA_PROC / "features_scaled.csv").to_numpy()
    P = pd.read_csv(DATA_PROC / "pca_scores_90.csv").to_numpy()
    ids = pd.read_csv(DATA_PROC / "ids.csv")
    print(f"  raw space: {Z.shape},  PCA space: {P.shape}")

    inertia = {k: fit_kmeans(P, k).inertia_ for k in K_RANGE}
    sweep = sweep_k(Z, P)
    sweep.to_csv(REPORTS / "k_selection.csv", index=False)
    plot_selection(sweep, inertia)
    plot_dendrogram(P, ids)

    print("\n  metric summary (PCA space):")
    show = (sweep[sweep["space"] == "pca"]
            .pivot_table(index="k", columns="algo",
                         values="silhouette").round(3))
    print(show.to_string())

    eps, db_grid = tune_dbscan(P)
    db_grid.to_csv(REPORTS / "dbscan_grid.csv", index=False)
    print(f"\n  DBSCAN eps from k-distance knee: {eps:.2f}")
    print(db_grid.to_string(index=False))

    k, k_reason, stab = choose_k(sweep, P)
    stab.to_csv(REPORTS / "stability_by_k.csv", index=False)
    banner(f"CHOSEN k = {k}")

    # ---- fit and store every candidate at the chosen k --------------------
    labels = {}
    fitted = {}
    for space, Xfit in (("raw", Z), ("pca", P)):
        km = fit_kmeans(Xfit, k)
        labels[f"kmeans_{space}"] = km.labels_
        fitted[f"kmeans_{space}"] = km

        wd = fit_ward(Xfit, k)
        labels[f"ward_{space}"] = wd.labels_
        fitted[f"ward_{space}"] = wd

        cov = "full" if space == "pca" else "diag"
        gm = fit_gmm(Xfit, k, cov)
        labels[f"gmm_{space}"] = gm.predict(Xfit)
        fitted[f"gmm_{space}"] = gm

    # DBSCAN: keep the setting that produces the most usable partition
    usable = db_grid[(db_grid["n_clusters"] >= 2) & (db_grid["noise_share"] < 0.5)]
    row = (usable.sort_values("noise_share").iloc[0] if len(usable)
           else db_grid.sort_values("noise_share").iloc[0])
    db = DBSCAN(eps=float(row["eps"]), min_samples=int(row["min_samples"])).fit(P)
    labels["dbscan_pca"] = db.labels_
    fitted["dbscan_pca"] = db
    print(f"  DBSCAN final: eps={row['eps']}, min_samples={int(row['min_samples'])} "
          f"-> {int(row['n_clusters'])} clusters, {row['noise_share']:.0%} noise")

    lab_df = pd.DataFrame(labels)
    lab_df.to_csv(DATA_PROC / "cluster_labels_all.csv", index=False)

    print("\n  cluster sizes at the chosen k:")
    for name, lab in labels.items():
        sizes = pd.Series(lab).value_counts().sort_index()
        small = sizes[sizes < MIN_CLUSTER_SIZE]
        warn = f"   <-- WARNING: {len(small)} cluster(s) under {MIN_CLUSTER_SIZE}" if len(small) else ""
        print(f"    {name:14s} {sizes.to_dict()}{warn}")

    joblib.dump({"models": fitted, "k": k, "dbscan_eps": float(row["eps"]),
                 "dbscan_min_samples": int(row["min_samples"])},
                MODELS / "cluster_models.joblib")
    save_json({"k": k, "rule": k_reason,
               "min_cluster_size": MIN_CLUSTER_SIZE,
               "min_stability_ari": MIN_STABILITY_ARI,
               "k_range_tested": list(K_RANGE)},
              CONFIG / "k_choice.json")
    print(f"\n  saved models/cluster_models.joblib and "
          f"data/processed/cluster_labels_all.csv (k = {k})")


if __name__ == "__main__":
    main()
