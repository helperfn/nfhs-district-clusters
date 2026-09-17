"""
PHASE 9 -- assign a cluster to values that were not in the training data.

`assign_cluster(values)` takes a plain dict of indicator -> number (any subset
of the indicators; missing ones are filled in) and pushes it through exactly the
saved pipeline:

    fill gaps (KNN imputer)  ->  clip to the 1st/99th percentile
      ->  standardise (saved scaler)  ->  PCA (saved rotation)
      ->  final clustering model

Nothing is re-fitted, so a district assigned today gets the same answer as one
assigned during training. This is the function the Streamlit app calls.

It also runs the UNSEEN-DATA TEST: refit the entire pipeline with one state
removed, assign that state's districts using only the reduced pipeline, and
check whether they land in the same profile as when they were included. That is
the closest thing an unsupervised project has to a test set.

Run:  python src/predict.py      (after profile.py)
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from common import CONFIG, DATA_PROC, MODELS, REPORTS, SEED, banner, load_json

UNSEEN_STATES = ["Karnataka", "Bihar", "Assam"]


# --------------------------------------------------------------------------
class Pipeline:
    """Everything needed to turn raw indicator values into a cluster."""

    def __init__(self):
        imp = joblib.load(MODELS / "imputer.joblib")
        sp = joblib.load(MODELS / "scaler_pca.joblib")
        fm = joblib.load(MODELS / "final_model.joblib")
        self.final_cfg = load_json(CONFIG / "final_model.json")

        self.impute_pipeline = imp["pipeline"]          # scaler + KNNImputer
        self.impute_columns = imp["columns"]            # columns the imputer expects
        self.columns = imp["final_columns"]             # features after de-duplication
        self.clip_lo, self.clip_hi = imp["clip_lo"], imp["clip_hi"]
        self.scaler, self.pca = sp["scaler"], sp["pca"]
        self.estimator = fm["estimator"]
        self.n_components = fm["n_components"]

        self.X = pd.read_csv(DATA_PROC / "features.csv")
        self.ids = pd.read_csv(DATA_PROC / "ids.csv")
        self.clusters = pd.read_csv(DATA_PROC / "district_clusters.csv")
        self.names = load_json(CONFIG / "cluster_names.json")
        self.national_mean = self.X.mean()

    # ---- the three transformation steps ----------------------------------
    def prepare(self, values: dict) -> tuple[pd.DataFrame, list[str]]:
        """Fill missing indicators, clip outliers. Returns (one-row frame, imputed names)."""
        row = pd.Series({c: np.nan for c in self.impute_columns}, dtype=float)
        for k, v in values.items():
            if k in row.index and v is not None and not pd.isna(v):
                row[k] = float(v)
        imputed = [c for c in self.columns if pd.isna(row[c])]

        filled = self.impute_pipeline.named_steps["impute"].transform(
            self.impute_pipeline.named_steps["scale"].transform(row.to_frame().T))
        filled = self.impute_pipeline.named_steps["scale"].inverse_transform(filled)
        out = pd.DataFrame(filled, columns=self.impute_columns)[self.columns]
        return out.clip(lower=self.clip_lo, upper=self.clip_hi, axis=1), imputed

    def project(self, X_row: pd.DataFrame) -> np.ndarray:
        """Standardise, then rotate into the PCA space the final model uses."""
        Z = self.scaler.transform(X_row[self.columns])
        if self.n_components is None:
            return Z
        return self.pca.transform(Z)[:, :self.n_components]

    def all_scores(self) -> np.ndarray:
        f = self.final_cfg["pca_scores_file"]
        return pd.read_csv(DATA_PROC / (f or "features_scaled.csv")).to_numpy()

    # ---- the public function ---------------------------------------------
    def assign_cluster(self, values: dict) -> dict:
        """
        Assign one set of district values to a development profile.

        Returns cluster id and name, the distance (or probability) for every
        cluster, the five most similar real districts, and which indicators had
        to be imputed - the user should know how much of the answer is their
        data and how much is the model filling gaps.
        """
        X_row, imputed = self.prepare(values)
        point = self.project(X_row)

        scores = self.all_scores()
        labels = self.clusters["cluster"].to_numpy()
        centroids = pd.DataFrame(scores).groupby(labels).mean().to_numpy()
        distances = np.linalg.norm(centroids - point, axis=1)

        if isinstance(self.estimator, GaussianMixture):
            cluster = int(self.estimator.predict(point)[0])
            probs = self.estimator.predict_proba(point)[0]
        elif isinstance(self.estimator, KMeans):
            cluster = int(self.estimator.predict(point)[0])
            probs = None
        else:                                   # Ward has no predict(): use nearest centroid
            cluster = int(np.argmin(distances))
            probs = None

        d = np.linalg.norm(scores - point, axis=1)
        near = self.clusters.assign(distance=d).nsmallest(5, "distance")

        return {
            "cluster": cluster,
            "cluster_name": self.names[str(cluster)]["name"],
            "policy_focus": self.names[str(cluster)]["policy_focus"],
            "distances": {int(i): float(v) for i, v in enumerate(distances)},
            "probabilities": None if probs is None else {int(i): float(p) for i, p in enumerate(probs)},
            "nearest_districts": near[["state", "district", "cluster", "cluster_name", "distance"]]
                                 .to_dict("records"),
            "imputed_fields": imputed,
            "point": point[0],
            "prepared_values": X_row.iloc[0],
        }


# --------------------------------------------------------------------------
def unseen_state_test(state: str) -> dict:
    """
    Refit EVERYTHING without one state, then assign that state's districts.

    'Everything' matters: the scaler, the PCA rotation and the clustering are
    all rebuilt from the remaining districts, so the held-out state contributed
    nothing at all. We then compare the profile each of its districts gets with
    the profile it had in the full-data model, using the Adjusted Rand Index
    (which ignores the fact that cluster numbers get shuffled between fits).
    """
    X = pd.read_csv(DATA_PROC / "features.csv")
    ids = pd.read_csv(DATA_PROC / "ids.csv")
    full = pd.read_csv(DATA_PROC / "district_clusters.csv")["cluster"].to_numpy()
    cfgm = load_json(CONFIG / "final_model.json")
    k, n_comp = cfgm["k"], cfgm["n_components"]

    held = (ids["state"] == state).to_numpy()
    if held.sum() == 0:
        return {}

    scaler = StandardScaler().fit(X[~held])
    Ztr = scaler.transform(X[~held])
    if n_comp:
        pca = PCA(random_state=SEED).fit(Ztr)
        Ptr = pca.transform(Ztr)[:, :n_comp]
        Pte = pca.transform(scaler.transform(X[held]))[:, :n_comp]
    else:
        Ptr, Pte = Ztr, scaler.transform(X[held])

    # Refit the SAME algorithm the final model uses. Refitting K-means here while
    # the final model is a GMM would compare two different methods and make the
    # test look far worse than it is.
    algo = cfgm["algo"]
    if algo == "gmm":
        cov = "full" if n_comp else "diag"
        model = GaussianMixture(n_components=k, covariance_type=cov, n_init=10,
                                random_state=SEED).fit(Ptr)
        assigned = model.predict(Pte)
    elif algo == "ward":
        # Ward has no predict(): assign each held-out district to the nearest
        # centroid of the clusters learned on the remaining districts.
        train_labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(Ptr).labels_
        centroids = np.vstack([Ptr[train_labels == c].mean(axis=0) for c in np.unique(train_labels)])
        assigned = np.argmin(((Pte[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2), axis=1)
    else:
        model = KMeans(n_clusters=k, n_init=50, random_state=SEED).fit(Ptr)
        assigned = model.predict(Pte)

    return {
        "state": state,
        "n_districts": int(held.sum()),
        "ari_vs_full_model": float(adjusted_rand_score(full[held], assigned)),
        # how often two districts of that state that were together in the full
        # model are still together when the state was never seen
        "pair_agreement": float(np.mean([
            (full[held][i] == full[held][j]) == (assigned[i] == assigned[j])
            for i in range(held.sum()) for j in range(i + 1, held.sum())
        ])) if held.sum() > 1 else np.nan,
    }


def nfhs4_transitions(pipe: Pipeline) -> pd.DataFrame:
    """
    Optional extra: push each district's NFHS-4 (2015-16) values through the
    SAME pipeline and see which profile it would have been in five years
    earlier. Indicators missing in NFHS-4 are imputed, so this is indicative
    rather than exact - it is reported as a direction of travel, not a result.
    """
    w4 = pd.read_csv(DATA_PROC.parent / "interim" / "nfhs4_wide_raw.csv")
    ids = pd.read_csv(DATA_PROC / "ids.csv")
    now = pd.read_csv(DATA_PROC / "district_clusters.csv")

    merged = ids.merge(w4, on=["state", "district"], how="left")
    rows = []
    for i, r in merged.iterrows():
        vals = {c: r[c] for c in pipe.columns if c in merged.columns and not pd.isna(r[c])}
        if len(vals) < 20:                   # too little NFHS-4 data to be meaningful
            continue
        res = pipe.assign_cluster(vals)
        rows.append(dict(state=r["state"], district=r["district"],
                         cluster_2015=res["cluster"], cluster_2019=int(now.loc[i, "cluster"]),
                         n_indicators_available=len(vals)))
    return pd.DataFrame(rows)


def main() -> None:
    banner("PHASE 9 - ASSIGNING NEW DISTRICTS")
    pipe = Pipeline()
    print(f"  pipeline: {len(pipe.columns)} features -> "
          f"{pipe.n_components or 'no'} PCA components -> "
          f"{pipe.final_cfg['algo']} (k = {pipe.final_cfg['k']})")

    # ---- demo 1: a real district, fed back in as if it were new -----------
    demo_idx = 0
    demo_vals = pipe.X.iloc[demo_idx].to_dict()
    res = pipe.assign_cluster(demo_vals)
    truth = pd.read_csv(DATA_PROC / "district_clusters.csv").iloc[demo_idx]
    print(f"\n  sanity check - {truth['district']} ({truth['state']}) fed back in:")
    print(f"    assigned cluster {res['cluster']} ({res['cluster_name']})")
    print(f"    actual cluster   {truth['cluster']} ({truth['cluster_name']})")
    assert res["cluster"] == truth["cluster"], "pipeline is not self-consistent"

    # ---- demo 2: a partial, hand-entered district -------------------------
    partial = {"women_10yr_schooling": 25.0, "improved_sanitation": 45.0,
               "clean_cooking_fuel": 20.0, "institutional_births": 60.0,
               "fully_vaccinated_card_or_recall": 55.0}
    res2 = pipe.assign_cluster(partial)
    print(f"\n  partial input ({len(partial)} indicators given, "
          f"{len(res2['imputed_fields'])} imputed):")
    print(f"    -> cluster {res2['cluster']} ({res2['cluster_name']})")
    print("    nearest districts: "
          + ", ".join(f"{d['district']} ({d['state']})" for d in res2["nearest_districts"]))

    # ---- the unseen-state test -------------------------------------------
    banner("UNSEEN-STATE TEST (whole pipeline refitted without the state)")
    results = [unseen_state_test(s) for s in UNSEEN_STATES]
    res_df = pd.DataFrame([r for r in results if r])
    print(res_df.round(3).to_string(index=False))
    res_df.to_csv(REPORTS / "unseen_state_test.csv", index=False)

    # ---- optional NFHS-4 comparison ---------------------------------------
    banner("NFHS-4 (2015-16) PASSED THROUGH THE SAME PIPELINE")
    trans = nfhs4_transitions(pipe)
    if len(trans):
        trans.to_csv(REPORTS / "nfhs4_transitions.csv", index=False)
        table = pd.crosstab(trans["cluster_2015"], trans["cluster_2019"])
        table.to_csv(REPORTS / "nfhs4_transition_matrix.csv")
        moved = (trans["cluster_2015"] != trans["cluster_2019"]).mean()
        print(f"  districts compared: {len(trans)}")
        print(f"  changed profile between NFHS-4 and NFHS-5: {moved:.0%}")
        print(table.to_string())

    lines = [
        "# Phase 9 - assigning districts the model has not seen", "",
        "## Unseen-state test", "",
        "The scaler, the PCA rotation and the clustering were all refitted from "
        "scratch with one state removed; that state's districts were then assigned "
        "with `assign_cluster`. ARI compares those assignments with the profiles the "
        "same districts received in the full-data model.", "",
        res_df.round(3).to_markdown(index=False), "",
        f"Mean ARI across the three held-out states: "
        f"**{res_df['ari_vs_full_model'].mean():.3f}**, mean pair agreement "
        f"**{res_df['pair_agreement'].mean():.1%}**.", "",
    ]
    if len(trans):
        lines += [
            "## NFHS-4 -> NFHS-5 profile movement", "",
            f"{len(trans)} districts had enough NFHS-4 indicators to pass through the "
            f"pipeline; **{(trans['cluster_2015'] != trans['cluster_2019']).mean():.0%}** "
            "of them sit in a different profile in 2019-21 than they would have in 2015-16. "
            "NFHS-4 lacks several NFHS-5 indicators, which are imputed, so treat this as "
            "direction of travel rather than a measurement.", "",
            pd.crosstab(trans["cluster_2015"], trans["cluster_2019"]).to_markdown(), "",
        ]
    (REPORTS / "unseen_data.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n  saved reports/unseen_state_test.csv and reports/unseen_data.md")


if __name__ == "__main__":
    main()
