"""
PHASE 8 -- describe the clusters in language a policy audience can use.

For each cluster we compute a mean z-score per domain, with every indicator
first flipped so that POSITIVE ALWAYS MEANS BETTER. That single convention is
what makes the heatmap and the radar charts readable: a cluster whose nutrition
bar points inwards is a cluster with a nutrition problem, in every chart.

Cluster names are generated automatically from the strongest and weakest
domains (e.g. "Strong services, weak nutrition") and written to
config/cluster_names.json. That file is meant to be edited by hand - the
Streamlit app reads it live, so renaming a cluster there changes the whole app
without re-running anything.

Run:  python src/profile.py     (after evaluate.py)
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from common import (CONFIG, DATA_PROC, FIGURES, REPORTS, banner, load_json,
                    save_json)
from indicator_meta import DOMAIN_LABELS

sns.set_theme(style="whitegrid", context="talk")

STRONG = 0.35      # |mean z| above this counts as a real strength / gap
WEAK = -0.35


def oriented_z(X: pd.DataFrame, cfg: pd.DataFrame) -> pd.DataFrame:
    """Standardise every feature, then flip the ones where higher is worse."""
    direction = dict(zip(cfg["short_name"], cfg["direction"]))
    z = (X - X.mean()) / X.std()
    for c in z.columns:
        z[c] = z[c] * direction[c]
    return z


def domain_matrix(z: pd.DataFrame, cfg: pd.DataFrame) -> pd.DataFrame:
    """District x domain matrix of mean oriented z-scores."""
    dom = dict(zip(cfg["short_name"], cfg["domain"]))
    return z.T.groupby(z.columns.map(dom)).mean().T


SHORT_DOMAIN = {
    "population_household": "civil registration", "education": "schooling",
    "wash": "sanitation", "energy": "clean energy", "maternal_health": "maternal care",
    "child_health_immunisation": "child health", "nutrition": "nutrition",
    "anaemia": "anaemia", "ncd": "NCDs", "women_empowerment": "women's agency",
    "tobacco_alcohol": "tobacco & alcohol",
}

# NCD prevalence (high blood sugar, high BP) RISES with development: the
# best-off districts score worst on it. Including it in the level or in the
# "what is this cluster bad at" logic would produce nonsense labels such as
# "high development, weak NCDs", so it is excluded from naming - but it still
# appears in every profile table and chart, because it is a real burden.
NAMING_DOMAINS_EXCLUDED = ("ncd",)


def name_cluster(profile: pd.Series) -> str:
    """
    Turn a cluster's domain profile into a short, honest label: a development
    LEVEL plus the SHAPE of its gaps. Entirely mechanical - the names come from
    the numbers, not from what we hoped to find.
    """
    p = profile.drop(list(NAMING_DOMAINS_EXCLUDED), errors="ignore")
    level = p.mean()
    word = ("High development" if level > 0.35
            else "Low development" if level < -0.35 else "Middle development")

    gaps = p[p < -0.40].sort_values().head(2)
    strengths = p[p > 0.60].sort_values(ascending=False).head(1)
    if len(gaps):
        # "weak sanitation" reads naturally; "weak tobacco & alcohol" does not,
        # so burden-type domains get their own wording.
        phrase = {"tobacco_alcohol": "high tobacco & alcohol use",
                  "anaemia": "high anaemia"}
        return f"{word}, " + " and ".join(
            phrase.get(d, f"weak {SHORT_DOMAIN[d]}") for d in gaps.index)
    if len(strengths):
        return f"{word}, strong " + SHORT_DOMAIN[strengths.index[0]]
    return f"{word}, no standout gap"


def disambiguate(names: dict[int, str], prof: pd.DataFrame) -> dict[int, str]:
    """
    Two clusters can land on the same generic label ("no standout gap"). When
    that happens, add whichever domain most distinguishes each one from the
    average cluster, so every profile in the app has a unique name.
    """
    counts = pd.Series(list(names.values())).value_counts()
    for dupe in counts[counts > 1].index:
        for c in [c for c, n in names.items() if n == dupe]:
            diff = prof.loc[c] - prof.mean()
            d = diff.abs().idxmax()
            tag = ("above average on " if diff[d] > 0 else "below average on ") + SHORT_DOMAIN[d]
            base_name = names[c].replace(", no standout gap", "")
            names[c] = f"{base_name}, {tag}"
    return names


def policy_line(profile: pd.Series) -> str:
    """One sentence of intervention focus, driven by the two weakest domains."""
    gaps = profile.sort_values().head(2)
    short = {k: v.split(" (")[0] for k, v in DOMAIN_LABELS.items()}
    focus = {
        "nutrition": "child feeding and supplementary nutrition",
        "anaemia": "iron-folic-acid supplementation and anaemia screening",
        "wash": "sanitation coverage and safe drinking water",
        "energy": "clean cooking fuel and electrification",
        "education": "girls' schooling and adult literacy",
        "maternal_health": "antenatal care and institutional delivery",
        "child_health_immunisation": "routine immunisation and sick-child care",
        "women_empowerment": "family planning access and delaying marriage",
        "ncd": "blood pressure / blood sugar screening",
        "population_household": "civil registration and health insurance enrolment",
        "tobacco_alcohol": "tobacco and alcohol control",
    }
    parts = [focus[d] for d in gaps.index if gaps[d] < 0]
    if not parts:
        return "No domain is significantly below the national average - maintain services."
    return "Priority: " + "; then ".join(parts) + "."


def radar(ax, values: np.ndarray, labels: list[str], title: str, colour: str) -> None:
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    vals = list(values) + [values[0]]
    angles_c = angles + [angles[0]]
    ax.plot(angles_c, vals, color=colour, lw=2)
    ax.fill(angles_c, vals, color=colour, alpha=0.25)
    ax.plot(angles_c, [0] * len(angles_c), color="#888", lw=1, ls="--")
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(-1.5, 1.5)
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(["-1 sd", "national avg", "+1 sd"], fontsize=6)
    ax.set_title(title, fontsize=10, pad=18)


def main() -> None:
    banner("PHASE 8 - CLUSTER PROFILES")
    X = pd.read_csv(DATA_PROC / "features.csv")
    ids = pd.read_csv(DATA_PROC / "ids.csv")
    cfg = pd.read_csv(CONFIG / "indicators.csv")
    base = pd.read_csv(DATA_PROC / "baselines.csv")
    Y = pd.read_csv(DATA_PROC / "outcomes.csv")
    labels = pd.read_csv(DATA_PROC / "final_labels.csv")["cluster"].to_numpy()
    final = load_json(CONFIG / "final_model.json")
    scores = pd.read_csv(DATA_PROC / final["pca_scores_file"]).to_numpy() \
        if final["pca_scores_file"] else pd.read_csv(DATA_PROC / "features_scaled.csv").to_numpy()

    z = oriented_z(X, cfg)
    dom = domain_matrix(z, cfg)
    dom_labels = [DOMAIN_LABELS[c] for c in dom.columns]

    prof = dom.groupby(labels).mean()
    prof.index.name = "cluster"
    prof.to_csv(REPORTS / "cluster_domain_profiles.csv")

    # ---- heatmap ----------------------------------------------------------
    plt.figure(figsize=(13, 6))
    sns.heatmap(prof, annot=True, fmt=".2f", center=0, cmap="RdYlGn",
                vmin=-1.2, vmax=1.2, xticklabels=dom_labels,
                cbar_kws={"label": "mean z-score (+ = better than national average)"})
    plt.xticks(rotation=35, ha="right", fontsize=9)
    plt.ylabel("cluster")
    plt.title(f"Development profiles: mean domain z-score per cluster "
              f"({final['final_grouping']}, k = {final['k']})", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURES / "18_cluster_domain_heatmap.png", dpi=140)
    plt.close()

    # ---- radar charts -----------------------------------------------------
    k = len(prof)
    ncols = min(4, k)
    nrows = int(np.ceil(k / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5.5 * nrows),
                             subplot_kw={"projection": "polar"})
    palette = sns.color_palette("tab10", k)
    names_auto = disambiguate({int(c): name_cluster(prof.loc[c]) for c in prof.index}, prof)
    for ax, c in zip(np.atleast_1d(axes).ravel(), prof.index):
        radar(ax, prof.loc[c].to_numpy(), dom_labels,
              f"Cluster {c}: {names_auto[int(c)]}\n({(labels == c).sum()} districts)",
              palette[int(c)])
    for ax in np.atleast_1d(axes).ravel()[k:]:
        ax.axis("off")
    fig.suptitle("Each cluster's shape, not just its level", fontsize=16)
    fig.tight_layout()
    fig.savefig(FIGURES / "19_cluster_radars.png", dpi=140)
    plt.close(fig)

    # ---- per-cluster description -----------------------------------------
    centroids = pd.DataFrame(scores).groupby(labels).mean()
    dist_to_own = np.linalg.norm(scores - centroids.loc[labels].to_numpy(), axis=1)

    out = ids.copy()
    out["cluster"] = labels
    out["cluster_name"] = [names_auto[int(c)] for c in labels]
    out["composite_score"] = base["composite_score"]
    out["composite_group"] = base["composite_group"]
    out["distance_to_centroid"] = dist_to_own
    for c in dom.columns:
        out[f"domain_{c}"] = dom[c]
    out.to_csv(DATA_PROC / "district_clusters.csv", index=False)

    lines = [f"# Cluster profiles - {final['final_grouping']}, k = {final['k']}", "",
             "Domain scores are mean z-scores with every indicator oriented so that "
             "**positive = better than the national average**.", ""]
    for c in prof.index:
        sub = out[out["cluster"] == c]
        p = prof.loc[c]
        strengths = p[p > STRONG].sort_values(ascending=False)
        gaps = p[p < WEAK].sort_values()
        typical = sub.nsmallest(5, "distance_to_centroid")
        outcome_means = Y.groupby(labels).mean().loc[c]
        lines += [
            f"## Cluster {c} - {names_auto[int(c)]}", "",
            f"- **Districts:** {len(sub)}",
            f"- **States most represented:** "
            + ", ".join(f"{s} ({n})" for s, n in sub['state'].value_counts().head(4).items()),
            f"- **Mean composite score:** {sub['composite_score'].mean():.3f} "
            f"(national mean {base['composite_score'].mean():.3f})",
            f"- **Strengths:** "
            + (", ".join(f"{DOMAIN_LABELS[d]} ({v:+.2f})" for d, v in strengths.items())
               if len(strengths) else "none above +0.35 sd"),
            f"- **Gaps:** "
            + (", ".join(f"{DOMAIN_LABELS[d]} ({v:+.2f})" for d, v in gaps.items())
               if len(gaps) else "none below -0.35 sd"),
            f"- **Held-out outcomes (never used to build the cluster):** "
            f"stunting {outcome_means['child_stunted']:.1f}%, "
            f"underweight {outcome_means['child_underweight']:.1f}%, "
            f"child anaemia {outcome_means['children_anaemic']:.1f}%, "
            f"women's anaemia {outcome_means['women_15_49_anaemic']:.1f}%",
            f"- **Most typical districts:** "
            + ", ".join(f"{r.district} ({r.state})" for r in typical.itertuples()),
            f"- **Policy focus:** {policy_line(p)}", "",
        ]
    (REPORTS / "cluster_profiles.md").write_text("\n".join(lines), encoding="utf-8")

    save_json({str(c): {"name": names_auto[int(c)],
                        "policy_focus": policy_line(prof.loc[c]),
                        "n_districts": int((labels == c).sum())}
               for c in prof.index}, CONFIG / "cluster_names.json")

    print(prof.round(2).to_string())
    print("\n  cluster names (edit config/cluster_names.json to rename):")
    for c, n in names_auto.items():
        print(f"    {c}: {n}   [{(labels == c).sum()} districts]")
    print("\n  saved reports/cluster_profiles.md, data/processed/district_clusters.csv, figures 18-19")


if __name__ == "__main__":
    main()
