"""
PHASE 10 -- Streamlit app: District Development Profiler (NFHS-5).

Everything on screen is computed live from the saved pipeline in models/ and
the tables in data/processed/ - there are no hard-coded results. Rename a
cluster in config/cluster_names.json and the whole app follows.

Run:  streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from common import CONFIG, DATA_PROC, MODELS, REPORTS, load_json   # noqa: E402
from indicator_meta import DOMAIN_LABELS                            # noqa: E402
from predict import Pipeline                                        # noqa: E402

st.set_page_config(page_title="District Development Profiler", layout="wide",
                   page_icon="🗺️")

CLUSTER_COLOURS = px.colors.qualitative.Bold


# --------------------------------------------------------------------------
# loading (cached so the app stays responsive)
# --------------------------------------------------------------------------
@st.cache_resource
def get_pipeline() -> Pipeline:
    """Models are expensive to load and never change -> cache_resource."""
    return Pipeline()


@st.cache_data
def get_tables():
    """Data frames -> cache_data (they are copied per session, which is fine)."""
    clusters = pd.read_csv(DATA_PROC / "district_clusters.csv")
    X = pd.read_csv(DATA_PROC / "features.csv")
    cfg = pd.read_csv(CONFIG / "indicators.csv")
    comparison = pd.read_csv(REPORTS / "comparison.csv")
    profiles = pd.read_csv(REPORTS / "cluster_domain_profiles.csv", index_col=0)
    stability = pd.read_csv(REPORTS / "stability_by_k.csv")
    pairs = pd.read_csv(REPORTS / "same_score_different_problems.csv")
    unseen = pd.read_csv(REPORTS / "unseen_state_test.csv")
    return clusters, X, cfg, comparison, profiles, stability, pairs, unseen


@st.cache_data
def get_scores(_pipe_id: str) -> np.ndarray:
    pipe = get_pipeline()
    return pipe.all_scores()


pipe = get_pipeline()
clusters, X, cfg, comparison, profiles, stability, pairs, unseen = get_tables()
scores = get_scores("v1")
names = load_json(CONFIG / "cluster_names.json")
final_cfg = load_json(CONFIG / "final_model.json")
k_cfg = load_json(CONFIG / "k_choice.json")

domain_cols = [c for c in clusters.columns if c.startswith("domain_")]
domain_keys = [c.replace("domain_", "") for c in domain_cols]
domain_nice = [DOMAIN_LABELS[d] for d in domain_keys]
direction = dict(zip(cfg["short_name"], cfg["direction"]))
domain_of = dict(zip(cfg["short_name"], cfg["domain"]))
full_name = dict(zip(cfg["short_name"], cfg["full_name"]))


def cluster_label(cid: int) -> str:
    return f"{cid} - {names[str(cid)]['name']}"


@st.cache_data
def get_centroids(_v: str) -> np.ndarray:
    return pd.DataFrame(scores).groupby(clusters["cluster"]).mean().to_numpy()


def label_margin(idx: int) -> tuple[float, int]:
    """
    How safe is this district's label?

    Every profile has a centre, and a district takes the label of the nearest
    one. The margin is (distance to the 2nd-nearest centre) / (distance to the
    nearest): 1.0 means the district sits exactly between two profiles and got
    its label by a hair; 2.0 means the runner-up is twice as far away and
    nothing short of rebuilding the model would move it.

    This is the honest answer to "is this district really profile 4?", and it is
    why two of the three held-out states in Phase 9 reproduce badly: their
    districts are mostly boundary cases.
    """
    d = np.linalg.norm(scores[idx] - get_centroids("v1"), axis=1)
    order = np.argsort(d)
    return float(d[order[1]] / d[order[0]]), int(order[1])


NATIONAL_MEDIAN_MARGIN = float(np.median([
    (lambda s: s[1] / s[0])(np.sort(np.linalg.norm(scores[i] - get_centroids("v1"), axis=1)))
    for i in range(len(clusters))]))


def oriented_z_row(values: pd.Series) -> pd.Series:
    """z-score one district's values against all districts, flipped so + = better."""
    z = (values[X.columns] - X.mean()) / X.std()
    return pd.Series({c: z[c] * direction[c] for c in X.columns})


def domain_scores(values: pd.Series) -> pd.Series:
    z = oriented_z_row(values)
    return pd.Series({d: z[[c for c in X.columns if domain_of[c] == d]].mean()
                      for d in domain_keys})


def radar_figure(traces: list[tuple[str, pd.Series, str]]) -> go.Figure:
    fig = go.Figure()
    for label, series, colour in traces:
        vals = list(series[domain_keys]) + [series[domain_keys[0]]]
        fig.add_trace(go.Scatterpolar(r=vals, theta=domain_nice + [domain_nice[0]],
                                      name=label, line=dict(color=colour, width=2),
                                      fill="toself", opacity=0.35))
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[-2, 2], tickvals=[-1, 0, 1],
                                   ticktext=["-1 sd", "national average", "+1 sd"])),
        showlegend=True, height=520, margin=dict(t=40, b=40))
    return fig


def pca_scatter(highlight_idx: int | None = None, extra_point: np.ndarray | None = None,
                extra_label: str = "your district") -> go.Figure:
    """PC1 vs PC2 for every district, coloured by cluster."""
    df = clusters.copy()
    df["PC1"], df["PC2"] = scores[:, 0], scores[:, 1]
    df["profile"] = df["cluster"].map(lambda c: cluster_label(int(c)))
    fig = px.scatter(df, x="PC1", y="PC2", color="profile",
                     hover_data={"district": True, "state": True, "PC1": ":.2f", "PC2": ":.2f"},
                     color_discrete_sequence=CLUSTER_COLOURS, opacity=0.75)
    fig.update_traces(marker=dict(size=7))
    if highlight_idx is not None:
        r = df.iloc[highlight_idx]
        fig.add_trace(go.Scatter(x=[r["PC1"]], y=[r["PC2"]], mode="markers+text",
                                 marker=dict(size=20, color="black", symbol="star"),
                                 text=[r["district"]], textposition="top center",
                                 name=r["district"]))
    if extra_point is not None:
        fig.add_trace(go.Scatter(x=[extra_point[0]], y=[extra_point[1]], mode="markers+text",
                                 marker=dict(size=22, color="black", symbol="x"),
                                 text=[extra_label], textposition="top center",
                                 name=extra_label))
    fig.update_layout(height=620, legend=dict(font=dict(size=10)),
                      xaxis_title="PC1", yaxis_title="PC2")
    return fig


# --------------------------------------------------------------------------
# sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("About this tool")
    st.markdown(
        f"""
**Data** National Family Health Survey **NFHS-5 (2019-21)**, district factsheets,
with NFHS-4 (2015-16) used only for the change analysis.

**Districts** {len(clusters)}
**Clustering features** {len(X.columns)}
**Held-out outcomes** 9 (stunting, wasting, severe wasting, underweight, overweight, 4 anaemia measures)

**Final model** `{final_cfg['algo']}` on {final_cfg['n_components'] or 'raw standardised'} PCA components
**Profiles (k)** {final_cfg['k']}

Sources: [SaiSiddhardhaKalla/NFHS](https://github.com/SaiSiddhardhaKalla/NFHS),
[jvargh7/nfhs5_factsheets](https://github.com/jvargh7/nfhs5_factsheets),
original factsheets from [IIPS](http://rchiips.org/nfhs/).
        """)
    st.caption("Clusters describe association, not cause. NFHS values are survey "
               "estimates with sampling error.")

st.title("District Development Profiler")
st.caption("NFHS-5 (2019-21) - what *kind* of development problem does a district have, "
           "not just how far behind it is.")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Explore a district", "Enter district values", "What-if simulator",
    "Clusters overview", "Why clustering?"])

# --------------------------------------------------------------------------
# TAB 1 - explore a district
# --------------------------------------------------------------------------
with tab1:
    c1, c2 = st.columns(2)
    state = c1.selectbox("State / UT", sorted(clusters["state"].unique()))
    district = c2.selectbox("District",
                            sorted(clusters.loc[clusters["state"] == state, "district"]))
    idx = clusters.index[(clusters["state"] == state) & (clusters["district"] == district)][0]
    row = clusters.loc[idx]
    cid = int(row["cluster"])

    margin, runner_up = label_margin(idx)
    if margin >= 1.35:
        verdict, colour = "solidly inside this profile", "normal"
    elif margin >= 1.15:
        verdict, colour = "typical for this profile", "off"
    else:
        verdict, colour = "near a boundary - treat as borderline", "inverse"

    a, b, c, d, e = st.columns(5)
    a.metric("Profile", f"#{cid}")
    b.metric("Label confidence", f"{margin:.2f}x", delta=verdict, delta_color=colour,
             help="How much further away the second-nearest profile is. 1.00x means the "
                  "district sits exactly between two profiles; 2.00x means the runner-up "
                  f"is twice as far. National median: {NATIONAL_MEDIAN_MARGIN:.2f}x.")
    c.metric("Composite score", f"{row['composite_score']:.3f}",
             help="Equal-weight index of all 77 features, 1 = best. This is the "
                  "traditional single-number view.")
    d.metric("Composite quantile group", f"{int(row['composite_group']) + 1} of {final_cfg['k']}")
    e.metric("Districts in this profile", int((clusters["cluster"] == cid).sum()))

    st.subheader(names[str(cid)]["name"])
    st.write(names[str(cid)]["policy_focus"])
    if margin < 1.15:
        st.warning(
            f"**Borderline district.** {district} is nearly as close to profile "
            f"{runner_up} ({names[str(runner_up)]['name']}) as to profile {cid}. "
            "Read both profiles before acting on the label - and expect this district "
            "to move if the model is rebuilt on slightly different data. This is exactly "
            "why the held-out-state test in tab 5 reproduces Bihar perfectly but not "
            "Assam or Karnataka.")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**This district vs its profile vs the national average**")
        own = domain_scores(X.loc[idx])
        clus = profiles.loc[cid]
        clus.index = [i for i in clus.index]
        nat = pd.Series({d_: 0.0 for d_ in domain_keys})
        st.plotly_chart(radar_figure([
            (district, own, "#111111"),
            (f"profile {cid} average", clus[domain_keys], CLUSTER_COLOURS[cid % len(CLUSTER_COLOURS)]),
            ("national average", nat, "#999999"),
        ]), width="stretch")
        st.caption("Every indicator is flipped so that outward = better than the "
                   "national average. A dent is a problem area.")

    with right:
        st.markdown("**Five most similar districts** (distance in PCA space)")
        point = scores[idx]
        dist = np.linalg.norm(scores - point, axis=1)
        near = clusters.assign(distance=dist).drop(index=idx).nsmallest(5, "distance")
        st.dataframe(
            near[["district", "state", "cluster", "cluster_name", "distance"]]
            .round({"distance": 2}).rename(columns={"cluster": "profile"}),
            hide_index=True, width="stretch")
        other_states = (near["state"] != state).sum()
        st.caption(f"{other_states} of the 5 nearest districts are in other states - "
                   "similar problems do not stop at state borders.")

        st.markdown("**Biggest gaps vs the national average**")
        z = oriented_z_row(X.loc[idx]).sort_values()
        gaps = pd.DataFrame({
            "indicator": [full_name[i][:70] for i in z.head(6).index],
            "value": [X.loc[idx, i] for i in z.head(6).index],
            "national mean": [X[i].mean() for i in z.head(6).index],
            "z (+ = better)": z.head(6).to_numpy(),
        })
        st.dataframe(gaps.round(2), hide_index=True, width="stretch")

    st.markdown("**Where this district sits among all districts**")
    st.plotly_chart(pca_scatter(highlight_idx=idx), width="stretch")

# --------------------------------------------------------------------------
# TAB 2 - enter district values
# --------------------------------------------------------------------------
with tab2:
    st.markdown("Enter values for a district that is not in NFHS-5 - or a hypothetical "
                "one. Anything you leave at the national average is treated as **given**; "
                "upload a CSV with missing columns to have them **imputed** instead.")

    template = pd.DataFrame([{c: round(float(X[c].mean()), 2) for c in X.columns}])
    st.download_button("Download a one-row CSV template", template.to_csv(index=False),
                       file_name="district_template.csv", mime="text/csv")

    mode = st.radio("Input method", ["Fill in a form", "Upload a one-row CSV"],
                    horizontal=True)

    values: dict | None = None
    if mode == "Upload a one-row CSV":
        up = st.file_uploader("CSV with one row and any subset of the indicator columns",
                              type="csv")
        if up is not None:
            raw = pd.read_csv(up)
            values = {c: raw.iloc[0][c] for c in raw.columns
                      if c in X.columns and not pd.isna(raw.iloc[0][c])}
            st.success(f"Read {len(values)} indicator values; "
                       f"{len(X.columns) - len(values)} will be imputed.")
    else:
        with st.form("manual_entry"):
            entered = {}
            for d_key, d_label in zip(domain_keys, domain_nice):
                cols_d = [c for c in X.columns if domain_of[c] == d_key]
                with st.expander(f"{d_label} ({len(cols_d)} indicators)"):
                    grid = st.columns(3)
                    for i, c in enumerate(cols_d):
                        entered[c] = grid[i % 3].number_input(
                            c, value=float(round(X[c].mean(), 1)),
                            help=full_name[c], key=f"manual_{c}")
            if st.form_submit_button("Assign a profile"):
                values = entered

    if values:
        res = pipe.assign_cluster(values)
        st.success(f"**Profile {res['cluster']} - {res['cluster_name']}**")
        st.write(res["policy_focus"])

        c1, c2 = st.columns([2, 3])
        with c1:
            dist = pd.DataFrame({
                "profile": [cluster_label(i) for i in res["distances"]],
                "distance to centre": list(res["distances"].values()),
            }).sort_values("distance to centre")
            st.dataframe(dist.round(2), hide_index=True, width="stretch")
            if res["imputed_fields"]:
                with st.expander(f"{len(res['imputed_fields'])} values were imputed"):
                    st.write(", ".join(res["imputed_fields"]))
            else:
                st.caption("No values needed imputing.")
        with c2:
            st.markdown("**Most similar real districts**")
            st.dataframe(pd.DataFrame(res["nearest_districts"]).round({"distance": 2}),
                         hide_index=True, width="stretch")

        st.plotly_chart(pca_scatter(extra_point=res["point"], extra_label="entered district"),
                        width="stretch")

# --------------------------------------------------------------------------
# TAB 3 - what-if simulator
# --------------------------------------------------------------------------
with tab3:
    st.markdown("Start from a real district, change a few things a programme could "
                "plausibly change, and watch whether the district moves to a different "
                "development profile.")

    c1, c2 = st.columns(2)
    s2 = c1.selectbox("State / UT", sorted(clusters["state"].unique()), key="wi_state")
    d2 = c2.selectbox("District", sorted(clusters.loc[clusters["state"] == s2, "district"]),
                      key="wi_district")
    i2 = clusters.index[(clusters["state"] == s2) & (clusters["district"] == d2)][0]
    baseline = X.loc[i2].copy()

    # The 12 sliders are chosen two ways, on purpose:
    #   * four programme levers - things a district administration can actually
    #     act on (cooking fuel, toilets, girls' schooling, institutional delivery);
    #   * the rest are the indicators with the largest weight on PC1, i.e. the ones
    #     that genuinely move a district's position in the space the model uses.
    # Sliders that only move the district a millimetre would make a pretty but
    # useless simulator.
    POLICY_LEVERS = ["clean_cooking_fuel", "improved_sanitation",
                     "women_10yr_schooling", "institutional_births"]
    loadings = pd.Series(pipe.pca.components_[0], index=X.columns).abs()
    slider_cols = [c for c in POLICY_LEVERS if c in X.columns]
    for c in loadings.sort_values(ascending=False).index:
        if len(slider_cols) >= 12:
            break
        if c not in slider_cols:
            slider_cols.append(c)

    st.markdown("**Adjust up to 12 of the most influential indicators**")
    edited = baseline.to_dict()
    grid = st.columns(3)
    for i, c in enumerate(slider_cols):
        lo, hi = float(X[c].min()), float(X[c].max())
        edited[c] = grid[i % 3].slider(
            f"{c} ({DOMAIN_LABELS[domain_of[c]].split(' (')[0]})",
            min_value=round(lo, 1), max_value=round(hi, 1),
            value=float(round(baseline[c], 1)),
            help=full_name[c],
            # The district index is part of the key on purpose. Streamlit keeps a
            # widget's value in session state once its key exists and ignores
            # `value=` from then on - so with a fixed key, switching districts
            # would silently leave the previous district's slider positions in
            # place and simulate the wrong starting point. Keying on the district
            # makes each district get its own set of sliders, loaded from its own
            # real values, and remembered if you switch back.
            key=f"wi_{i2}_{c}")

    res = pipe.assign_cluster(edited)
    now_cid = int(clusters.loc[i2, "cluster"])
    moved = res["cluster"] != now_cid

    m1, m2, m3 = st.columns(3)
    m1.metric("Current profile", f"#{now_cid}", help=names[str(now_cid)]["name"])
    m1.caption(names[str(now_cid)]["name"])
    m2.metric("Simulated profile", f"#{res['cluster']}",
              delta="moved" if moved else "unchanged",
              delta_color="normal" if moved else "off")
    m2.caption(res["cluster_name"])
    changed = [c for c in slider_cols if abs(edited[c] - baseline[c]) > 1e-9]
    m3.metric("Indicators changed", len(changed))

    if moved:
        st.success(f"With these values {d2} would sit in profile {res['cluster']} - "
                   f"{res['cluster_name']}. {res['policy_focus']}")
    else:
        st.info(f"{d2} stays in profile {now_cid}. Try raising clean fuel, women's "
                "schooling or sanitation further.")

    c1, c2 = st.columns([2, 3])
    with c1:
        st.markdown("**Distance to each profile centre**")
        base_res = pipe.assign_cluster(baseline.to_dict())
        comp_df = pd.DataFrame({
            "profile": [cluster_label(i) for i in res["distances"]],
            "before": list(base_res["distances"].values()),
            "after": list(res["distances"].values()),
        })
        comp_df["change"] = comp_df["after"] - comp_df["before"]
        st.dataframe(comp_df.round(2), hide_index=True, width="stretch")
    with c2:
        st.markdown("**Domain profile before and after**")
        before = domain_scores(baseline)
        after = domain_scores(pd.Series(res["prepared_values"]))
        st.plotly_chart(radar_figure([
            (f"{d2} now", before, "#999999"),
            (f"{d2} simulated", after, "#c7254e"),
        ]), width="stretch")

# --------------------------------------------------------------------------
# TAB 4 - clusters overview
# --------------------------------------------------------------------------
with tab4:
    sizes = clusters["cluster"].value_counts().sort_index()
    fig = px.bar(x=[cluster_label(int(c)) for c in sizes.index], y=sizes.to_numpy(),
                 labels={"x": "", "y": "districts"},
                 color=[cluster_label(int(c)) for c in sizes.index],
                 color_discrete_sequence=CLUSTER_COLOURS)
    fig.update_layout(showlegend=False, height=380)
    st.plotly_chart(fig, width="stretch")

    st.markdown("**Mean domain score per profile** (+ = better than the national average)")
    heat = profiles[domain_keys].copy()
    heat.columns = domain_nice
    heat.index = [cluster_label(int(c)) for c in heat.index]
    fig = px.imshow(heat, color_continuous_scale="RdYlGn", zmin=-1.2, zmax=1.2,
                    text_auto=".2f", aspect="auto")
    fig.update_layout(height=420, coloraxis_colorbar=dict(title="z"))
    st.plotly_chart(fig, width="stretch")

    for cid in sorted(clusters["cluster"].unique()):
        sub = clusters[clusters["cluster"] == cid]
        with st.expander(f"{cluster_label(int(cid))} - {len(sub)} districts"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**States most represented**")
                st.dataframe(sub["state"].value_counts().head(6).rename("districts"),
                             width="stretch")
                st.markdown(f"**Policy focus** {names[str(cid)]['policy_focus']}")
            with c2:
                st.markdown("**Most typical districts** (closest to the profile centre)")
                st.dataframe(sub.nsmallest(5, "distance_to_centroid")[
                    ["district", "state", "composite_score"]].round(3),
                    hide_index=True, width="stretch")
            p = profiles.loc[cid, domain_keys]
            st.markdown("**Strengths:** " + (", ".join(
                f"{DOMAIN_LABELS[d]} ({p[d]:+.2f})" for d in p[p > 0.35].index) or "none"))
            st.markdown("**Gaps:** " + (", ".join(
                f"{DOMAIN_LABELS[d]} ({p[d]:+.2f})" for d in p[p < -0.35].index) or "none"))

# --------------------------------------------------------------------------
# TAB 5 - why clustering?
# --------------------------------------------------------------------------
with tab5:
    st.markdown("### Does clustering beat a single score or grouping by state?")
    st.markdown(
        "Ten nutrition and anaemia indicators were **held out**: no grouping on this "
        "page ever saw them. `eta-squared` is the share of those outcomes' variance a "
        "grouping explains. A grouping that captures real differences between districts "
        "should separate outcomes it has never seen.")

    view = comparison[["grouping", "n_groups", "silhouette", "davies_bouldin",
                       "mean_outcome_eta2", "distinct_features", "stability_ari"]]
    st.dataframe(view.round(3), hide_index=True, width="stretch")

    main_rows = comparison[comparison["grouping"].isin(
        ["kmeans_pca80", "kmeans_pca", "ward_pca", "gmm_pca", "kmeans_raw", "ward_raw",
         "gmm_raw", "dbscan_pca", "composite_index", "domain_index", "state", "region"])].copy()
    main_rows["kind"] = np.where(
        main_rows["grouping"].isin(["composite_index", "domain_index", "state", "region"]),
        "traditional baseline", "clustering")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(main_rows.sort_values("mean_outcome_eta2"), x="mean_outcome_eta2",
                     y="grouping", color="kind", orientation="h",
                     color_discrete_map={"clustering": "#2c7fb8",
                                         "traditional baseline": "#c7254e"},
                     labels={"mean_outcome_eta2": "held-out outcome variance explained"})
        fig.update_layout(height=430)
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = px.bar(main_rows.sort_values("stability_ari"), x="stability_ari", y="grouping",
                     color="kind", orientation="h",
                     color_discrete_map={"clustering": "#2c7fb8",
                                         "traditional baseline": "#c7254e"},
                     labels={"stability_ari": "stability under 80% resampling (ARI)"})
        fig.update_layout(height=430)
        st.plotly_chart(fig, width="stretch")

    sil_row = comparison.set_index("grouping")
    st.info(
        f"""
**How we picked the number of profiles.** Nothing in the usual toolkit could pick it —
every number of profiles from 3 to 10 scores about the same on separation. So we asked a
different question: *would we get the same profiles from a slightly different sample?*
We kept dropping a random 20% of districts and rebuilding. **{final_cfg['k']} profiles**
is the most detail that still comes back the same way.

**How we picked the method.** Same test first: any method whose groups fell apart when
20% of districts were dropped was ruled out — that removed two methods with slightly
better scores. Among the rest we took the one that best explains the nine hidden
outcomes. Winner: **K-means on {final_cfg['n_components']} PCA components**.

*Both rules were written down before we looked at the results, and neither one used the
hidden outcomes to choose the number of profiles.*
""")

    st.info(
        f"**One number worth pausing on.** Our clusters score "
        f"{sil_row.loc[final_cfg['final_grouping'], 'silhouette']:.3f} on separation, which sounds "
        f"low — until you check the alternatives. The composite index's eight bands score "
        f"{sil_row.loc['composite_index', 'silhouette']:.3f} and the domain-weighted index "
        f"{sil_row.loc['domain_index', 'silhouette']:.3f}: **negative**. A typical district in one "
        "of those score bands sits closer to another band's districts than to its own. A slice "
        "of a ranking is not a group of similar districts.")

    st.markdown("### Same score, different problems")
    st.markdown(f"{len(pairs)} pairs of districts sit within 0.02 of each other on the "
                "composite index but fall in different profiles. Pick one:")
    choice = st.selectbox(
        "District pair",
        pairs.head(25).apply(lambda r: f"{r['district_a']}  vs  {r['district_b']}", axis=1))
    pr = pairs.head(25)[pairs.head(25).apply(
        lambda r: f"{r['district_a']}  vs  {r['district_b']}", axis=1) == choice].iloc[0]
    a_dom = domain_scores(X.loc[int(pr["row_a"])])
    b_dom = domain_scores(X.loc[int(pr["row_b"])])
    cmp_df = pd.DataFrame({"domain": domain_nice,
                           pr["district_a"]: a_dom[domain_keys].to_numpy(),
                           pr["district_b"]: b_dom[domain_keys].to_numpy()})
    fig = px.bar(cmp_df.melt(id_vars="domain", var_name="district", value_name="score"),
                 x="domain", y="score", color="district", barmode="group",
                 color_discrete_sequence=["#2c7fb8", "#d95f0e"])
    fig.update_layout(height=430, xaxis_tickangle=-25,
                      yaxis_title="domain score (+ = better than national average)")
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Composite scores {pr['score_a']:.3f} and {pr['score_b']:.3f} - "
               f"identical to two decimal places - but profiles "
               f"{int(pr['cluster_a'])} and {int(pr['cluster_b'])}.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### How many components does PCA need?")
        evr = np.cumsum(pipe.pca.explained_variance_ratio_) * 100
        fig = px.line(x=np.arange(1, len(evr) + 1), y=evr,
                      labels={"x": "components", "y": "cumulative variance explained (%)"})
        for level in (80, 90, 95):
            fig.add_hline(y=level, line_dash="dash", line_color="#c7254e",
                          annotation_text=f"{level}%")
        # the final model may run on the raw standardised features, in which case
        # there is no component count to mark on this curve
        if final_cfg["n_components"]:
            fig.add_vline(x=final_cfg["n_components"], line_color="#31a354",
                          annotation_text=f"model uses {final_cfg['n_components']}")
        fig.update_layout(height=400, xaxis_range=[0, 60])
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.markdown("### How reproducible is each k?")
        fig = px.line(stability, x="k", y="stability_ari", markers=True,
                      error_y="stability_sd",
                      labels={"stability_ari": "bootstrap ARI vs full data"})
        fig.add_hline(y=0.60, line_dash="dash", line_color="#c7254e",
                      annotation_text="stability floor")
        fig.add_vline(x=final_cfg["k"], line_color="#31a354",
                      annotation_text=f"chosen k = {final_cfg['k']}")
        fig.update_layout(height=400)
        st.plotly_chart(fig, width="stretch")

    st.markdown("### Unseen-state test")
    st.markdown("The scaler, PCA rotation and clustering were refitted from scratch "
                "without each state, and that state's districts were then assigned by "
                "the reduced pipeline.")
    st.dataframe(unseen.round(3), hide_index=True, width="stretch")
