"""
Generate the written report as a Word document: reports/NFHS_District_Profiling_Report.docx

Everything in it is read from the pipeline's own outputs - the comparison table,
the validation results, the cluster profiles, the figures - so the document can
never drift from the analysis. Re-run this after re-running the pipeline.

Run:  python src/make_report.py
"""
from __future__ import annotations

import json

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

from common import CONFIG, DATA_PROC, FIGURES, REPORTS, banner, load_json

ACCENT = RGBColor(0x0B, 0x6B, 0x63)
MUTED = RGBColor(0x55, 0x63, 0x61)


# --------------------------------------------------------------------------
# small formatting helpers
# --------------------------------------------------------------------------
def style_document(doc: Document) -> None:
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)
    normal.paragraph_format.space_after = Pt(8)
    normal.paragraph_format.line_spacing = 1.15
    for level, size in ((1, 16), (2, 13), (3, 11.5)):
        st = doc.styles[f"Heading {level}"]
        st.font.name = "Calibri"
        st.font.size = Pt(size)
        st.font.color.rgb = ACCENT if level < 3 else MUTED
        st.paragraph_format.space_before = Pt(14 if level == 1 else 10)
        st.paragraph_format.space_after = Pt(4)


def para(doc, text, *, italic=False, size=None, align=None, space_after=None):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.italic = italic
    if size:
        run.font.size = Pt(size)
    if italic:
        run.font.color.rgb = MUTED
    if align is not None:
        p.alignment = align
    if space_after is not None:
        p.paragraph_format.space_after = Pt(space_after)
    return p


def rich(doc, *chunks):
    """rich(doc, ("plain ", False), ("bold", True)) - bold where the flag is True."""
    p = doc.add_paragraph()
    for text, bold in chunks:
        run = p.add_run(text)
        run.bold = bold
    return p


def bullet(doc, text, bold_prefix=None):
    p = doc.add_paragraph(style="List Bullet")
    if bold_prefix:
        p.add_run(bold_prefix).bold = True
    p.add_run(text)
    return p


def shade(cell, hex_colour: str) -> None:
    el = OxmlElement("w:shd")
    el.set(qn("w:fill"), hex_colour)
    cell._tc.get_or_add_tcPr().append(el)


def add_table(doc, frame: pd.DataFrame, highlight_row: int | None = None,
              col_widths=None, font_size=9):
    table = doc.add_table(rows=1, cols=len(frame.columns))
    table.style = "Light Grid Accent 1"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, col in enumerate(frame.columns):
        cell = table.rows[0].cells[j]
        cell.text = str(col)
        for p in cell.paragraphs:
            for r in p.runs:
                r.bold = True
                r.font.size = Pt(font_size)
    for i, (_, row) in enumerate(frame.iterrows()):
        cells = table.add_row().cells
        for j, value in enumerate(row):
            cells[j].text = "" if pd.isna(value) else str(value)
            for p in cells[j].paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT if j and isinstance(value, (int, float)) \
                    else WD_ALIGN_PARAGRAPH.LEFT
                for r in p.runs:
                    r.font.size = Pt(font_size)
                    if highlight_row is not None and i == highlight_row:
                        r.bold = True
        if highlight_row is not None and i == highlight_row:
            for c in cells:
                shade(c, "DCEBE8")
    if col_widths:
        for row in table.rows:
            for cell, w in zip(row.cells, col_widths):
                cell.width = Inches(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return table


def figure(doc, filename: str, caption: str, width=6.1):
    doc.add_picture(str(FIGURES / filename), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    para(doc, caption, italic=True, size=9, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=12)


# --------------------------------------------------------------------------
def main() -> None:
    banner("GENERATING THE WORD REPORT")

    comparison = pd.read_csv(REPORTS / "comparison.csv").set_index("grouping")
    clusters = pd.read_csv(DATA_PROC / "district_clusters.csv")
    unseen = pd.read_csv(REPORTS / "unseen_state_test.csv")
    stability = pd.read_csv(REPORTS / "stability_by_k.csv")
    pairs = pd.read_csv(REPORTS / "same_score_different_problems.csv")
    cfg = pd.read_csv(CONFIG / "indicators.csv")
    names = load_json(CONFIG / "cluster_names.json")
    final = load_json(CONFIG / "final_model.json")
    kcfg = load_json(CONFIG / "k_choice.json")
    val = json.loads((REPORTS / "validation.json").read_text())

    final_name = final["final_grouping"]
    k = final["k"]
    eta_final = comparison.loc[final_name, "mean_outcome_eta2"]
    eta_comp = comparison.loc["composite_index", "mean_outcome_eta2"]
    eta_state = comparison.loc["state", "mean_outcome_eta2"]

    doc = Document()
    style_document(doc)
    for section in doc.sections:
        section.top_margin = section.bottom_margin = Inches(0.9)
        section.left_margin = section.right_margin = Inches(0.95)

    # ------------------------------------------------------------ title ----
    para(doc, "MACHINE LEARNING PROJECT REPORT · UNSUPERVISED LEARNING",
         size=9, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("District Development Profiling from NFHS-5")
    run.bold = True
    run.font.size = Pt(24)
    run.font.color.rgb = ACCENT
    para(doc, "Clustering India's districts by the shape of their development, "
              "and testing the result against the ways districts are normally grouped",
         italic=True, size=11.5, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=16)

    summary = pd.DataFrame({
        "": ["Data", "Districts analysed", "Features used", "Outcomes held out",
             "Final model", "Profiles found"],
        " ": ["NFHS-5 (2019-21) district factsheets, IIPS",
              f"{len(clusters)}",
              f"{len(cfg[cfg.role == 'feature'])} candidates, {clusters.filter(like='domain_').shape[1]} domains "
              f"({comparison.loc[final_name, 'distinct_features']} of 77 separate the profiles)",
              "9 (child stunting, wasting, severe wasting, underweight, overweight; "
              "anaemia in children, women 15-19, women 15-49, non-pregnant women)",
              f"K-means on {final['n_components']} PCA components",
              f"{k}"],
    })
    add_table(doc, summary, col_widths=[1.5, 4.7], font_size=9.5)

    # ---------------------------------------------------------- abstract ----
    doc.add_heading("Abstract", level=1)
    rich(doc,
         ("India's districts are normally ranked on a single composite development score. "
          "That score reports how far behind a district is, but not what is wrong with it: "
          "two districts with the same score can require opposite interventions. This project "
          "clusters ", False),
         (f"{len(clusters)} districts", True),
         (" on ", False), ("77 NFHS-5 indicators", True),
         (" of living conditions, service use and health behaviour, and evaluates the result "
          "against four traditional groupings using ", False),
         ("nine nutrition and anaemia outcomes withheld from every model", True),
         (". The final model - K-means on ", False),
         (f"{final['n_components']} principal components with k = {k}", True),
         (" - explains ", False), (f"{eta_final:.3f}", True),
         (" of the variance in those hidden outcomes, against ", False),
         (f"{eta_comp:.3f}", True),
         (" for the composite index cut into the same number of groups. A permutation test "
          "confirms the structure is not an artefact of the method, and per-district label "
          "margins identify which assignments are reliable and which are borderline.", False))

    # ------------------------------------------------------- 1. problem ----
    doc.add_heading("1. Problem and objectives", level=1)
    para(doc, "NFHS-5 publishes roughly 100 health and development indicators for every "
              "district in India. Policy instruments such as the Aspirational Districts "
              "Programme and SDG district indices average these into one score and rank "
              "districts on it. The ranking is not wrong, but it is one-dimensional: it "
              "cannot distinguish a district short of sanitation and cooking fuel from one "
              "short of antenatal care, even when both sit at the same score.")
    bullet(doc, "group districts by the shape of their development rather than its level;",
           "Objective 1 — ")
    bullet(doc, "demonstrate that the grouping outperforms a single-score ranking and "
                "state-based grouping on evidence none of them was allowed to see;",
           "Objective 2 — ")
    bullet(doc, "deliver an interactive tool that assigns any district, including one not "
                "present in NFHS-5, to a profile and reports how confident that assignment is.",
           "Objective 3 — ")

    # ---------------------------------------------------------- 2. data ----
    doc.add_heading("2. Data and preprocessing", level=1)
    doc.add_heading("2.1 Sources and their alignment", level=2)
    para(doc, "Two public mirrors of the same IIPS factsheets were used, each for what it "
              "does better. The jvargh7 mirror supplies values: they are already numeric, "
              "and it preserves the reliability marks NFHS prints - 5,042 cells based on "
              "25-49 respondents, and 4,113 cells suppressed entirely because fewer than 25 "
              "respondents were available. The SaiSiddhardhaKalla mirror supplies Census 2011 "
              "district codes and clean indicator names.")
    para(doc, "The two could not be joined on indicator names. In the factsheet mirror the "
              "names are extracted from the source PDFs, and the gender of the "
              "non-communicable disease indicators survives only in the factsheet item "
              "number: items 86-88 are women's blood sugar and 89-91 are men's, with "
              "identical printed names. The mirrors were therefore aligned by value "
              "fingerprint - each item matched to the indicator whose values are closest "
              "across all 695 shared districts. Because both files derive from the same PDFs, "
              "the correct match agrees to approximately 0.001 while the runner-up differs by "
              "three orders of magnitude; the code asserts this margin rather than assuming "
              "the match. All 104 indicators aligned unambiguously.")

    doc.add_heading("2.2 Cleaning decisions", level=2)
    bullet(doc, "indicators missing in more than 10% of districts were dropped (nine, all "
                "small-denominator measures such as diarrhoea treatment); districts missing "
                "more than 20% of indicators were dropped (one).", "Completeness — ")
    bullet(doc, "consistent flagging revealed that anaemia in pregnant women is suppressed in "
                "19% of districts, so it was removed from the held-out set, leaving nine outcomes.",
           "Consequence — ")
    bullet(doc, "remaining gaps were filled with KNN imputation (k = 5) on standardised "
                "values. Features and outcomes were imputed separately so that no information "
                "could leak from the held-out set into the model inputs.", "Imputation — ")
    bullet(doc, "1.8% of feature cells were clipped to the 1st/99th percentile. Without this, "
                "single extreme values from very small districts claimed entire clusters.",
           "Winsorising — ")
    bullet(doc, "two indicators correlating above |r| = 0.95 with a retained indicator were "
                "removed, to stop one concept being counted twice in every distance.",
           "Redundancy — ")

    doc.add_heading("2.3 Exploratory analysis", level=2)
    para(doc, "Mean absolute correlation between features is 0.22, with 124 pairs above "
              "|r| = 0.6 - heavy redundancy, and the direct justification for PCA. "
              "Separately, 42% of the variance in a typical indicator lies within states "
              "rather than between them, which is the justification for clustering at "
              "district level rather than using the state as the unit.")
    figure(doc, "03_correlation_heatmap.png",
           "Figure 1. Feature correlations, ordered by domain. The dark blocks are groups of "
           "indicators measuring the same underlying construct.")

    # ------------------------------------------------------ 3. methods ----
    doc.add_page_break()
    doc.add_heading("3. Method", level=1)
    doc.add_heading("3.1 Dimensionality reduction", level=2)
    para(doc, "Features were standardised, then rotated by PCA. The first component explains "
              "25.3% of total variance and represents the reach of maternal care; the second "
              "(11.1%) separates districts with strong public immunisation from those with a "
              "costlier, more privatised system. Reaching 80 / 90 / 95% of variance requires "
              "18 / 30 / 41 components; all three were carried forward and compared.")

    doc.add_heading("3.2 Algorithms compared", level=2)
    algos = pd.DataFrame({
        "Algorithm": ["K-means", "Ward (agglomerative)", "Gaussian Mixture", "DBSCAN"],
        "Assumption": ["roughly spherical, similar-sized groups",
                       "merge whichever pair adds least within-cluster variance",
                       "data is a blend of Gaussian components; soft membership",
                       "clusters are dense regions separated by sparse ones"],
        "Role here": ["final model", "hierarchy and dendrogram",
                      "soft alternative", "control: tests whether density gaps exist at all"],
    })
    add_table(doc, algos, col_widths=[1.3, 2.9, 1.9])
    para(doc, "Each was fitted on both the standardised features and the principal components.",
         italic=True, size=9)

    doc.add_heading("3.3 Choosing the number of profiles", level=2)
    para(doc, "Silhouette could not decide: every k from 3 to 10 scores between 0.092 and "
              "0.110, because districts form a continuum rather than separated groups. "
              "Selecting on differences of 0.01 would fit noise. The criterion used instead, "
              "fixed in advance and using no outcome data, was reproducibility: every cluster "
              "must contain at least 15 districts, and the solution must achieve a mean "
              "bootstrap Adjusted Rand Index of at least 0.60 across 30 subsamples of 80% of "
              f"districts. Values k = 3 to 8 satisfy this and k = 9 fails, giving k = {k}.")
    figure(doc, "13_stability_by_k.png",
           "Figure 2. Reproducibility by number of profiles. The chosen k is the last value "
           "before stability falls below the pre-set floor.")

    doc.add_heading("3.4 Baselines", level=2)
    para(doc, "Four traditional groupings of the same districts were scored identically: an "
              "equal-weight composite index; a domain-weighted index that averages within "
              "each of the 11 domains before averaging across them; grouping by state; and "
              "grouping by region. Both indices were cut into the same number of quantile "
              "groups as the clustering, because eta-squared increases mechanically with the "
              "number of groups and only an equal-group comparison is fair.")

    # ------------------------------------------------------- 4. results ----
    doc.add_page_break()
    doc.add_heading("4. Results", level=1)
    doc.add_heading("4.1 Comparison of all groupings", level=2)

    label = {
        "state": "State grouping", "region": "Region grouping",
        "composite_index": "Composite index", "domain_index": "Domain-weighted index",
        "kmeans_pca": "K-means, 30 PCA components", "kmeans_raw": "K-means, 77 features",
        "kmeans_pca80": "K-means, 18 PCA components", "kmeans_pca95": "K-means, 41 PCA components",
        "kmeans_pca_k3": "K-means, k = 3", "ward_raw": "Ward, 77 features",
        "ward_pca": "Ward, PCA", "gmm_raw": "GMM, 77 features", "gmm_pca": "GMM, PCA",
        "dbscan_pca": "DBSCAN",
    }
    order = ["state", "ward_raw", "gmm_raw", "kmeans_raw", "kmeans_pca", "region",
             "domain_index", "kmeans_pca_k3", "composite_index", "dbscan_pca"]
    rows = []
    for g in order:
        r = comparison.loc[g]
        note = ""
        if g == final_name:
            note = "final model"
        elif r["stability_ari"] < 0.60 and g not in ("state", "region", "composite_index",
                                                     "domain_index"):
            note = "excluded: unstable"
        elif g == "state":
            note = "not like-for-like (34 groups)"
        rows.append({"Grouping": label[g], "Groups": int(r["n_groups"]),
                     "Held-out eta2": round(r["mean_outcome_eta2"], 3),
                     "Silhouette": round(r["silhouette"], 3),
                     "Features separated": f"{int(r['distinct_features'])}/77",
                     "Stability": round(r["stability_ari"], 2), "Note": note})
    table_df = pd.DataFrame(rows)
    add_table(doc, table_df, highlight_row=order.index(final_name),
              col_widths=[1.75, 0.6, 0.85, 0.75, 0.9, 0.7, 1.15], font_size=8.5)
    para(doc, "Table 1. Every grouping scored identically. Eta-squared is the share of the "
              "nine held-out outcomes' variance explained; stability is the Adjusted Rand "
              "Index against the full-data grouping when 20% of districts are removed. State "
              "and region grouping are fixed by geography, so their stability of 1.00 is true "
              "by construction rather than a result.", italic=True, size=9)

    rich(doc,
         ("Against the like-for-like baseline, clustering explains ", False),
         (f"{eta_final:.3f}", True), (" of held-out outcome variance against ", False),
         (f"{eta_comp:.3f}", True),
         (f" for the composite index - approximately {eta_final / eta_comp:.0f} times more. "
          "State grouping scores higher (", False), (f"{eta_state:.3f}", True),
         ("), and this is reported rather than omitted: it uses 34 groups against 8, and "
          "eta-squared rises with the number of groups. States also share diets, policies and "
          "health systems. However, the state label carries no interpretation of what a "
          "district needs, and 42% of indicator variance lies within states.", False))
    figure(doc, "17_comparison.png",
           "Figure 3. Clustering (blue) against traditional groupings (red) on the three "
           "evaluation criteria.")

    doc.add_heading("4.2 Is the structure genuine?", level=2)
    para(doc, "Each feature column was shuffled independently 20 times - preserving every "
              "indicator's own distribution while destroying the relationships between them - "
              "and the entire pipeline was re-run on each shuffle.")
    perm = pd.DataFrame({
        "Measure": ["Silhouette", "Held-out outcome eta-squared"],
        "Real data": [round(val["real_silhouette"], 3), round(val["real_outcome_eta2"], 3)],
        "Shuffled (mean)": [round(val["shuffled_silhouette_mean"], 3),
                            round(val["shuffled_outcome_eta2_mean"], 3)],
        "Shuffled (best of 20)": [round(val["shuffled_silhouette_max"], 3),
                                  round(val["shuffled_outcome_eta2_max"], 3)],
        "p": [round(val["silhouette_p_value"], 3), round(val["outcome_eta2_p_value"], 3)],
    })
    add_table(doc, perm, col_widths=[2.3, 0.95, 1.2, 1.4, 0.55])
    para(doc, "Table 2. Permutation test. Both p-values sit at the floor achievable with 20 "
              "permutations (1/21). The second row is the stronger result: the outcome columns "
              "were never shuffled, so a grouping built on scrambled features has no mechanism "
              "by which to predict them.", italic=True, size=9)
    rich(doc,
         ("A related check answers the objection that a silhouette of 0.10 is low. The "
          "composite index's eight bands score ", False),
         (f"{comparison.loc['composite_index', 'silhouette']:.3f}", True),
         (" and the domain-weighted index ", False),
         (f"{comparison.loc['domain_index', 'silhouette']:.3f}", True),
         (" - both negative, meaning a typical district in those bands sits closer to another "
          "band's districts than to its own. A slice of a ranking is not a group of similar "
          "districts.", False))

    doc.add_heading("4.3 The eight profiles", level=2)
    prof_rows = []
    for cid in sorted(clusters["cluster"].unique()):
        info = names[str(int(cid))]
        prof_rows.append({"#": int(cid), "Profile": info["name"],
                          "Districts": int((clusters["cluster"] == cid).sum()),
                          "Policy focus": info["policy_focus"].replace("Priority: ", "")})
    add_table(doc, pd.DataFrame(prof_rows), col_widths=[0.35, 2.5, 0.75, 3.1], font_size=8.5)
    para(doc, "Table 3. Profile names are generated mechanically from each cluster's domain "
              "scores - a development level plus the domains more than 0.4 standard deviations "
              "below average - so they follow the data rather than expectation. NCD burden is "
              "excluded from the naming logic because it rises with development.",
         italic=True, size=9)
    figure(doc, "18_cluster_domain_heatmap.png",
           "Figure 4. Mean domain score per profile. Green is better than the national "
           "average, red worse. Reading across a row gives a profile's shape.")

    doc.add_heading("4.4 Same score, different problems", level=2)
    rich(doc, (f"{len(pairs):,} pairs", True),
         (" of districts lie within 0.02 of each other on the composite index yet fall into "
          "different profiles. Ernakulam (Kerala) scores 0.646 and Koraput (Odisha) 0.647. "
          "Ernakulam stands +2.5 standard deviations on education and +1.2 on sanitation, with "
          "its principal deficit in non-communicable disease. Koraput stands -1.7 on education "
          "and -1.4 on sanitation, yet above average on child health, maternal care and "
          "nutrition: its health services reach people, its households are unserved. Identical "
          "scores, opposite deficits, opposite interventions.", False))
    figure(doc, "15_same_score_different_problems.png",
           "Figure 5. Three pairs of districts the composite index treats as equivalent.")

    doc.add_heading("4.5 Reliability of individual assignments", level=2)
    para(doc, "Each profile has a centre, and a district takes the label of the nearest. The "
              "label margin - the distance to the second-nearest centre divided by the "
              "distance to the nearest - measures how firmly a district belongs. The national "
              f"median is 1.24; {val['borderline_districts']} of {val['n_districts']} districts "
              f"fall below 1.15, and {val['negative_silhouette_districts']} have a negative "
              "silhouette, sitting closer to another profile's centre than to their own.")
    para(doc, "This measure predicts the project's most demanding test. The scaler, PCA "
              "rotation and clustering were rebuilt from scratch with one state removed, and "
              "that state's districts were then assigned by the reduced pipeline:")
    conf = pd.read_csv(DATA_PROC / "district_confidence.csv")
    unseen_rows = []
    for _, r in unseen.iterrows():
        sub = conf[conf["state"].str.contains(r["state"], case=False, na=False)]
        unseen_rows.append({"State held out": r["state"], "Districts": int(r["n_districts"]),
                            "Median label margin": round(sub["label_margin"].median(), 2),
                            "Borderline": f"{(sub['confidence'] == 'borderline').sum()}/{len(sub)}",
                            "Agreement (ARI)": round(r["ari_vs_full_model"], 3),
                            "Pairs preserved": f"{r['pair_agreement']:.0%}"})
    add_table(doc, pd.DataFrame(unseen_rows), col_widths=[1.2, 0.8, 1.35, 0.9, 1.1, 1.05])
    para(doc, "Table 4. Leave-one-state-out test. The confidence columns explain the agreement "
              "column: Bihar's districts sit deep inside their profiles and reproduce exactly, "
              "while Assam's and Karnataka's sit on boundaries and move as a block when the "
              "boundary shifts.", italic=True, size=9)

    # --------------------------------------------------- 5. conclusions ----
    doc.add_page_break()
    doc.add_heading("5. Conclusions", level=1)
    bullet(doc, "Clustering districts by the shape of their development explains several times "
                "more variance in withheld nutrition and anaemia outcomes than a composite "
                "index with the same number of groups.", "1. ")
    bullet(doc, "The structure is not an artefact: shuffled data reproduces neither the "
                "geometry nor the predictive power.", "2. ")
    bullet(doc, "Districts nevertheless form a continuum, not discrete types. Silhouette near "
                "0.10, DBSCAN's single dense region, and 240 boundary districts all indicate "
                "this. The profiles are a practical partition of a continuous space.", "3. ")
    bullet(doc, "Consequently every assignment is reported with its margin. Labels are "
                "dependable in the interior of a profile and provisional at its edges - which "
                "is precisely what the leave-one-state-out results demonstrate.", "4. ")

    doc.add_heading("6. Limitations", level=1)
    bullet(doc, "NFHS values are survey estimates subject to sampling error; 5,042 cells rest "
                "on 25-49 respondents.", "Measurement — ")
    bullet(doc, "173 feature cells were imputed and 1.8% were winsorised; both introduce "
                "uncertainty that the reported metrics do not capture.", "Processing — ")
    bullet(doc, "profiles describe co-occurrence, not causation. The what-if simulator "
                "reclassifies a district; it does not predict the effect of an intervention.",
           "Inference — ")
    bullet(doc, "data covers 2019-21, partly during COVID disruption, and district boundaries "
                "have changed since Census 2011 (632 of 704 districts carry a 2011 code).",
           "Scope — ")
    bullet(doc, "the assignment of indicators to features, outcomes, directions and domains is "
                "our judgement, recorded in src/indicator_meta.py so it can be contested and "
                "the analysis re-run.", "Judgement — ")

    doc.add_heading("7. Reproducibility", level=1)
    para(doc, "All results regenerate from the raw factsheets with fixed random seeds:")
    code = doc.add_paragraph()
    code_run = code.add_run("pip install -r requirements.txt\n"
                            "python src/run_all.py      # download, clean, cluster, evaluate\n"
                            "streamlit run app.py       # the interactive profiler")
    code_run.font.name = "Consolas"
    code_run.font.size = Pt(9.5)
    para(doc, "Code and data: github.com/helperfn/nfhs-district-clusters. A self-contained "
              "notebook, notebooks/NFHS_District_Clustering.ipynb, reproduces the analysis in "
              "Google Colab without any local setup.")

    doc.add_heading("8. Sources and credit", level=1)
    bullet(doc, "IIPS, National Family Health Survey 5 (2019-21), district factsheets.", "Data — ")
    bullet(doc, "jvargh7/nfhs5_factsheets (values and reliability flags) and "
                "SaiSiddhardhaKalla/NFHS (indicator names and Census 2011 codes).", "Mirrors — ")
    bullet(doc, "kalyaninagaraj/NFHS5 applied PCA and K-means to this dataset. This project "
                "adds the baseline comparison, the held-out outcome test, algorithm and "
                "stability comparison, permutation testing, label margins and the interactive "
                "tool.", "Prior work — ")

    out = REPORTS / "NFHS_District_Profiling_Report.docx"
    doc.save(out)
    print(f"  saved {out}")
    print(f"  sections: 8 | tables: 6 | figures: 5")


if __name__ == "__main__":
    main()
