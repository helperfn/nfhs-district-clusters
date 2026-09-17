# District Development Profiling from NFHS-5

Clustering India's ~700 districts into **development profiles** from the National
Family Health Survey (NFHS-5, 2019-21), and testing whether those profiles are
more useful than the way districts are normally grouped: a single composite
score, or the state they happen to be in.

---

## 1. The project in plain words

Every district in India gets about 100 health and development numbers from NFHS-5.
The normal thing to do is squash them into **one score** and rank districts 1st to
704th. That tells you how far behind a district is. It never tells you *what is
wrong with it*.

So instead we grouped districts by the **shape** of their problems - 8 development
profiles - and then had to prove the grouping was worth anything. The proof:

> We **hid 9 numbers** from every model - child stunting, wasting, underweight,
> overweight, and anaemia in children and women. No grouping on this page ever saw
> them. Then we asked: which grouping best separates districts on the numbers it
> was never shown?

Our clustering explains **0.233** of the variation in those hidden numbers. The
composite index, cut into the same number of groups, explains **0.034**. About
seven times better, on evidence neither of them was allowed to look at.

Three things worth knowing before you read further:

* The 77 features used to build the clusters are living conditions (electricity,
  toilets, cooking fuel), service use (antenatal visits, immunisation, family
  planning) and behaviours (tobacco, alcohol, breastfeeding).
* Child health **is** in the features - vaccination coverage, diarrhoea and ARI.
  Only the nutrition and anaemia **outcomes** are held out, and they are held out
  to *test* the clusters, never to build them.
* Everything works at district level. "Group by state" is not a different unit of
  analysis - it is a rival **rule** for grouping the same 704 districts, scored
  exactly the same way.

## 2. Headline results

| Grouping | Groups | Held-out outcome variance explained (mean eta-squared) | Features that differ between groups | Stability (ARI) |
|---|---|---|---|---|
| **K-means on 30 PCA components (final model)** | 8 | **0.233** | 76 / 77 | 0.64 |
| K-means on the 77 standardised features | 8 | 0.248 | 76 / 77 | 0.62 |
| Ward (hierarchical) on raw features | 8 | 0.272 | 76 / 77 | 0.50 *(excluded - unstable)* |
| GMM on raw features | 8 | 0.265 | 76 / 77 | 0.50 *(excluded - unstable)* |
| Composite index, 8 quantile groups | 8 | 0.034 | 53 / 77 | 0.90 |
| Domain-weighted index, 8 quantile groups | 8 | 0.086 | 64 / 77 | 0.17 |
| Region grouping | 6 | 0.206 | 75 / 77 | 1.00 (fixed) |
| State grouping | 34 | 0.512 | 77 / 77 | 1.00 (fixed) |
| DBSCAN | 2 + noise | 0.008 | 14 / 77 | 0.72 |
| K-means, k = 3 (the silhouette-optimal k) | 3 | 0.071 | 70 / 77 | 0.93 |

**What this shows**

- Against the like-for-like baseline (**same number of groups**), clustering
  explains **0.233** of held-out outcome variance versus **0.034** for the
  composite index. A single score ranks districts; it does not say what is wrong
  with them.
- **State grouping scores highest (0.512)** and the report says so plainly. It has
  34 groups against 8, and eta-squared rises mechanically with more groups, so this
  is not a like-for-like comparison. States also share diet, policy and health
  systems. But "Bihar" does not tell a health officer what to fix, and **42%** of
  the variance in a typical indicator sits *within* states.
- **5,632 pairs of districts** sit within 0.02 of each other on the composite index
  yet fall in different profiles - plotted in
  `reports/figures/15_same_score_different_problems.png`.
- Two models scored marginally higher on the held-out test (Ward 0.272, GMM 0.265)
  but were **excluded for instability**: their groupings barely survive dropping 20%
  of districts (ARI 0.50). The stability floor is the same 0.60 used to choose k.
- DBSCAN is the honest negative result: districts form one continuous cloud, not
  separated blobs, so a density method finds one big cluster and a few outliers.

**The eight profiles** (names generated from the domain profiles, stored in
`config/cluster_names.json` and editable by hand):

| # | Profile | Districts |
|---|---|---|
| 0 | Middle development, strong clean energy | 42 |
| 1 | Low development, high tobacco & alcohol use and weak maternal care | 53 |
| 2 | High development, strong clean energy | 101 |
| 3 | High development, strong schooling | 51 |
| 4 | Middle development, weak schooling | 149 |
| 5 | Middle development, no standout gap | 104 |
| 6 | Low development, weak clean energy and weak civil registration | 116 |
| 7 | Middle development, weak women's agency | 88 |

Full descriptions, typical districts and policy focus: `reports/cluster_profiles.md`.

## 3. How much to trust a district's label

Every profile has a centre, and a district takes the label of the nearest one. So
for each district we record the **label margin**: how much further away the
second-nearest profile is.

* margin 2.0 = the runner-up is twice as far, the district is deep inside its profile
* margin 1.0 = the district sits exactly between two profiles and got its label by a hair

National median: **1.24x**. **240 of 704 districts** sit below 1.15x, i.e. on a
boundary. The app shows this on every district (tab 1) and warns when it is low.

This single number explains the **unseen-state test**, where the scaler, PCA
rotation and clustering are all refitted from scratch with one state removed and
that state's districts are then assigned by the reduced pipeline:

| State held out | Districts | Median label margin | ARI vs full model | Pair agreement |
|---|---|---|---|---|
| Bihar | 38 | 1.46 | 1.000 | 100% |
| Karnataka | 30 | 1.16 | 0.303 | 67% |
| Assam | 33 | 1.14 | 0.169 | 59% |

Bihar's districts sit deep inside their profiles, so rebuilding the model without
them changes nothing. Assam's and Karnataka's are mostly boundary cases, so they
move as a block. **That is not a broken model - it is the continuum showing
through, and it tells you which districts' labels to trust.**

## 4. Data

| Source | Use |
|---|---|
| [jvargh7/nfhs5_factsheets](https://github.com/jvargh7/nfhs5_factsheets) (`districts.csv`) | **values and reliability flags** - already numeric, and keeps NFHS's "based on 25-49 cases" / "not shown, fewer than 25 cases" marks for every indicator |
| [SaiSiddhardhaKalla/NFHS](https://github.com/SaiSiddhardhaKalla/NFHS) (`India.csv`) | Census 2011 district codes, clean indicator names, and the reference the other file is aligned against |
| [IIPS NFHS factsheets](http://rchiips.org/nfhs/) | the original source both mirrors were parsed from |
| [kalyaninagaraj/NFHS5](https://github.com/kalyaninagaraj/NFHS5) | prior work (PCA + K-means on the same data). This project adds baseline comparison, a held-out outcome test, algorithm comparison, stability testing, label margins and an interactive tool. Credited with thanks. |

**How the two mirrors are aligned.** Not by indicator name - the factsheet mirror's
names come straight out of the PDFs, and the gender of the NCD indicators survives
only in the factsheet **item number** (items 86-88 are women's blood sugar, 89-91
men's; the names are identical). Instead each item is matched to the India.csv
indicator whose **values** are closest across all shared districts. Both files were
parsed from the same PDFs, so the true match agrees to ~0.001 while the runner-up is
off by a thousand times more; the code asserts that gap rather than trusting the
match. See `reports/source_alignment.csv`.

After cleaning: **704 districts x 77 clustering features + 9 held-out outcomes.**

## 5. How to run

```bash
pip install -r requirements.txt

python src/run_all.py        # every phase in order, ~3 minutes
streamlit run app.py         # the interactive tool
```

Or one phase at a time (this is also the dependency order):

```bash
python src/download.py    # 1  fetch both mirrors, describe them
python src/clean.py       # 2  align sources, pivot, impute, winsorise, de-duplicate
python src/eda.py         # 3  figures 01-04 + reports/eda_summary.md
python src/features.py    # 4  standardise + PCA, figures 05-08
python src/cluster.py     # 6  4 algorithms x 2 spaces, choose k, figures 09-13
python src/baselines.py   # 5  composite / domain / state / region, figure 14
python src/evaluate.py    # 7  the comparison, figures 15-17
python src/profile.py     # 8  cluster profiles, figures 18-19
python src/predict.py     # 9  assign_cluster() + unseen-state test
```

`cluster.py` runs before `baselines.py` because the baselines are cut into the same
number of groups as the final clustering - otherwise the comparison would not be fair.

There is also a self-contained Colab notebook, `notebooks/NFHS_District_Clustering.ipynb`,
which downloads its own data and reproduces every number above in ~4 minutes.

## 6. What each part does

```
src/download.py        fetch both mirrors, report format and coverage
src/clean.py           align the mirrors by value fingerprint, pivot to wide,
                       drop sparse columns/rows, KNN-impute, winsorise,
                       drop near-duplicate indicators
src/indicator_meta.py  the judgement layer: domain, direction and role for all
                       104 indicators -> config/indicators.csv
src/eda.py             missingness, distributions, correlation blocks,
                       within-state spread
src/features.py        StandardScaler + PCA, component naming, scree/loadings
src/baselines.py       the four traditional groupings
src/cluster.py         K-means, Ward, GMM, DBSCAN on raw and PCA features;
                       chooses k by reproducibility
src/evaluate.py        internal metrics, held-out outcome test, distinctness,
                       bootstrap stability, ARI agreement, same-score pairs,
                       final model choice (with a stability floor)
src/profile.py         domain profiles, radar charts, automatic cluster names
src/predict.py         assign_cluster() for new input + unseen-state test
                       + NFHS-4 pass-through
app.py                 5-tab Streamlit tool, all results computed live
```

Key decisions written up in `reports/`: `source_alignment.csv`, `cleaning_report.md`,
`pca_components.md`, `final_choice.md`, `cluster_profiles.md`, `unseen_data.md`.
Teaching material: `LEARNING.md` (concepts + viva prep), `CODE_EXPLAINED.md`
(line by line), `APP_GUIDE.md` (the app and the demo script).

## 7. Limitations

- **Survey estimates, not a census.** Every NFHS value carries sampling error.
  5,042 cells are flagged as based on 25-49 respondents; 4,113 cells the factsheets
  suppressed entirely (fewer than 25 respondents) are treated as missing.
- **Consistent flagging changes what survives.** Now that every indicator is flagged
  the same way, anaemia in **pregnant women** turns out to be suppressed in 19% of
  districts and is dropped, leaving 9 held-out outcomes rather than 10. Nine features
  with more than 10% missing - mostly small-denominator ones like diarrhoea treatment -
  were dropped rather than imputed.
- **Imputation adds uncertainty.** 173 feature cells were filled with KNN (k=5).
- **Winsorising.** 1.8% of feature cells were clipped to the 1st/99th percentile,
  because single extreme values from very small districts otherwise pulled whole
  clusters onto themselves. The unclipped matrix is kept at
  `data/interim/features_unclipped.csv`.
- **Association, not cause.** A profile describes which conditions occur together.
  It does not show that fixing one would fix another. The what-if simulator
  re-*classifies* a district; it does not predict what an intervention would achieve.
- **One point in time.** NFHS-5 was collected 2019-21, partly during COVID
  disruption. The NFHS-4 comparison in `reports/unseen_data.md` is indicative only:
  NFHS-4 lacks several NFHS-5 indicators, which are imputed.
- **Judgement is unavoidable.** Which indicators are features, which are held-out
  outcomes, each one's direction and domain - all our decisions, written down in
  `src/indicator_meta.py` so they can be argued with and changed.
- **Boundaries move.** District boundaries have changed since Census 2011; only 632
  of 704 districts carry a 2011 code, and the factsheet mirror uses the merged UTs
  (34 states/UTs rather than 37).
- **Clusters are groups, not grades.** Silhouette around 0.10, DBSCAN's single blob
  and 240 boundary districts all say the same thing: this is a continuum. The
  profiles are practical labels cut across it, not seven or eight natural kinds of
  district - which is why every label comes with a margin.
