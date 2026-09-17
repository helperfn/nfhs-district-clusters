# LEARNING.md - what we built, in plain English, plus viva prep

---

# Part 0 - The whole project in one page

**The problem.** Every district in India gets ~100 health and development numbers
from NFHS-5 (2019-21). Policy squashes them into one score and ranks districts 1st
to 704th. That says how far behind a district is. It never says *what is wrong with
it*. Two districts with the same score can need opposite things.

**What we did.** Grouped districts by the **shape** of their problems instead of the
level - 8 development profiles. Nobody labelled districts in advance, so the
structure had to come out of the data. That is clustering, and it is *unsupervised*
learning.

**What went into the grouping.** 77 indicators covering:
- living conditions - electricity, toilets, drinking water, clean cooking fuel
- service use - antenatal visits, institutional delivery, immunisation, family planning
- behaviours and status - tobacco, alcohol, women's schooling, age at marriage
- NCD burden - blood sugar, blood pressure, cancer screening

**What we deliberately kept out, and why.** Nine outcomes: child stunting, wasting,
severe wasting, underweight, overweight, and anaemia in children, women 15-19,
women 15-49 and non-pregnant women.

These were **hidden from every model**. Not to cluster them - to *test* the
clusters. If profiles built only from living conditions and services also separate
districts on stunting and anaemia, numbers the model never saw, then the profiles
capture something real rather than an artefact of which columns we fed in.

> Careful here, it is the thing people get wrong: **child health is in the
> features** - vaccination coverage, diarrhoea and ARI prevalence are all used to
> build the clusters. Only the nutrition and anaemia *outcomes* are held out.

**What we compared against.** Everything works on the same 704 districts. The
baselines are rival **rules for grouping them**:

| Rival | What it does |
|---|---|
| composite index | flip the negative indicators, scale each to 0-1, average all 77 into one score, cut into 8 equal groups. This is how the Aspirational Districts Programme and SDG district indices rank districts. |
| domain-weighted index | average within each of the 11 domains first, then across domains, so a domain with 15 indicators does not outvote one with 2. Then the same 8-way cut. |
| state grouping | put each district in its state - how money is actually allocated |
| region grouping | North / South / East / West / Central / North-East |

**And the models we compared.** K-means, Ward (hierarchical), Gaussian Mixture and
DBSCAN, each on two versions of the data: the 77 standardised indicators, and the
PCA components.

**The result.** Clustering explains **0.233** of the hidden outcomes' variance; the
composite index, with the same number of groups, explains **0.034**. State grouping
explains more (0.512) but with 34 groups against 8, which is not a fair comparison,
and it still cannot say what a district needs.

**The honest catch.** Districts do not fall into 8 neat boxes. They are spread along
a smooth continuum, so the profiles are practical labels cut across it - useful for
targeting, not a claim that India has eight natural kinds of district. Which is why
every district also carries a **label margin** saying how safe its label is.

---

# Phase 1 - Download and inspect

**What happens.** Two mirrors of the same NFHS-5 factsheets are downloaded, and each
is used for what it is good at.

* **`districts.csv` (jvargh7) is the source of values.** Already numeric, and it keeps
  NFHS's reliability marks in their own column.
* **`India.csv` (SaiSiddhardhaKalla) supplies Census 2011 codes and clean indicator
  names**, and is the reference the other file is aligned against.

Both are **long** format: one row per (district, indicator). Machine learning needs
**wide**: one row per district.

**Viva questions**

1. *Why two sources?* One has trustworthy values and flags; the other has
   trustworthy names and census codes. Using each for its strength is the whole point.
2. *Why not the IIPS PDFs directly?* They are ~700 separate PDF factsheets. The
   mirrors are those PDFs already parsed, and we cross-check one against the other.
3. *How many districts?* 705 in the factsheet mirror, 704 after cleaning.

---

# Phase 2 - Cleaning

### The reliability flags

NFHS prints an estimate based on 25-49 respondents in brackets, and refuses to print
one based on fewer than 25. The factsheet mirror preserves both facts:

| Flag | meaning | what we do |
|---|---|---|
| blank | ordinary estimate | keep |
| "Based on 25-49 unweighted cases" | small sample | keep, but flag it (5,042 cells) |
| "Percentage not shown; fewer than 25" | suppressed by NFHS | already blank -> missing (4,113 cells) |

**An earlier version of this project got this wrong, and it is worth knowing why.**
It took values from India.csv and joined the flags on by indicator *name*. Only 82%
of cells matched, and the misses were not random: 17 indicators - including child
anaemia and the whole blood-pressure / blood-sugar block - matched nothing at all. So
small-sample estimates were removed for some indicators and kept for others. Same
data, inconsistent treatment. If an examiner asks "why is anaemia never suppressed
but stunting is?", "a text-matching artefact" is not an answer you want to give.

### Aligning the two mirrors by value, not by name

The factsheet mirror's names come straight out of the PDFs, and two things go wrong:
the gender of the NCD indicators lives **only in the item number** (items 86-88 are
women's blood sugar, 89-91 men's - the names are identical), and a few names are
garbled by the text extraction.

So each factsheet item is matched to the India.csv indicator whose **values** are
closest across all shared districts. Both files were parsed from the same PDFs, so
the true match agrees to about 0.001 while the runner-up is off by a thousand times
more - and the code *asserts* that gap rather than trusting the match. All 104
indicators aligned unambiguously.

### The rest

1. **Missingness rules.** Drop an indicator missing in >10% of districts (9 dropped,
   all small-denominator: diarrhoea treatment, non-breastfed infant diet); drop a
   district missing >20% of indicators (1: Jabalpur).
   Consistent flagging changes what survives: **anaemia in pregnant women** is
   suppressed in 19% of districts - pregnant women are a small subgroup in a district
   sample - so it is dropped, leaving **9** held-out outcomes.
2. **KNN imputation (k=5) on standardised values**: a gap is filled with the average
   of the 5 most similar districts (173 cells). Standardising first matters because
   "similar" is a distance, and an indicator in rupees would drown out percentages.
3. **Winsorising** at the 1st/99th percentile (1,002 cells, 1.8%), so one very small
   district cannot claim a whole cluster.
4. **Near-duplicate removal** at |r| > 0.95 (2 dropped: `polio3`, `adequate_diet_total`).

**Features and outcomes are imputed separately** - if outcome columns helped fill
feature columns, the held-out test would be contaminated.

**Viva questions**

1. *Why align by value instead of name?* Because the names are unreliable in exactly
   the place it matters - the men's and women's NCD rows have identical names. Values
   are the ground truth both files share.
2. *Why drop the 9 sparse indicators instead of imputing?* Imputing 68% of a column
   invents most of it; the "data" becomes the model's own guesses.
3. *Why KNN and not the mean?* Mean imputation pulls every gap to the middle and
   shrinks variance. KNN uses districts that actually resemble this one.
4. *Is winsorising cheating?* It changes 1.8% of cells and is declared. Without it,
   Chandigarh alone formed a cluster - one district is not a development profile.

---

# Phase 3 - EDA

Four figures, each justifying a later decision: the missing heatmap (why we dropped
columns), histograms (why we standardise), the domain-ordered correlation heatmap
(why PCA), and state boxplots (why cluster districts, not states).

**The two numbers to remember**
- Mean |correlation| between features = **0.22**, with **124 pairs above 0.6**.
- **42%** of the variance in a typical indicator is *within* states.

**Viva questions**

1. *What motivates PCA here?* The correlation heatmap: thick blocks of indicators
   measuring the same underlying thing.
2. *Why not just group by state?* Because 42% of the variation is inside states.
3. *Why are distributions skewed?* Many indicators are bounded percentages that pile
   up near 0 or 100 (electricity, BCG coverage).

---

# Phase 4 - Scaling and PCA

`StandardScaler` puts every feature on mean 0, sd 1. PCA then rotates the 77
correlated features onto uncorrelated axes ordered by variance explained. PC1 alone
explains **25.3%**. Reaching 80 / 90 / 95% needs **18 / 30 / 41** components. All
three are compared in Phase 7.

**What the first four components actually mean** (read off their biggest loadings):

| | Meaning | Share |
|---|---|---|
| PC1 | reach of maternal care: iron tablets, antenatal and postnatal visits at one end; young population, high-order births, home deliveries, tobacco at the other | 25.3% |
| PC2 | public immunisation reach vs a costlier, more privatised system (high out-of-pocket delivery costs, more hypertension, more schooling) | 11.1% |
| PC3 | which lifestyle disease dominates - high blood sugar vs high blood pressure | 7.0% |
| PC4 | tobacco and alcohol vs household basics (clean water, cooking fuel) - broadly the North-East pattern | 5.2% |

**Viva questions**

1. *What does PCA do?* Finds the direction of greatest variance, then the next at
   right angles, and so on. Each component is a weighted mix of the original indicators.
2. *Why standardise first?* PCA maximises variance, so without scaling the
   indicator with the biggest numbers would define PC1 by itself.
3. *Why keep 30 components and not 2?* Two are for the picture. The model needs the
   information. Phase 7 tests 18 / 30 / 41 and the differences are small.
4. *Does PCA lose information?* Yes, deliberately - the discarded directions are
   mostly noise.

---

# Phase 5 - Traditional baselines

Four rival groupings, all cut into **8 groups** - the same k as the clustering, so
the comparison is like-for-like. The two indices correlate at Spearman **0.85**: a
single score is a single score however you weight it.

**Viva questions**

1. *Why flip the negative indicators?* You cannot average "literacy 80%" with
   "stunting 40%" while they point in opposite directions. After flipping and
   min-max scaling, 1 always means best.
2. *Why give the baselines the same number of groups?* Because eta-squared rises
   mechanically with more groups. Same k = the difference is about *how* you group.
3. *Are these fair baselines?* Yes - a composite index is exactly how the
   Aspirational Districts Programme ranks districts.

---

# Phase 6 - Clustering

**The four algorithms**
- **K-means** - pick k centres, assign each district to the nearest, move each centre
  to its members' mean, repeat. `n_init=50` because the result depends on the random start.
- **Ward (hierarchical)** - start with every district alone, repeatedly merge the pair
  that increases within-cluster variance least. Gives the dendrogram. A merge can
  never be undone, which is why it is less stable here (0.50 vs K-means' 0.64).
- **GMM** - assumes a blend of Gaussian blobs and gives each district a *probability*
  of belonging to each. Full covariance on PCA features, diagonal on the 77 raw ones.
- **DBSCAN** - grows clusters from dense regions, labels sparse points as noise. Here
  it finds one giant cluster plus a few outliers - direct evidence of a continuum.

**Choosing k = 8.** The silhouette curve is flat (0.092-0.110 for every k from 3 to
10, best at k=4), so reading a winner off 0.01 differences would be fitting noise.
The rule used instead, fixed in advance and using no outcome data: every cluster must
have >= 15 districts; the solution must be reproducible (mean bootstrap ARI >= 0.60
over 30 x 80% subsamples); take the largest k before stability first breaks. k = 3-8
pass, k = 9 fails, so **k = 8** (ARI 0.64).

**Viva questions**

1. *Why is the silhouette so low (~0.10)?* Because there are no natural gaps between
   districts. The clusters are a useful partition of a continuum - and the report
   says exactly that.
2. *Why not take the best silhouette (k=3 or 4)?* The differences are inside the
   noise, and three broad tiers carry far less information about *what kind* of
   problem a district has. k=3 is still in the comparison table, scoring 0.071 on the
   held-out test against k=8's 0.233.
3. *What is the Adjusted Rand Index?* Agreement between two groupings of the same
   items, corrected for chance: 1 = identical, 0 = no better than random.
4. *Why does DBSCAN fail?* It needs density differences. This data has one dense
   cloud, so eps either swallows everything or marks a third of districts as noise.

---

# Phase 7 - Evaluation

**Five tests, every grouping scored identically.**

1. **Internal quality** - silhouette, Davies-Bouldin, Calinski-Harabasz, all in the
   same 77-feature space.
2. **Held-out outcome test (the key one)** - nine nutrition and anaemia indicators no
   grouping ever saw. eta-squared = the share of each outcome's variance explained by
   the grouping (one-way ANOVA, SS_between / SS_total). Clustering **0.233**,
   composite index **0.034**, state grouping **0.512**.
3. **Profile distinctness** - Kruskal-Wallis per feature, Bonferroni-corrected at
   0.01/77. Clustering separates 76 of 77 features; the composite index, 53.
4. **Stability** - 50 bootstrap samples of 80% of districts, ARI against the full-data
   labels. Index baselines are rebuilt the same way; state and region are
   deterministic, so their 1.00 is by construction and is labelled as such.
5. **Agreement** - ARI between every pair of groupings (figure 16).

**Final model: K-means on 30 PCA components, k = 8.** The rule, in order:

1. a **stability floor** - mean bootstrap ARI >= 0.60, the same floor used to choose k;
2. held-out eta-squared, with gaps under 0.02 treated as ties;
3. stability, then silhouette.

The floor matters. Ward on raw features (0.272) and GMM on raw features (0.265) score
marginally higher on the held-out test but are barely reproducible (both 0.50), and
they were excluded. When an earlier version of the rule let GMM win, the unseen-state
test in Phase 9 collapsed - because "refit without one state" and "refit without 20%
of districts" are the same question.

**Why the held-out test is fair.** The outcomes were separated in Phase 2, imputed
separately, never scaled with the features, never used to choose k, and never used to
choose the algorithm. Nothing in the pipeline could have peeked.

**Viva questions**

1. *What is eta-squared?* The proportion of a variable's variance explained by group
   membership; 0.23 means the profiles account for 23% of the differences between
   districts on outcomes they never saw.
2. *State grouping beat your model. Doesn't that sink the project?* It has 34 groups
   against 8 - eta-squared rises mechanically with more groups - and states share
   diet, policy and health systems. But "Bihar" does not tell a health officer what to
   do, and 42% of the variance is within states. The like-for-like comparison is
   against the composite index at the same k, where clustering wins about 7 to 1.
3. *Why Kruskal-Wallis instead of ANOVA?* Several indicators are skewed;
   Kruskal-Wallis is the rank-based version and assumes no particular distribution.
4. *Why Bonferroni?* 77 tests at p<0.01 would produce false positives by chance;
   dividing the threshold by 77 controls that.
5. *What does "same score, different problems" show?* 5,632 district pairs sit within
   0.02 on the composite index but in different profiles. Figure 15 draws three.
   Example: **Ernakulam (Kerala) 0.646 and Koraput (Odisha) 0.647** - Ernakulam is
   +2.5 sd on education and +1.2 on sanitation but negative on NCDs; Koraput is -1.7
   on education and -1.4 on sanitation but *positive* on child health and nutrition.
   Same score, opposite problems, opposite interventions.

---

# Phase 8 - Cluster profiles

Every feature is z-scored and flipped so **positive always means better**, then
averaged within each of the 11 domains. Names are generated mechanically from the
profile (development level + the weakest domains), saved to
`config/cluster_names.json` for hand-editing - the app reads that file live.

NCDs are excluded from the naming logic (not from the charts): high blood sugar and
blood pressure *rise* with development, so "high development, weak NCDs" would be
nonsense. It is a real finding, not a bug.

**Viva questions**

1. *How were the names produced?* From the mean domain z-scores: a level word, then
   the domains more than 0.4 sd below average.
2. *Why z-scores rather than raw percentages?* So domains on different scales sit on
   one chart, and "above/below the national average" means the same thing everywhere.
3. *What is the most typical district in a cluster?* The one closest to the cluster
   centre in PCA space.

---

# Phase 9 - New districts, label margins, and the unseen-data test

`assign_cluster(values)` pushes any dict of indicator values through the saved
pipeline - impute, clip, scale, PCA, predict - refitting nothing, and returns the
cluster, the distance to every cluster centre, the five most similar real districts,
and which fields had to be imputed.

### The label margin

Every profile has a centre; a district takes the label of the nearest one. The margin
is *(distance to the 2nd-nearest centre) / (distance to the nearest)*:

* **2.0** - the runner-up is twice as far. Deep inside its profile, nothing moves it.
* **1.0** - exactly between two profiles. It got its label by a hair.

National median **1.24x**; **240 of 704** districts sit below 1.15x. The most
borderline districts in the country are Jhajjar and Mahendragarh (Haryana), Nicobar,
Darjeeling and Goalpara (Assam), all at ~1.00. The most solidly placed is Madurai at 2.39.

### The unseen-state test

Refit **everything** - scaler, PCA rotation and clustering - without one state, then
assign that state's districts with the reduced pipeline:

| State | Median label margin | ARI vs full model |
|---|---|---|
| Bihar | 1.46 | 1.000 |
| Karnataka | 1.16 | 0.303 |
| Assam | 1.14 | 0.169 |

Read the two columns together and the result explains itself. Bihar's districts are
interior, so rebuilding without them changes nothing. Assam's and Karnataka's are
mostly boundary cases, and boundary districts do not flip independently - they flip
as a block, because they are similar to each other and all sit on the same side of
the same line.

**Viva questions**

1. *Can you validate an unsupervised model?* Not with accuracy - there are no labels.
   You can test whether it reproduces itself on data it never saw (the unseen-state
   test) and whether it separates held-out outcomes (the eta-squared test).
2. *Why does Assam do badly?* Its districts sit near a profile boundary. The margin
   column predicts the ARI column. It is a statement about which labels to trust,
   not a broken model.
3. *Why is the bootstrap so much kinder than the state test?* Random resampling
   removes districts scattered everywhere, so the boundaries barely move, and it is
   judged on 560 mixed districts. Removing a whole state cuts out a contiguous chunk
   and is then judged on only 30 similar districts, which move together.
4. *What happens with missing input?* The saved KNN imputer fills it, and the app
   lists every field it filled.

---

# Phase 10 - The app

Five tabs: explore a district (profile, **label confidence**, radar vs cluster and
national average, composite score for contrast, 5 nearest districts, PCA scatter);
enter values (form or CSV upload, with imputation reported); what-if simulator (12
sliders - four policy levers plus the eight most influential indicators - with the
cluster recomputed live); clusters overview; and "why clustering?", which shows the
Phase 7 comparison, the same-score pairs, the PCA variance curve, the stability curve
and the unseen-state results.

Models load through `st.cache_resource`, tables through `st.cache_data`. Nothing on
screen is hard-coded.

---

# The 10 answers to have ready

1. **Why clustering?** There are no labels. Nobody has tagged districts with their
   "true" development type - the structure has to come out of the data.
2. **Why standardise?** Distance-based methods otherwise follow whichever column has
   the biggest numbers.
3. **Why PCA?** 124 feature pairs correlate above 0.6; correlated columns
   double-count, and 77 dimensions make distances uninformative.
4. **How was k chosen?** Reproducibility, not the flat silhouette curve: the largest k
   whose clusters all have >= 15 districts and whose bootstrap ARI stays above 0.60 -
   k = 8.
5. **How was the algorithm chosen?** A stability floor first, then held-out outcome
   eta-squared, then stability, then silhouette. K-means on 30 PCA components.
6. **What do the metrics mean?** Silhouette: how much closer a district is to its own
   cluster than the next (higher better). Davies-Bouldin: spread over separation
   (lower better). Calinski-Harabasz: between- over within-cluster variance (higher
   better). ARI: agreement corrected for chance. eta-squared: variance explained by
   grouping.
7. **Why is the held-out test fair?** The nine outcomes were separated in Phase 2,
   imputed on their own, and never touched by scaling, PCA, k selection or model
   selection.
8. **What did the baselines show?** At the same number of groups the composite index
   explains 0.034 of held-out outcome variance against clustering's 0.233. State
   grouping explains more (0.512) but with 34 groups and no interpretation.
9. **What do the clusters mean?** Eight profiles differing in *shape*, not only level
   - see the table in README.md and `reports/cluster_profiles.md`.
10. **What are the weaknesses?** Survey error, imputation, a continuum with no natural
    gaps (silhouette ~0.10, 240 boundary districts), one time point, and the judgement
    calls in `src/indicator_meta.py`.
