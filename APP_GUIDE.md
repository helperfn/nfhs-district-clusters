# APP_GUIDE.md — what the app shows, and how to demo it

Three parts:

1. **What every element on screen means** (tab by tab, number by number)
2. **How to run it** — including what to do when the projector laptop misbehaves
3. **A timed demo script** with the exact districts to type and the words to say

---

# PART 1 — What everything on the app says

## The sidebar (always visible, left)

| Line | What it means | Say it like this |
|---|---|---|
| **Data: NFHS-5 (2019-21)** | India's National Family Health Survey, 5th round. NFHS-4 (2015-16) is used only for the change analysis, never for clustering. | "Government survey, every district, 2019 to 2021." |
| **Districts: 704** | Districts left after cleaning (705 in the source; one was missing too much data). | |
| **Clustering features: 77** | The indicators the model is allowed to use: living conditions, service use, behaviours. | |
| **Held-out outcomes: 9** | Child stunting, wasting, severe wasting, underweight, overweight, and anaemia in children, women 15-19, women 15-49 and non-pregnant women. **The model never sees these.** | "These are our exam paper — the model never gets to look at them." |
| **Final model: kmeans on 30 PCA components** | K-means clustering, run on 30 principal components instead of the raw 77 indicators. | |
| **Profiles (k): 8** | Eight development profiles. | |
| Source links | Both GitHub mirrors and the original IIPS factsheets. | |
| Caption at the bottom | "Association, not cause" + survey error. Leave it visible — it pre-empts the obvious criticism. | |

**If asked why child health isn't held out:** it isn't an outcome here. Vaccination
coverage, diarrhoea and ARI prevalence are *features* — they describe the services a
district has. Only the nutrition and anaemia **results** are held out, and only to
test the clusters.

---

## Tab 1 — "Explore a district"

**Two dropdowns:** state, then district. Picking a state filters the district list.

**Five boxes across the top**

| Box | Meaning | Watch out for |
|---|---|---|
| **Profile #** | Which of the 8 profiles this district belongs to. | A number, not a rank. Profile 6 is not "worse than" profile 3 — they're different shapes. |
| **Label confidence** | How much further away the second-nearest profile is. **1.00× = the district sits exactly between two profiles; 2.00× = the runner-up is twice as far.** National median 1.24×. | This is the headline honesty feature — see below. |
| **Composite score** | The traditional single-number index, 0 to 1, 1 = best. Shown **for contrast**, not because we use it. | This is the thing the project argues against. |
| **Composite quantile group** | Which eighth of the ranking it falls in (1 = worst eighth, 8 = best). | |
| **Districts in this profile** | How many of the 704 districts share this profile. | |

### The label confidence box — what to say about it

Every profile has a centre; a district takes the label of the nearest one. The
confidence figure is the ratio of the second-nearest distance to the nearest:

* **≥ 1.35×** → "solidly inside this profile"
* **1.15–1.35×** → "typical for this profile"
* **< 1.15×** → "near a boundary — treat as borderline", and a **warning box appears**
  naming the runner-up profile

240 of 704 districts fall in that last band. This is the number that explains the
unseen-state test in tab 5 (Bihar's districts have a median margin of 1.46 and
reproduce perfectly; Assam's 1.14 and don't).

**Below that:** the profile's name and its one-line policy focus, both read live from
`config/cluster_names.json`.

**Left — the radar chart.** Eleven spokes, one per domain. Three shapes overlaid:

* black = this district
* coloured = the average district in its profile
* grey circle at zero = the national average

**Every indicator is flipped so that outward = better.** A dent pointing inward is a
problem area, always, on every chart in the app. The rings are −1 SD, national
average, +1 SD.

**Right — three things:**

* **Five most similar districts**, by distance in PCA space (the 30-number summary of
  each district). The caption counts how many are in *other* states — the point being
  that similar problems don't stop at state borders.
* **Biggest gaps vs the national average** — the six indicators where this district is
  furthest below average, with its value, the national mean and the z-score. This is
  the "so what do we actually fix" table.

**Bottom — the PCA scatter.** Every district as a dot, placed by its first two
principal components, coloured by profile, selected district marked with a black star.
PC1 (horizontal) is broadly the reach of maternal and child services; PC2 (vertical)
separates public-immunisation-strong districts from higher-cost, more privatised ones.

> If asked "why only 2 axes when the model uses 30?" — the model uses 30; the picture
> uses the two biggest because a screen is flat.

---

## Tab 2 — "Enter district values"

For a district that is **not** in NFHS-5, or a hypothetical one.

* **Download a one-row CSV template** — all 77 columns pre-filled with the national
  average. Edit it, upload it back.
* **Form input** — number boxes grouped into collapsible sections by domain, pre-filled
  with the national average. Anything you leave alone counts as *given* data.
* **CSV upload** — columns you leave out are **imputed** from the five most similar
  districts. The more honest option for partial data.

**After you submit:**

| Element | Meaning |
|---|---|
| Green banner | The assigned profile and its name. |
| Policy focus line | The profile's standard intervention priority. |
| **Distance to each profile centre** | Sorted. If the top two are close, the district is borderline — say so out loud; it shows you understand the model. |
| **"N values were imputed"** expander | Exactly which indicators the model filled in. |
| **Most similar real districts** | The five closest actual districts, with their profiles. |
| PCA scatter | The entered district as a black **✕** among all 704 real districts. |

---

## Tab 3 — "What-if simulator"

Pick a real district; it loads that district's actual values. Twelve sliders appear,
chosen two ways: four **programme levers** a district administration can actually act
on (clean cooking fuel, sanitation, girls' schooling, institutional delivery), plus the
eight indicators with the largest weight on PC1 — the ones that genuinely move a
district's position. Sliders that barely move anything would make a pretty but useless
simulator.

| Element | Meaning |
|---|---|
| **Current profile** | Where the district sits today. |
| **Simulated profile** | Where it sits with your slider values — recomputed live, with a "moved"/"unchanged" tag. |
| **Indicators changed** | How many sliders you actually moved. |
| **Distance to each profile centre: before / after / change** | Negative change = moving *towards* that profile. Even when the label doesn't flip, this shows the district drifting. |
| **Domain profile before and after** | Two radar shapes: grey = now, red = simulated. |

**The honest caveat to say out loud:** this shows *where a district with those values
would be classified*, not what would happen if you built the stoves. Reclassification,
not causal prediction.

---

## Tab 4 — "Clusters overview"

* **Bar chart** — districts per profile.
* **Domain heatmap** — all 8 profiles × 11 domains. Green = better than the national
  average, red = worse, numbers are z-scores. **This single chart is the project's main
  result**: read across a row to see a profile's shape.
* **One expander per profile** — states most represented, the five most typical
  districts (closest to the profile centre), strengths, gaps, policy focus.

Worth knowing: the high-development profiles show **red on NCDs** — high blood sugar
and blood pressure rise with development. A real finding, not a bug, and the reason
NCDs are excluded from the automatic naming.

---

## Tab 5 — "Why clustering?" (the evaluation tab)

**The comparison table** — every grouping scored the same way:

| Column | Meaning | Direction |
|---|---|---|
| `n_groups` | How many groups the method makes | — |
| `silhouette` | How cleanly separated the groups are | higher better (all ~0.10 here — a continuum) |
| `davies_bouldin` | Spread ÷ separation | lower better |
| `mean_outcome_eta2` | **The key column.** Share of the nine held-out outcomes' variance explained | higher better |
| `distinct_features` | How many of the 77 indicators differ significantly between groups (Kruskal–Wallis, Bonferroni p<0.01) | higher better |
| `stability_ari` | Agreement with the full-data grouping when 20% of districts are dropped | higher better |

**The two bar charts** — blue = clustering, red = traditional baselines.

Numbers to have memorised:

* clustering **0.233** vs composite index **0.034** — same number of groups, ~7× better
* state grouping **0.512** — but 34 groups against 8; not a like-for-like comparison
* DBSCAN **0.008** — the honest negative result
* Ward (0.272) and GMM (0.265) scored *higher* but were **excluded for instability**
  (both 0.50, below the 0.60 floor) — the table shows them, the blue info box explains
  the exclusion

**Blue info box** — the exact selection rules for the model and for k, printed from the
config files so they can't drift from what the code actually did.

**"Same score, different problems"** — a dropdown of district pairs whose composite
scores are within 0.02 of each other but which fell in different profiles (5,632 of
them). Picking one draws both districts' domain profiles side by side.

**Bottom row** — the PCA variance curve (how many components reach 80/90/95%, with a
marker at the 30 we use), and the stability-vs-k curve (why k = 8: the last k before
reproducibility breaks).

**Unseen-state test table** — the pipeline rebuilt from scratch without each state:
Bihar 1.00, Karnataka 0.30, Assam 0.17. Do not hide the low rows; explain them with
the label margin (see the demo script).

---

# PART 2 — How to run it

## Normal start

```bash
cd nfhs-clusters
streamlit run app.py
```

A browser tab opens at **http://localhost:8501**. First load takes ~5 seconds while
the models load; after that everything is cached and instant. Stop with `Ctrl+C`.

## If the outputs are missing (fresh machine / after `git clone`)

```bash
pip install -r requirements.txt
python src/run_all.py        # ~3 minutes, regenerates everything
streamlit run app.py
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `FileNotFoundError: district_clusters.csv` | The pipeline hasn't been run. `python src/run_all.py`. |
| `ModuleNotFoundError: streamlit` | `pip install -r requirements.txt`. |
| Port 8501 already in use | `streamlit run app.py --server.port 8600`, then open that port. |
| Browser doesn't open by itself | Type `localhost:8501` into the address bar. |
| Sliders feel slow on an old laptop | Move one at a time; each move re-runs the assignment. |
| No internet in the exam hall | Fine — the app needs **no** internet. Only `src/download.py` does, and its output is already in `data/raw/`. |

## Before you walk into the room

1. Start the app **10 minutes early** and click through all five tabs once, so
   everything is cached.
2. Open `reports/figures/17_comparison.png` in a second browser tab as a fallback.
3. Keep `notebooks/NFHS_District_Clustering.ipynb` open in a third tab — the fallback
   that shows the code actually running.
4. Have `reports/final_choice.md` open for the numbers.

---

# PART 3 — The demo script (6 minutes)

Times are cumulative. Words in quotes are what you say. Every district and number
below was verified against the current model.

### 0:00 — Frame the problem (no app yet, 30 seconds)

> "NFHS-5 gives about 100 health numbers for every district in India. The normal thing
> is to squash them into one score and rank districts 1st to 704th. That tells you how
> far behind a district is. It never tells you what's wrong with it. We grouped
> districts by the *shape* of their problems instead, and then tested whether that beats
> the ranking."

### 0:30 — Tab 5, top table only (1 minute) — lead with the evidence

Open **"Why clustering?"** first. Point at `mean_outcome_eta2`.

> "We hid nine indicators — child stunting, underweight, anaemia. No model in this table
> ever saw them. This column is how much of those hidden numbers each grouping explains.
> Our clustering: 0.23. The traditional composite index, cut into the same eight groups:
> 0.03. Seven times better on evidence neither of them was allowed to see."

Then pre-empt the obvious question yourself — this always lands well:

> "The row at the top is state grouping, at 0.51. It does better — but it has 34 groups
> against our 8, and more groups always score higher on this measure. And knowing a
> district is 'in Bihar' still doesn't tell a health officer what to fix."

If someone notices Ward and GMM scoring higher than the chosen model:

> "Both were excluded for instability — they score 0.50 on reproducibility, below our
> 0.60 floor. We set that floor before selecting, and it's the same floor we used to
> choose k."

### 1:30 — Same score, different problems (1 minute) — the money shot

Scroll to the pair dropdown. **Pick Ernakulam (Kerala) vs Koraput (Odisha).**

> "Both score 0.65 on the composite index — identical to two decimal places. A ranking
> says treat them the same. Look at the shapes."

Point at the bars:

> "Ernakulam is +2.5 standard deviations on education and +1.2 on sanitation, but
> *negative* on non-communicable disease — a diabetes and blood-pressure problem. Koraput
> is −1.7 on education and −1.4 on sanitation, but *positive* on child health and
> nutrition — its health services are actually reaching people. Same score. Opposite
> problems. Opposite interventions."

### 2:30 — Tab 1, explore a district (1.5 minutes)

Select **Karnataka → Bangalore**.

> "Profile 2, high development. The radar shows education, energy and sanitation pushing
> outward, with one dent — NCDs."

Point at the similar-districts table:

> "Its closest neighbours include Surat in Gujarat. Similar problems don't stop at state
> borders — which is exactly what grouping by state misses."

Now the confidence box — **this is the strongest single thing in the app**:

> "This says 1.16×. It means the second-nearest profile is only 16% further away than
> its own. Bangalore is a borderline district. Compare that with Madurai at 2.39× —
> nothing would move Madurai. We report a label *and* how much to trust it."

Then switch to **Karnataka → Yadgir**.

> "Same state, profile 4 — middle development, weak schooling. Its nearest neighbours are
> Koppal, Raichur, Bagalkot — north Karnataka. One state, two completely different
> situations."

### 4:00 — Tab 3, what-if simulator (1 minute)

Load **Karnataka → Raichur** (profile 6, low development). Drag **all four policy
levers** — `clean_cooking_fuel`, `improved_sanitation`, `women_10yr_schooling`,
`institutional_births` — to the far right.

*(Verified: Raichur moves from profile 6 to profile 7. Two levers alone are not enough —
which is a useful thing to point out.)*

> "Push all four levers to the top of the national range and Raichur reclassifies from
> profile 6 to profile 7. Note that two levers alone don't move it — this isn't a
> district that's one intervention away."

Then the honesty line — examiners reward it:

> "To be clear, this is reclassification, not prediction. It says where a district with
> these values *would sit*, not what would happen if you built the stoves."

### 5:00 — Tab 2, enter a new district (30 seconds)

Fill in four or five boxes in the form and submit.

> "For a district that isn't in NFHS-5, or a new district after a boundary change: enter
> what you know, the model fills the rest from the five most similar districts — and it
> tells you exactly which values it filled in."

### 5:30 — Close on the honest limitation (30 seconds)

Back to tab 5, point at the unseen-state table.

> "We rebuilt the entire model — scaler, PCA and clustering — without one state at a
> time. Bihar came back identical, ARI 1.00. Karnataka 0.30, Assam 0.17. That's not the
> model breaking; it's the label confidence from tab 1, at state scale. Bihar's districts
> have a median margin of 1.46 — deep inside their profiles. Assam's is 1.14 — on a
> boundary, so they move as a block. The honest conclusion is that this is a continuum,
> and we tell you which labels to trust."

---

# PART 4 — Questions you will get, with short answers

| Question | Answer |
|---|---|
| "Why 8 clusters?" | "The silhouette is flat from k=3 to 10, so we didn't pick on that. We picked the largest k that still reproduces itself when you drop 20% of districts — k=8 passes at ARI 0.64, k=9 fails." |
| "Is the clustering just ranking districts again?" | "No — the agreement between our clusters and the composite index's groups is barely above chance. Several profiles sit at roughly the same average score with different problems." |
| "Why PCA?" | "124 pairs of indicators correlate above 0.6. Correlated columns double-count in every distance calculation. PCA rotates them onto uncorrelated axes; 30 of them carry 90% of the variation." |
| "Why did you hold out child nutrition?" | "To test the clusters, not to group them. If profiles built only from living conditions and services also separate districts on stunting and anaemia, the profiles are capturing something real." |
| "Isn't child health a feature though?" | "Yes — vaccination, diarrhoea and ARI are features. Only the nutrition and anaemia *outcomes* are held out." |
| "How do you validate an unsupervised model?" | "Two ways. The held-out outcome test — nine indicators the model never saw. And the unseen-state test — rebuild everything without a state and see if its districts land in the same profiles." |
| "Why did Assam fail?" | "Its districts sit near a boundary — median label margin 1.14 against Bihar's 1.46. Boundary districts flip as a block. It's a warning about which labels to trust, not a broken model." |
| "Why is DBSCAN in there if it didn't work?" | "As a control. It needs density gaps between groups. It found one big cluster and a few outliers — direct evidence that districts form a continuum, which is why we don't claim eight natural types." |
| "Your two data sources disagree — which did you trust?" | "Neither blindly. We took values and reliability flags from the factsheet mirror, names and census codes from the other, and aligned them by comparing values district by district. The code asserts every match beats its runner-up by at least ten times." |
| "What if someone disagrees with your indicator choices?" | "All of it is in one file, `src/indicator_meta.py`. Change it, re-run `run_all.py`, and you get their version of the analysis in three minutes." |
| "Real-world use?" | "Targeting. Instead of 'these 116 districts are behind', you get 'these 116 need cooking fuel and civil registration first, these 53 need maternal care and tobacco control first'." |
