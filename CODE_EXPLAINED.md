# CODE_EXPLAINED.md — every line and every concept

This walks through `notebooks/NFHS_District_Clustering.ipynb` cell by cell. The
same code lives in the `src/*.py` files (noted at the top of each section), so
whichever version you are asked about, this file covers it.

Read it with the notebook open beside you.

---

## Before anything: five ideas you need

| Idea | In one sentence |
|---|---|
| **DataFrame** | A table. Rows = districts, columns = indicators. `df["col"]` picks a column, `df.loc[3]` picks row 3. |
| **Standardise (z-score)** | Rewrite every number as "how many standard deviations above/below the average". Makes columns with different units comparable. |
| **Distance** | How far apart two districts are once every indicator is a coordinate. Clustering is entirely about distance. |
| **Fit / transform / predict** | scikit-learn's three verbs: `fit` learns from data, `transform` applies what was learned, `predict` assigns a label. |
| **Random seed** | A fixed starting number for anything random, so the code gives the same answer every time you run it. |

---

# Cell 1 — Setup

```python
def ensure(pip_name, import_name=None):
    try:
        importlib.import_module(import_name or pip_name)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip_name], check=True)
```

* `importlib.import_module("sklearn")` tries to load a package **by name given as text**.
* If the package isn't there Python raises `ImportError`, which `except ImportError:` catches.
* `subprocess.run([...])` runs a command as if you typed it in a terminal.
  `sys.executable` is the path to the Python that is running right now — using it
  guarantees the install goes into *this* Python, not some other one on the machine.
* `import_name or pip_name` — some packages install under one name and import under
  another (`pip install scikit-learn`, `import sklearn`). `or` returns the first
  truthy value, so `None or "scikit-learn"` gives `"scikit-learn"`.
* **Why this exists:** the notebook has to run in Colab, on your laptop, and on the
  examiner's machine without anyone editing it.

```python
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.width", 160, "display.max_columns", 40)
```

Cosmetic: hide library warnings, make seaborn charts consistent, stop pandas
truncating wide tables.

```python
SEED = 42
np.random.seed(SEED)
```

**The reproducibility line.** K-means starts from random centres; the bootstrap
draws random subsamples. Fixing the seed means your results and the examiner's
results are identical. 42 is arbitrary — any fixed number works.

```python
ROOT = Path("nfhs_output")
for sub in ("raw", "processed", "models", "figures"):
    (ROOT / sub).mkdir(parents=True, exist_ok=True)
```

* `Path` is the modern way to handle file paths; `ROOT / "raw"` builds
  `nfhs_output/raw` with the right slash on Windows *and* Mac.
* `parents=True` creates missing parent folders; `exist_ok=True` means "don't
  crash if it already exists".

---

# Cell 2 — Download (`src/download.py`)

```python
SOURCES = {"India.csv": "https://raw.githubusercontent.com/.../India.csv", ...}
```

A **dictionary**: `{key: value}` pairs. Here filename → URL.
`raw.githubusercontent.com` serves the file's actual bytes; the normal
`github.com/...` URL would give you an HTML web page instead.

```python
if dest.exists() and dest.stat().st_size > 0:
    continue
urllib.request.urlretrieve(url, dest)
```

* Skip the download if the file is already there and non-empty — so re-running the
  notebook doesn't re-download 10 MB.
* `urlretrieve(url, dest)` downloads to a file.
* Wrapped in `try/except` so a dead link prints a message instead of killing the run
  (the second file is optional).

**Concept — why two sources?** `factsheets.csv` keeps NFHS's reliability marks and
already-numeric values, so it supplies the **values**. `India.csv` has clean indicator
names and Census 2011 codes, so it supplies the **identity** of each indicator and the
district codes. Cell 6 aligns them.

---

# Cell 3 — Inspect both mirrors

```python
raw = pd.read_csv(ROOT / "raw" / "India.csv", low_memory=False, dtype=str)
fs_raw = pd.read_csv(ROOT / "raw" / "factsheets.csv", low_memory=False)
```

* `dtype=str` on the first file reads **everything as text**, so nothing is silently
  coerced before we look at it.
* `low_memory=False` makes pandas read in one pass instead of guessing column types
  from chunks.
* The second file needs neither: its values are already numeric.

```python
raw[col] = raw[col].astype(str).str.strip()
```

`.str.strip()` removes leading and trailing spaces. This one line matters: the source
contains both `"Balod"` and `"Balod "`, which pandas would otherwise treat as two
different districts.

```python
print(fs_raw["Flag_NFHS5"].value_counts(dropna=False).to_string())
```

`value_counts(dropna=False)` counts each distinct value **including blanks**. Printing
it here is the evidence for the whole design: 5,042 cells marked "based on 25-49
unweighted cases", 4,113 marked "not shown", the rest blank. That column is the reason
this file is the value source.

**Concept — why two sources.** One file has trustworthy *values and flags*; the other
has trustworthy *names and census codes*. Using each for its strength, and aligning
them, is the core data-engineering decision in the project.

**Concept — long vs wide.** Long: one row per (district, indicator). Wide: one row per
district, one column per indicator. Every ML library needs wide, because one row must
equal one observation.

---

# Cell 4 — The indicator metadata (`src/indicator_meta.py`)

```python
def canon(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())
```

* `re.sub(pattern, replacement, text)` = find-and-replace with a **regular
  expression** (a pattern language for text).
* `[^a-z0-9]` means "any character that is NOT a lowercase letter or digit"; the `^`
  inside brackets means *not*.
* So `canon("Male Blood sugar level  high (141-160 mg/dl) (%)")` →
  `"malebloodsugarlevelhigh141160mgdl"`, and the misspelt `"MaleBlood sugar..."`
  produces **the same string**. That is how duplicate indicator names get detected
  without guessing.
* `-> str` is a type hint: documentation for humans, not enforced by Python.

```python
INDICATORS = [
    ("Women who are literate (%)", "women_literate", "education", +1, "feature"),
    ...
]
```

A list of **tuples** — fixed-length groups of values. The five slots are
`(full name, short name, domain, direction, role)`.

* **direction** `+1` = higher is better, `-1` = higher is worse. Needed in cell 15,
  because you cannot average "literacy 80%" and "stunting 40%" while they point in
  opposite directions.
* **role**:
  * `feature` — used to build clusters (77 of them survive cleaning)
  * `outcome` — **held out**; the honesty test in cell 21 depends on these never
    being seen by any model
  * `drop` — the family-planning method mix, where no honest direction exists
    (more condoms is not "better" than more IUDs)

```python
cfg = pd.DataFrame([{...} for f, s, d, dirn, r in INDICATORS])
```

A **list comprehension** — `[expression for item in list]` builds a list in one line.
Here each tuple is unpacked into five variables and turned into a dictionary, and
the list of dictionaries becomes a DataFrame.

```python
missing = {canon(i) for i in raw["Indicator"].unique()} - set(cfg["code"])
assert not missing, f"indicators with no metadata: {missing}"
```

* `{... for ...}` with braces is a **set comprehension** — a collection with no
  duplicates and fast membership tests.
* `set_a - set_b` = items in A that are not in B.
* `assert condition, message` crashes immediately if the condition is false.
  **Why:** if NFHS added an indicator we hadn't classified, silently ignoring it
  would corrupt the analysis. Better to stop.

---

# Cell 5 — Values and reliability flags (`src/clean.py`)

```python
item = fs["Indicator"].str.extract(r"^\s*(\d+)\s*\.")[0]
fallback = fs["Indicator"].str.findall(r"(\d+)\s*\.\s*[A-Za-z]").str[-1]
fs["item"] = item.fillna(fallback).astype(float)
assert fs["item"].notna().all()
```

**Indicators are keyed by their factsheet item number, not their name.** This is the
most important line in the cleaning code, and the reason is concrete: in this file,
items 86–88 are *women's* blood sugar and 89–91 are *men's*, and **the names are
identical**. The gender survives only in the number. A few names are also garbled by
the PDF text extraction.

* `str.extract(pattern)` pulls the first capture group out of each string; `[0]` takes
  that column out of the returned frame.
* `^\s*(\d+)\s*\.` — `^` start of string, `\s*` optional spaces, `(\d+)` the number we
  want, `\.` a literal dot.
* Two rows have a section heading glued in front by the PDF extractor, so the leading
  pattern fails there. `findall(...).str[-1]` takes the **last** `NN.` in the string,
  which is the item's own number; `fillna` uses it only where needed.
* `assert` stops the run if any row still has no item number — better than quietly
  dropping data.

```python
flag_text = fs["Flag_NFHS5"].astype(str)
fs["low_reliability"] = flag_text.str.contains("25-49", na=False).astype(int)
fs["suppressed"] = flag_text.str.contains("not shown", na=False)
```

* `.astype(int)` turns True/False into 1/0 so the flag can later be aggregated with
  `max` — if any source row was flagged, the cell is flagged.
* Suppressed cells are already blank in this file: NFHS never printed them, so there is
  nothing to delete. That is the whole advantage over the other mirror, where those
  cells arrive looking like ordinary numbers.

```python
state_map = {x: (difflib.get_close_matches(canon(x), reference_states, n=1, cutoff=0.6)
                 or [canon(x)])[0] for x in fs["state"].unique()}
fs["key"] = fs["state"].map(state_map) + "|" + fs["district"].map(canon)
```

* `difflib.get_close_matches` is fuzzy text matching: `"NCT Delhi"` in one file,
  `"NCT of Delhi"` in the other; `cutoff=0.6` means at least 60% similar.
* `(... or [canon(x)])[0]` — if the match list is empty (falsy), fall back to the name
  itself, then take element 0. A compact "default value" idiom.
* The join key is `"kerala|ernakulam"` — state **and** district, because 7 district
  names repeat across states, so district alone is not unique.

```python
values5 = fs.pivot_table(index="key", columns="item", values="NFHS5", aggfunc="mean")
```

The pivot: rows become districts, columns become factsheet item numbers.

```python
st_code = pd.to_numeric(raw["ST_CEN_CD"], errors="coerce")
codes = (raw.assign(census_code=st_code * 1000 + dt_code).groupby("key")["census_code"]
            .agg(lambda x: x.dropna().iloc[0] if x.notna().any() else np.nan).to_frame())
```

* `pd.to_numeric(..., errors="coerce")` turns unparseable entries into NaN instead of
  raising — this column contains stray characters.
* The Census 2011 code is `state code × 1000 + district code`. Districts created after
  2011 have none, which is why only 632 of 704 districts carry one.
* The `lambda` takes the first non-missing value per district; `.agg` applies it
  group-wise.

---

# Cell 6 — Aligning the two mirrors by value fingerprint

This cell answers one question: *which of our 104 indicators is factsheet item 86?*
Name matching cannot answer it, so we match on the numbers themselves.

```python
shared = values5.index.intersection(reference.index)
prim, ref = values5.loc[shared], reference.loc[shared]
```

`.intersection` gives the districts present in **both** files (695 of them) — the only
rows where a comparison is meaningful.

```python
for it in prim.columns:
    diffs = ref.sub(prim[it], axis=0).abs().mean().dropna().sort_values()
```

For one factsheet item:

* `ref.sub(prim[it], axis=0)` subtracts that item's column from **every** India.csv
  indicator column, district by district (`axis=0` aligns on rows).
* `.abs().mean()` gives the average absolute difference per candidate indicator.
* `.sort_values()` puts the best candidate first.

```python
    best, runner_up = diffs.index[0], diffs.iloc[1]
    assert diffs.iloc[0] * 10 < runner_up, f"item {int(it)} is ambiguous"
```

**The assert is the point.** Both files were parsed from the same PDFs, so the correct
match agrees to about 0.001 while the next-best candidate is off by orders of
magnitude. Demanding a 10× gap means the alignment is *verified from the data*, not
assumed. In the real run, the worst item still beats its runner-up by 2481 to 0.16.

```python
def rename(mat):
    out = mat[[c for c in mat.columns if c in mapping]].rename(columns=mapping)
    return out.T.groupby(level=0).mean().T
```

* Keep only matched columns, rename item numbers to short names.
* `.T.groupby(level=0).mean().T` — transpose, group rows sharing a name, average,
  transpose back. This collapses any indicator the factsheet prints twice.

```python
wide5 = wide5.reindex(ids["key"]).reset_index(drop=True)
```

**`reindex`, not `.loc`.** Districts created after NFHS-4 have no 2015-16 row at all,
so `pivot_table` simply omits them and `.loc` would raise a KeyError. `reindex` puts
them back as blank rows, which is the honest representation.

**What to say in the viva:** "We aligned the two sources by value fingerprint rather
than by name, because the names lose the gender of the NCD indicators. The code asserts
that each match beats its runner-up by at least ten times, so the alignment is checked,
not trusted."

---

# Cell 7 — Missing data (`src/clean.py`)

```python
feature_cols = [c for c in cfg.loc[cfg["role"] == "feature", "short_name"] if c in wide5.columns]
outcome_cols = [c for c in cfg.loc[cfg["role"] == "outcome", "short_name"] if c in wide5.columns]
X_raw, Y_raw = wide5[feature_cols].copy(), wide5[outcome_cols].copy()
```

**The split that makes the project honest.** `X` is what the model may look at; `Y` is
sealed away. `.copy()` makes a real new table rather than a view onto the old one.

```python
miss_col = mat.isna().mean().sort_values(ascending=False)
drop_cols = miss_col[miss_col > max_col].index.tolist()
```

* `.isna()` → a table of True/False.
* `.mean()` on True/False treats True as 1 — so the mean **is the fraction missing**,
  per column. A neat idiom worth remembering.
* Anything missing in more than 10% of districts is dropped: nine features go, all
  small-denominator ones (diarrhoea treatment, non-breastfed infants' diet).

**Watch what this rule now catches.** With every indicator flagged consistently,
**anaemia in pregnant women** turns out to be suppressed in 19% of districts — pregnant
women are a small subgroup of a district sample — so it is dropped, leaving **9**
held-out outcomes instead of 10. Under the old name-based flag matching it slipped
through with its small-sample values intact. That is the concrete payoff of switching
sources.

```python
miss_row = mat.isna().mean(axis=1)
```

`axis=1` means "across the columns, per row" — the fraction of indicators missing for
each district. Districts above 20% are dropped (one: Jabalpur).

```python
pipe = Pipeline([("scale", StandardScaler()), ("impute", KNNImputer(n_neighbors=5))])
z = pipe.fit_transform(mat)
filled = pipe.named_steps["scale"].inverse_transform(z)
```

* A **Pipeline** chains steps so they always run in the same order, and so the whole
  chain can be saved and replayed on new data later (cell 29 depends on this).
* `fit_transform` = learn the means/SDs, then apply them.
* `KNNImputer(n_neighbors=5)`: a missing value is replaced by the average of the 5 most
  similar districts. **Standardising first is essential** — "similar" is a distance, and
  an indicator measured in thousands of rupees would otherwise overwhelm percentages.
* `inverse_transform` converts back to real units so the saved table is readable.

**Features and outcomes are imputed separately.** If outcome columns helped fill feature
columns, information would leak from the sealed set into the model's inputs and the test
in cell 21 would flatter the clusters.

---

# Cell 8 — Winsorising and de-duplication

```python
lo, hi = X.quantile(0.01), X.quantile(0.99)
X = X.clip(lower=lo, upper=hi, axis=1)
```

* `quantile(0.01)` per column = the value below which 1% of districts fall.
* `clip` pulls anything more extreme back to those bounds.
* **Why:** Chandigarh and Lakshadweep sit 15+ standard deviations out on single
  indicators. Left alone, K-means spends an entire cluster on one district — we saw
  exactly that before adding this line. 1.8% of cells are touched, and it's declared
  in the report. The unclipped matrix is kept.

```python
corr_abs = X.corr().abs()
for i, a in enumerate(corr_abs.columns):
    if a in dropped: continue
    for b in corr_abs.columns[i + 1:]:
```

* `.corr()` = correlation matrix (every column against every column); `.abs()`
  because −0.97 is just as redundant as +0.97.
* `enumerate` gives position **and** value; `columns[i+1:]` looks only at later
  columns so each pair is checked once.
* `continue` skips to the next loop step.

```python
            loser = b if order[a] <= order[b] else a
```

When two indicators correlate above 0.95, keep whichever appears earlier in the
metadata table — which is ordered so headline indicators come before their
sub-components. Here it keeps "fully vaccinated" over "polio 3 doses".
**Why drop at all:** two near-identical columns give that one concept double weight
in every distance and in PCA.

---

# Cells 9–11 — EDA (`src/eda.py`)

```python
sns.heatmap(X_missing_mask.T, cbar=False, cmap=["#f0f0f0", "#c0392b"])
```

`.T` transposes (rows↔columns) so indicators run down the side. A two-colour map
turns True/False into grey/red.

```python
abs_corr = corr.abs().where(~np.eye(len(corr), dtype=bool))
```

* `np.eye(n)` = identity matrix (1s on the diagonal); `~` flips True/False.
* `.where(mask)` keeps values where the mask is True and blanks the rest — this
  removes the diagonal, which is always 1.0 (a column correlates perfectly with
  itself) and would otherwise distort the average.
* Result: **mean |r| = 0.22, and 115 pairs above 0.6.** That is the argument for PCA.

```python
between = df.groupby("state")[col].apply(lambda s: len(s) * (s.mean() - grand) ** 2).sum()
within_share = 1 - between / total
```

* `lambda s: ...` is a one-line unnamed function; here it computes each state's
  contribution to **between-state** variance (group size × squared distance of the
  group mean from the overall mean).
* Total variance minus between-state variance = within-state variance.
* **Result: 41%** of a typical indicator's variance sits *inside* states. This is the
  number that justifies clustering districts instead of using states.

---

# Cells 12–14 — Standardise and PCA (`src/features.py`)

```python
scaler = StandardScaler().fit(X)
Z = scaler.transform(X)
```

`fit` learns each column's mean and standard deviation; `transform` applies
`(value − mean) / sd`. Keeping the fitted `scaler` object matters: in cell 28 a new
district must be scaled with the **training** means, not its own.

```python
pca = PCA(random_state=SEED).fit(Z)
cum = np.cumsum(pca.explained_variance_ratio_)
```

* **PCA in one sentence:** find the direction in which districts differ most, call it
  PC1; find the next-most-different direction at right angles to it, call it PC2; and
  so on. The new axes are uncorrelated by construction.
* `explained_variance_ratio_` = the share of total variation each component carries.
  The trailing underscore is scikit-learn's convention for "learned from data".
* `np.cumsum` = running total, so `cum[17]` is the variance captured by the first 18
  components.

```python
ks = {t: int(np.searchsorted(cum, t) + 1) for t in (0.80, 0.90, 0.95)}
```

`np.searchsorted(cum, 0.80)` finds where 0.80 would slot into the sorted running
total — i.e. how many components are needed to reach 80%. Answer here: **18 for 80%,
30 for 90%, 41 for 95%.** All three are carried forward and compared in cell 23, so
the choice isn't doing the work quietly.

```python
P80 = pca.transform(Z)[:, :ks[0.80]]
```

`[:, :18]` is NumPy slicing: **all rows, first 18 columns**. Each district is now 18
numbers instead of 77.

```python
s = pd.Series(pca.components_[i], index=X.columns)
print("high end:", ", ".join(s.sort_values(ascending=False).head(5).index))
```

`components_[i]` is the recipe for component *i* — one weight per original indicator.
Sorting shows what the axis means in words. PC1 turns out to run from
"mothers get antenatal care, iron tablets, postnatal checks" at one end to
"young population, high fertility, home births, high tobacco" at the other.

---

# Cells 15–16 — The baselines (`src/baselines.py`)

```python
for c in good.columns:
    if direction[c] < 0:
        good[c] = -good[c]
good = (good - good.min()) / (good.max() - good.min())
```

* Flip the "higher is worse" indicators so **every** column means "more is better".
* Then **min–max scaling**: subtract the minimum, divide by the range → every column
  runs 0 (worst district) to 1 (best district). Only now is averaging meaningful.
* Note this differs from z-scoring: min–max forces a fixed 0–1 range, which is what
  index-building conventionally uses.

```python
composite = good.mean(axis=1)
by_domain = good.T.groupby(good.columns.map(domain_of)).mean().T
domain_score = by_domain.mean(axis=1)
```

* `composite` = the ordinary equal-weight index: one number per district.
* The second line averages **within each domain first**, then across domains, so a
  domain with 15 indicators doesn't outvote one with 2. `.T ... .T` transposes,
  groups the (now) rows by domain, and transposes back.
* The two indices correlate at **Spearman 0.85** — a single score is a single score
  however you weight it.

```python
quantile_groups = lambda s, k: pd.qcut(s.rank(method="first"), q=k, labels=False).to_numpy()
```

* `qcut` cuts a continuous variable into **equal-sized** groups (quantiles).
* `.rank(method="first")` breaks ties so the group sizes come out exactly equal.
* `labels=False` returns 0,1,2,… instead of interval objects.
* **Critically, `k` is the same k as the clustering.** eta-squared rises
  mechanically with more groups, so giving the baseline the same number of groups is
  what makes the later comparison fair.

```python
base["state_group"] = base["state"].astype("category").cat.codes
```

`.astype("category").cat.codes` turns text labels into integers 0…36 — the same
format as cluster labels, so the same scoring functions work on it.

```python
assert not set(ids["state"]) - set(state_to_region)
```

Guard: if a state were missing from the region map it would silently become NaN and
quietly corrupt a baseline. Better to crash.

---

# Cells 17–20 — Clustering (`src/cluster.py`)

```python
def fit_kmeans(Xfit, k):
    return KMeans(n_clusters=k, n_init=50, random_state=SEED).fit(Xfit)
```

**K-means:** place k centres, assign each district to its nearest centre, move each
centre to the mean of its members, repeat until nothing moves.
`n_init=50` runs the whole thing 50 times from different random starts and keeps the
best, because the result depends on where the centres begin.

```python
def fit_ward(Xfit, k):
    return AgglomerativeClustering(n_clusters=k, linkage="ward").fit(Xfit)
```

**Ward / hierarchical:** start with every district in its own cluster and repeatedly
merge the pair whose merging increases within-cluster spread least. Produces the
dendrogram in cell 18. Downside: a merge can never be undone, which is why it is less
stable here (ARI 0.48 vs K-means' 0.68).

```python
GaussianMixture(n_components=k, covariance_type=cov, n_init=10, random_state=SEED)
```

**GMM:** assumes the data is a blend of Gaussian "blobs" and gives each district a
*probability* of belonging to each. `covariance_type="full"` lets blobs stretch and
tilt; `"diag"` is the cheaper version used on the 77 raw features, where "full" would
mean estimating 77×77 numbers per cluster from ~100 districts.

```python
def score_all(Zeval, labels):
    mask = labels >= 0
```

DBSCAN labels noise points `-1`; `labels >= 0` excludes them so a method isn't
rewarded for throwing away the hard cases. **All three metrics are computed in the
same 77-feature space** for every method, otherwise the numbers wouldn't be comparable.

| Metric | Reads | Meaning |
|---|---|---|
| `silhouette_score` | higher better, −1…1 | how much closer a district is to its own cluster than to the next nearest |
| `davies_bouldin_score` | lower better | cluster spread ÷ cluster separation |
| `calinski_harabasz_score` | higher better | between-cluster variance ÷ within-cluster variance |
| `gm.bic(Xfit)` | lower better | GMM fit, penalised for using more parameters |

```python
nn = NearestNeighbors(n_neighbors=5).fit(P90)
dists = np.sort(nn.kneighbors(P90)[0][:, -1])
line = dists[0] + (dists[-1] - dists[0]) * np.arange(len(dists)) / (len(dists) - 1)
eps = float(dists[int(np.argmax(line - dists))])
```

**Tuning DBSCAN's `eps`.** For every district, find the distance to its 5th nearest
neighbour; sort those distances. The curve stays flat then bends upward — the "knee"
is the usual choice for `eps`. The three lines find that knee by drawing a straight
line between the curve's endpoints and taking the point furthest below it
(`argmax` = position of the maximum).

**What DBSCAN tells us:** whatever `eps` we pick, it either swallows everything into
one cluster or calls a third of districts noise. Districts form one continuous cloud,
not separated blobs. That is a genuine finding, not a failure to report.

---

# Cell 19 — Choosing k

```python
for k in K_RANGE:
    basel = fit_kmeans(P90, k).labels_
    for b in range(30):
        idx = rng.choice(len(P90), int(0.8 * len(P90)), replace=False)
        sub_labels = KMeans(n_clusters=k, n_init=20, random_state=b).fit(P90[idx]).labels_
        aris.append(adjusted_rand_score(basel[idx], sub_labels))
```

* `rng.choice(n, size, replace=False)` draws a random 80% of districts **without**
  repeats.
* `P90[idx]` keeps just those rows.
* `adjusted_rand_score(a, b)` measures agreement between two groupings of the same
  items, corrected for chance: 1 = identical, 0 = no better than random. It ignores
  the fact that cluster *numbers* get shuffled between runs, which is exactly what
  you need here.
* `basel[idx]` compares like with like — the full-data labels **of the same
  districts** that were resampled.

```python
passing = sorted(stab[(stab.k.isin(size_ok)) & (stab.stability_ari >= 0.60)]["k"])
K = passing[0]
for cand in passing[1:]:
    if cand == K + 1: K = cand
    else: break
```

* `&` is element-wise AND for pandas conditions (each condition needs its own
  brackets).
* The loop walks up the passing k values and stops at the first gap — so we take the
  largest k in an **unbroken** run, rather than cherry-picking a lone k further out
  that scrapes over the line.

**Why not just take the best silhouette?** Because the silhouette curve is flat —
0.091 to 0.110 across every k from 3 to 10. Reading a winner off a 0.01 difference
would be fitting noise. So k is chosen on a question that matters: *would we get the
same groups from a slightly different sample?* Answer: **k = 8** (k = 3 to 8 clear the
0.60 floor; k = 9 fails). k = 3 and 4 have the highest silhouette and k = 3 is still
reported in the comparison table, where it scores 0.071 on the held-out test against
k = 8's 0.233.

---

# Cells 21–23 — Evaluation (`src/evaluate.py`)

```python
def eta_squared(values, labels):
    grand = v.mean()
    ss_total = ((v - grand) ** 2).sum()
    ss_between = sum(((v[l == g].mean() - grand) ** 2) * (l == g).sum() for g in np.unique(l))
    return ss_between / ss_total
```

**The most important function in the project.**

* `ss_total` — total squared distance of every district from the overall average.
* `ss_between` — for each group: how far the group's average is from the overall
  average, squared, times the group's size.
* Their ratio is **eta-squared**: the share of the outcome's variation that is
  explained by knowing which group a district is in. It is exactly the effect size of
  a one-way ANOVA.
* `v[l == g]` is boolean masking again: the outcome values of the districts in group g.

Applied to the **held-out** outcomes, this answers: *do these groups separate
districts on results the grouping never saw?*
Clustering **0.233**, composite index **0.034** at the same number of groups.

```python
def distinct_features(Xdf, labels):
    alpha = 0.01 / Xdf.shape[1]
    if stats.kruskal(*samples).pvalue < alpha: hits += 1
```

* **Kruskal–Wallis** tests whether several groups differ on a variable. It compares
  ranks rather than means, so it doesn't assume a normal distribution — the right
  choice here because several indicators are skewed.
* `*samples` unpacks a list into separate arguments (`kruskal(g0, g1, g2, ...)`).
* `alpha = 0.01 / 77` is the **Bonferroni correction**: running 77 tests at p<0.01
  would throw up false positives by chance, so the threshold is tightened by the
  number of tests.
* Result: the clusters separate **76 of 77** features; the composite index, 53.

```python
Yz = (Y - Y.mean()) / Y.std()
```

z-scores the outcomes so the ten of them are on one scale before averaging their
eta-squared values. (eta-squared is scale-free anyway; this keeps it tidy.)

```python
def stability(name, labels):
    if name in ("state", "region"):
        out.append(1.0)
```

State and region grouping don't change when you resample — they're fixed by
geography. Their 1.00 is **by construction**, and the report says so instead of
quietly letting them look like the most stable methods.

```python
    elif name in ("composite_index", "domain_index"):
        sub = good.iloc[idx]
        rescaled = (sub - sub.min()) / (sub.max() - sub.min())
        out.append(adjusted_rand_score(labels[idx], quantile_groups(rescaled.mean(axis=1), K)))
```

The index baselines are **rebuilt from scratch** on each subsample — re-scaled and
re-cut — so they face the same test as the clusterings rather than an easier one.

---

# Cell 24 — Picking the final model

```python
ETA_TOLERANCE = 0.02
MIN_STABILITY_ARI = 0.60

stable = candidates[candidates.stability_ari >= MIN_STABILITY_ARI]
excluded = sorted(set(candidates.grouping) - set(stable.grouping))
tied = stable[stable.mean_outcome_eta2 >= stable.mean_outcome_eta2.max() - ETA_TOLERANCE]
final_name = tied.sort_values(["stability_ari", "silhouette"], ascending=False).iloc[0]["grouping"]
```

The rule, in order: **stability floor → held-out eta-squared (ties within 0.02) →
stability → silhouette**, with any grouping having a cluster under 15 districts excluded.

**Why the floor exists, and why it is not cherry-picking.** Cell 19 already refused to
accept a *k* whose clustering falls apart when 20% of districts are removed. Applying the
same 0.60 standard to the *algorithm* is consistent, not convenient. It matters here:
Ward on raw features (eta² 0.272) and GMM on raw features (0.265) score marginally
higher than the winner but reproduce at only 0.50. When an earlier version of this rule
let GMM win, the unseen-state test in cell 30 collapsed — because "refit without one
state" and "refit without 20% of districts" are the same question asked twice.

`excluded` is printed rather than hidden, so the reader sees what the floor removed.

**Winner: K-means on 30 PCA components, k = 8** — eta² 0.233, stability 0.64.

Two honest points to make when presenting this cell:

* against the like-for-like baseline (composite index, same number of groups),
  clustering wins roughly 7 to 1 — 0.233 against 0.034;
* **state grouping scores highest (0.512)** — with 34 groups against 8, which flatters
  eta-squared, and states share diet, policy and health systems. Reported, not hidden.

---

# Cell 25 — Same score, different problems

```python
cand = ids.assign(score=composite, cluster=final_labels).sort_values("score").reset_index()
for i in range(len(cand) - 1):
    for j in range(i + 1, min(i + 12, len(cand))):
        if cand.loc[j, "score"] - cand.loc[i, "score"] > 0.02: break
```

* `.assign(...)` adds columns and returns a new table (chainable).
* Sorting by score first means near-equal districts are **adjacent**, so we only need
  to look at the next 11 rows — and can `break` as soon as the score gap exceeds
  0.02. That turns a 705×705 comparison into a quick scan.
* `reset_index()` keeps the original row numbers in a column called `index`, so we
  can still look up each district's domain profile.

```python
gap = float(np.abs(district_domains.loc[a] - district_domains.loc[b]).max())
```

For each pair, the biggest difference on any single domain — used to rank the pairs
so the three most striking ones get plotted. **5,377 pairs** qualify.

---

# Cells 26–27 — Profiles and names (`src/profile.py`)

```python
z_oriented = (X - X.mean()) / X.std()
for c in z_oriented.columns:
    z_oriented[c] *= direction[c]
district_domains = z_oriented.T.groupby(z_oriented.columns.map(domain_of)).mean().T
```

Standardise, then multiply by direction so **positive always means better**. Average
within each domain → each district gets 11 numbers. This one convention is what makes
every heatmap and radar chart in the project readable: a dent is always a problem.

```python
def name_cluster(profile):
    p = profile.drop(["ncd"], errors="ignore")
    level = p.mean()
    word = "High development" if level > 0.35 else "Low development" if level < -0.35 else "Middle development"
    gaps = p[p < -0.40].sort_values().head(2)
```

Names are generated **mechanically from the numbers**: a level word plus the domains
more than 0.4 SD below average. NCDs are excluded from the naming logic (not the
charts) because high blood sugar and blood pressure *rise* with development — "high
development, weak NCDs" would be nonsense.

```python
counts = pd.Series(list(names.values())).value_counts()
for dupe in counts[counts > 1].index:
```

If two clusters land on the same generic label, append whichever domain most
distinguishes each from the average cluster, so every profile has a unique name in
the app.

---

# Cell 28 — How safe is each district's label?

```python
centroids_all = pd.DataFrame(final_scores).groupby(final_labels).mean().to_numpy()
dist_to_centres = np.linalg.norm(final_scores[:, None, :] - centroids_all[None, :, :], axis=2)
sorted_d = np.sort(dist_to_centres, axis=1)
margin = sorted_d[:, 1] / sorted_d[:, 0]
```

* `groupby(final_labels).mean()` gives each profile's **centre** — the average district
  of that profile.
* `final_scores[:, None, :] - centroids_all[None, :, :]` is **broadcasting**: `None`
  inserts a length-1 axis, so a (704 × 30) array minus an (8 × 30) array produces
  (704 × 8 × 30) — every district against every centre in one step, no loop.
* `np.linalg.norm(..., axis=2)` collapses the last axis into a distance, leaving
  (704 × 8).
* `np.sort(..., axis=1)` sorts each district's eight distances: column 0 is its own
  centre, column 1 the runner-up.

**margin = runner-up distance ÷ own distance.**

* 2.0 → the second-best profile is twice as far. The district is deep inside its
  profile; nothing short of rebuilding the model would move it.
* 1.0 → exactly between two profiles. It got its label by a hair.

National median **1.24×**; **240 of 704** districts fall below 1.15×.

This one number is the honest answer to "is this district really profile 4?", and it
**predicts the unseen-state results in cell 30 before you run them**: Bihar's districts
have a median margin of 1.46 and reproduce at ARI 1.00; Assam's median is 1.14 and they
reproduce at 0.17.

---

# Cells 29–30 — New districts and the unseen-state test (`src/predict.py`)

```python
def assign_cluster(values: dict):
    row = pd.Series({c: np.nan for c in X_raw.columns}, dtype=float)
    for k_, v in values.items():
        if k_ in row.index and v is not None and not pd.isna(v):
            row[k_] = float(v)
    imputed = [c for c in X.columns if pd.isna(row[c])]
```

Start from an all-missing row, fill in whatever the user gave, and record what stayed
missing — the app shows that list, so the user can judge how much of the answer is their
data and how much is the model filling gaps.

```python
    filled = impute_pipe.named_steps["impute"].transform(
        impute_pipe.named_steps["scale"].transform(row.to_frame().T))
```

**`transform`, not `fit_transform`.** This is the whole point of saving the fitted
objects: the new district is scaled by the *training* data's means, and its gaps are
filled using the *training* districts as neighbours. Re-fitting on one row would be
meaningless, and the same district would get different answers on different days.

```python
check = assign_cluster(X.iloc[0].to_dict())
assert check["cluster"] == final_labels[0]
```

A self-consistency test: feed a training district back in through the "new district"
path and it must come out in the cluster it already belongs to. If this assert fires,
some step of the pipeline is not being replayed properly.

### The unseen-state test

```python
held = (ids["state"] == state).to_numpy()
sc = StandardScaler().fit(X[~held])
pc = PCA(random_state=SEED).fit(Ztr)
```

`~held` = NOT held out. Scaler, PCA **and** clustering are refitted from scratch on the
remaining districts, so the held-out state contributed nothing at all — not even to the
column means.

```python
    algo = SPECS[final_name]["algo"]
    if algo == "gmm":
        model = GaussianMixture(...).fit(Ptr); assigned = model.predict(Pte)
    elif algo == "ward":
        train_labels = AgglomerativeClustering(...).fit(Ptr).labels_
        cents = np.vstack([Ptr[train_labels == c].mean(axis=0) for c in np.unique(train_labels)])
        assigned = np.argmin(((Pte[:, None, :] - cents[None, :, :]) ** 2).sum(axis=2), axis=1)
    else:
        assigned = KMeans(...).fit(Ptr).predict(Pte)
```

**Refit the same algorithm the final model uses.** An earlier version always refit
K-means; when the selection rule picked a GMM, the test was silently comparing two
different methods and the scores collapsed. Ward has no `predict()` at all — it only
labels the data it was fitted on — so held-out districts are assigned to the nearest
centroid of the clusters learned on the training districts.

```python
    pair_agreement = np.mean([(truth[i] == truth[j]) == (assigned[i] == assigned[j])
                              for i in range(len(truth)) for j in range(i + 1, len(truth))])
```

A double comprehension over all district pairs of that state: were they together in the
full model, and are they still together now? Easier to explain in a viva than ARI —
*"100% of Bihar's district pairs kept the same relationship."*

**Results: Bihar ARI 1.000, Karnataka 0.303, Assam 0.169** — and the cell joins each
state's median label margin beside them (1.46, 1.16, 1.14), so the explanation sits in
the same table as the result. Interior districts reproduce; boundary districts move as a
block, because they resemble each other and all sit on the same side of the same line.

---

# Cell 31 — Saving

```python
joblib.dump({"scaler": scaler, "pca": pca, "model": final_model, "imputer": impute_pipe,
             "columns": list(X.columns), "n_components": n_components, "k": K,
             "cluster_names": names}, ROOT / "models" / "pipeline.joblib")
```

`joblib.dump` serialises fitted Python objects to disk (better than `pickle` for
NumPy-heavy objects). Everything needed to assign a new district is saved **together**
— including the column order, because feeding columns in a different order would
silently produce garbage. This file is what the Streamlit app loads.

---

## Glossary — the terms an examiner will ask about

| Term | Answer in one or two lines |
|---|---|
| **Unsupervised learning** | Learning structure with no correct answers to learn from. Nobody has labelled districts with their "true" development type. |
| **Standardisation (z-score)** | `(x − mean) / sd`. Puts every indicator on one scale so distances aren't dominated by big-numbered columns. |
| **Min–max scaling** | `(x − min) / (max − min)`. Forces a 0–1 range; used for the composite index, not for clustering. |
| **PCA** | A rotation onto new uncorrelated axes ordered by how much variation each explains. Keeps signal, drops redundancy. |
| **Explained variance ratio** | The share of total variation carried by each component. 18 components carry 80% here. |
| **Loading** | The weight an original indicator has in a component — how you read what a component means. |
| **K-means** | Repeatedly assign points to the nearest of k centres and move the centres to their members' mean. |
| **Ward linkage** | Bottom-up merging, always choosing the merge that adds least within-cluster spread. Gives a dendrogram. |
| **GMM** | Models the data as a blend of Gaussian blobs; gives probabilities, not hard labels. |
| **DBSCAN** | Density-based clustering that can label points as noise. Needs density gaps, which this data lacks. |
| **Silhouette** | −1 to 1; how much closer a point is to its own cluster than to the next. ~0.10 here = a continuum, not islands. |
| **Davies–Bouldin** | Spread ÷ separation, lower is better. |
| **Calinski–Harabasz** | Between-cluster ÷ within-cluster variance, higher is better. |
| **BIC** | Model fit penalised for parameter count; used to compare GMMs. |
| **ARI (Adjusted Rand Index)** | Agreement between two groupings, corrected for chance. Used for stability and cross-method agreement. |
| **eta-squared** | Share of a variable's variance explained by group membership; the ANOVA effect size. |
| **Kruskal–Wallis** | Rank-based test for "do these groups differ on this variable", with no normality assumption. |
| **Bonferroni correction** | Divide the significance threshold by the number of tests, to stop false positives piling up. |
| **Bootstrap** | Re-run the analysis on random subsamples to see how much the answer wobbles. |
| **KNN imputation** | Fill a gap with the average of the k most similar rows. |
| **Winsorising** | Clip extreme values to a percentile instead of deleting them. |
| **Held-out set** | Data deliberately kept away from the model so it can be used as an independent check. |
| **Leakage** | When information from the held-out set sneaks into training and makes results look better than they are. Avoided here by imputing features and outcomes separately. |
| **Value fingerprint alignment** | Matching two data sources by comparing their numbers rather than their labels, used here because the indicator names are unreliable. |
| **Label margin** | Distance to the second-nearest cluster centre ÷ distance to the nearest. 1.0 = on a boundary, 2.0 = deep inside its profile. |
| **Stability floor** | A minimum reproducibility (bootstrap ARI ≥ 0.60) a model must clear before it is eligible to be chosen at all. |
| **Broadcasting** | NumPy's rule for combining arrays of different shapes by inserting length-1 axes — it computes all district-to-centre distances without a loop. |
