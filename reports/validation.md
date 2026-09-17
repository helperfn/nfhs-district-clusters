# Phase 7b - validation

## 1. Permutation test

Each of the 77 feature columns was shuffled independently 20 times - which keeps every indicator's own distribution intact and destroys only the relationships between them - and the complete pipeline (standardise -> PCA -> K-means) was re-run on each shuffle.

| | real data | shuffled (mean) | shuffled (best of 20) | p |
|---|---|---|---|---|
| Silhouette | **0.097** | 0.009 | 0.013 | 0.048 |
| Held-out outcome eta-squared | **0.233** | 0.012 | 0.026 | 0.048 |

The second row is the stronger result: the outcome columns were never shuffled, so a grouping built on scrambled features has no way to predict them.

## 2. Silhouette against the baselines

| grouping        |   n_groups |   silhouette |   mean_outcome_eta2 |
|:----------------|-----------:|-------------:|--------------------:|
| kmeans_pca      |          8 |        0.097 |               0.233 |
| kmeans_raw      |          8 |        0.099 |               0.248 |
| state           |         34 |        0.113 |               0.512 |
| region          |          6 |        0.049 |               0.206 |
| domain_index    |          8 |       -0.004 |               0.086 |
| composite_index |          8 |       -0.009 |               0.034 |

## 3. Soft boundaries

- **240 of 704** districts are borderline (label margin below 1.15).
- **74** districts have a negative silhouette - they sit closer to another profile's centre than to their own.
- Per-district silhouette, GMM membership probability and label margin all agree; `data/processed/district_confidence.csv` carries all three for every district.

| state | median label margin | median GMM probability | borderline | unseen-state ARI |
|---|---|---|---|---|
| Bihar | 1.46 | 0.99 | 3/38 | 1.000 |
| Karnataka | 1.16 | 0.92 | 12/30 | 0.303 |
| Assam | 1.14 | 0.83 | 18/33 | 0.169 |

Read the last two columns together: the states that fail to reproduce are the ones whose districts sit on boundaries.
