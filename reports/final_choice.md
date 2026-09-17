# Phase 7 - final model choice

**Chosen model: `kmeans_pca` with k = 8.**

Selection order, fixed before the numbers were looked at: a stability floor (mean bootstrap ARI >= 0.6, the same floor used to choose k), then held-out outcome eta-squared, then stability, then silhouette, then interpretability. Groupings with any cluster under 15 districts were excluded. Excluded for instability: gmm_pca, gmm_raw, ward_pca, ward_raw.

## The numbers that decided it

| grouping        |   n_groups |   silhouette |   davies_bouldin |   mean_outcome_eta2 |   distinct_features |   stability_ari |
|:----------------|-----------:|-------------:|-----------------:|--------------------:|--------------------:|----------------:|
| state           |         34 |        0.113 |            1.905 |               0.512 |                  77 |           1     |
| ward_raw        |          8 |        0.078 |            2.345 |               0.272 |                  76 |           0.495 |
| gmm_pca         |          8 |        0.082 |            2.398 |               0.271 |                  76 |           0.469 |
| gmm_raw         |          8 |        0.088 |            2.326 |               0.265 |                  76 |           0.5   |
| kmeans_raw      |          8 |        0.099 |            2.385 |               0.248 |                  76 |           0.619 |
| kmeans_pca      |          8 |        0.097 |            2.376 |               0.233 |                  76 |           0.64  |
| kmeans_pca95    |          8 |        0.1   |            2.368 |               0.232 |                  76 |           0.618 |
| kmeans_pca80    |          8 |        0.101 |            2.353 |               0.218 |                  76 |           0.619 |
| ward_pca        |          8 |        0.089 |            2.406 |               0.206 |                  76 |           0.535 |
| region          |          6 |        0.049 |            2.893 |               0.206 |                  75 |           1     |
| domain_index    |          8 |       -0.004 |            7.246 |               0.086 |                  64 |           0.166 |
| kmeans_pca_k3   |          3 |        0.11  |            2.31  |               0.071 |                  70 |           0.935 |
| composite_index |          8 |       -0.009 |            8.788 |               0.034 |                  53 |           0.904 |
| dbscan_pca      |          2 |        0.275 |            0.941 |               0.008 |                  14 |           0.719 |

## Where the traditional baselines do as well or better

- The strongest baseline is **state** (mean outcome eta-squared 0.512 vs 0.233 for the chosen clustering).
- State grouping explains 0.512 of held-out outcome variance - with 37 groups, far more than the 7 the clustering uses. Any grouping with more groups has an arithmetic advantage on eta-squared, so this comparison flatters state grouping and is reported as such.
- The composite index cut into 8 quantiles explains 0.034. That is the like-for-like comparison (same number of groups), and it is the one the project's claim stands or falls on.

## Sensitivity to the PCA cut-off

| grouping      |   n_groups |   silhouette |   mean_outcome_eta2 |   stability_ari |
|:--------------|-----------:|-------------:|--------------------:|----------------:|
| kmeans_pca    |          8 |        0.097 |               0.233 |           0.64  |
| kmeans_pca95  |          8 |        0.1   |               0.232 |           0.618 |
| kmeans_pca80  |          8 |        0.101 |               0.218 |           0.619 |
| kmeans_pca_k3 |          3 |        0.11  |               0.071 |           0.935 |

## Same score, different problems

5632 pairs of districts sit within 0.02 of each other on the composite index yet fall in different clusters. The three with the largest domain gap are plotted in `reports/figures/15_same_score_different_problems.png`.

| district_a                        | district_b                        |   score_a |   score_b |   cluster_a |   cluster_b |   biggest_domain_gap |
|:----------------------------------|:----------------------------------|----------:|----------:|------------:|------------:|---------------------:|
| Nicobar (Andaman Nicobar Islands) | Amritsar (Punjab)                 |     0.576 |     0.578 |           3 |           2 |                0.922 |
| Ernakulam (Kerala)                | Koraput (Odisha)                  |     0.646 |     0.647 |           3 |           4 |                0.909 |
| Dantewada (Chhattisgarh)          | Alappuzha (Kerala)                |     0.614 |     0.616 |           4 |           3 |                0.882 |
| Pathanamthitta (Kerala)           | Rayagada (Odisha)                 |     0.599 |     0.6   |           3 |           4 |                0.85  |
| Anand (Gujarat)                   | Nicobar (Andaman Nicobar Islands) |     0.576 |     0.576 |           7 |           3 |                0.849 |
| Aizawl (Mizoram)                  | South Salmara Mancachar (Assam)   |     0.563 |     0.563 |           7 |           6 |                0.846 |
| Prakasam (Andhra Pradesh)         | Hardoi (Uttar Pradesh)            |     0.527 |     0.53  |           0 |           6 |                0.845 |
| Kaushambi (Uttar Pradesh)         | Hyderabad (Telangana)             |     0.564 |     0.567 |           6 |           0 |                0.84  |
| Hardoi (Uttar Pradesh)            | Imphal West (Manipur)             |     0.53  |     0.532 |           6 |           2 |                0.835 |
| South Salmara Mancachar (Assam)   | Suryapet (Telangana)              |     0.563 |     0.564 |           6 |           0 |                0.834 |