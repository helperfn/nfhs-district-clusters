# Phase 9 - assigning districts the model has not seen

## Unseen-state test

The scaler, the PCA rotation and the clustering were all refitted from scratch with one state removed; that state's districts were then assigned with `assign_cluster`. ARI compares those assignments with the profiles the same districts received in the full-data model.

| state     |   n_districts |   ari_vs_full_model |   pair_agreement |
|:----------|--------------:|--------------------:|-----------------:|
| Karnataka |            30 |               0.303 |            0.671 |
| Bihar     |            38 |               1     |            1     |
| Assam     |            33 |               0.169 |            0.593 |

Mean ARI across the three held-out states: **0.491**, mean pair agreement **75.5%**.

## NFHS-4 -> NFHS-5 profile movement

572 districts had enough NFHS-4 indicators to pass through the pipeline; **53%** of them sit in a different profile in 2019-21 than they would have in 2015-16. NFHS-4 lacks several NFHS-5 indicators, which are imputed, so treat this as direction of travel rather than a measurement.

|   cluster_2015 |   0 |   1 |   2 |   3 |   4 |   5 |   6 |   7 |
|---------------:|----:|----:|----:|----:|----:|----:|----:|----:|
|              0 |  12 |   0 |   1 |   0 |   0 |   0 |   0 |   0 |
|              1 |   0 |  34 |   2 |   1 |   3 |   2 |   0 |   6 |
|              2 |   0 |   0 |  23 |   6 |   0 |   2 |   0 |   1 |
|              3 |   0 |   0 |   3 |  32 |   0 |   0 |   0 |   0 |
|              4 |   0 |   0 |   2 |   0 |  10 |  26 |   1 |   1 |
|              5 |   0 |   0 |  13 |   0 |   0 |  19 |   0 |   1 |
|              6 |   1 |   2 |   5 |   0 | 106 |  15 | 108 |  24 |
|              7 |   0 |   0 |  32 |  12 |   4 |  33 |   0 |  29 |
