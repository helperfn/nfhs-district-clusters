# Phase 2 - cleaning report

- Values and reliability flags: **jvargh7 factsheet mirror** (already numeric, flags for every indicator). India.csv supplies Census 2011 codes and the indicator names, and the two are aligned by value fingerprint (see `reports/source_alignment.csv`).
- Districts kept: **704**
- Clustering features: **77** (from 88 candidates)
- Held-out outcomes: **9**
- Missing cells imputed with KNNImputer(k=5) on standardised values, features and outcomes imputed separately to protect the held-out test.

- Indicators dropped (> 10% missing): `adequate_diet_nonbreastfed` (91.1%), `complementary_feeding_6_8m` (90.9%), `diarrhoea_ors` (69.5%), `diarrhoea_care_sought` (69.5%), `diarrhoea_zinc` (69.5%), `home_birth_checkup_24h` (59.7%), `exclusive_breastfeeding` (37.0%), `ari_fever_care_sought` (31.5%), `caesarean_private` (21.4%)
- Districts dropped (> 20% of indicators missing): Madhya Pradesh / Jabalpur
- Indicators dropped (> 10% missing): `pregnant_women_anaemic` (18.8%)
- Near-duplicate dropped: `polio3` (|r| = 0.966 with `fully_vaccinated_card_or_recall`, kept)
- Near-duplicate dropped: `adequate_diet_total` (|r| = 0.961 with `adequate_diet_breastfed`, kept)
- Winsorised 1,002 feature cells (1.8%) at the 1st/99th percentile.