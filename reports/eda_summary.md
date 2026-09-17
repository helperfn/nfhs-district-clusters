# Phase 3 - EDA summary

- **704 districts x 77 features** after cleaning.
- Mean absolute correlation between features is **0.22**, and **124 feature pairs** correlate above |r| = 0.6. The indicators are far from independent, which is exactly the situation PCA is for: a handful of components can carry most of the information without the double-counting.
- **42%** of the variance in a typical indicator sits *within* states rather than between them. Districts inside the same state are not alike, so 'group by state' throws away most of the signal - this is the motivation for clustering at district level.
- The missing-value heatmap shows gaps are concentrated in small-denominator indicators (diarrhoea treatment, non-breastfed infant diet), where NFHS suppresses estimates based on fewer than 25 unweighted cases. Those columns were dropped rather than imputed.

## Most correlated feature pairs

| indicator A                         | indicator B                       |      |r| |
|:------------------------------------|:----------------------------------|---------:|
| women_blood_sugar_high_or_medicated | women_blood_sugar_very_high       | 0.946601 |
| caesarean_public                    | caesarean_total                   | 0.944233 |
| institutional_births                | skilled_birth_attendance          | 0.943698 |
| pnc_mother_2days                    | pnc_newborn_2days                 | 0.943236 |
| men_bp_elevated_or_medicated        | men_bp_mildly_elevated            | 0.938806 |
| hepb3_penta3                        | dpt3_penta3                       | 0.934732 |
| men_blood_sugar_very_high           | men_blood_sugar_high_or_medicated | 0.925644 |
| women_literate                      | female_ever_attended_school       | 0.92114  |
| dpt3_penta3                         | measles_dose1                     | 0.910031 |
| ifa_180_days                        | ifa_100_days                      | 0.909472 |

## Indicators with the most within-state variation

| indicator                   |   within_state_share |
|:----------------------------|---------------------:|
| sex_ratio_at_birth          |             0.891381 |
| ari_prevalence              |             0.790568 |
| vaccinated_public_facility  |             0.736729 |
| bcg                         |             0.720801 |
| diarrhoea_prevalence        |             0.701266 |
| vaccinated_private_facility |             0.659037 |
| fully_vaccinated_card_only  |             0.65142  |
| measles_dose1               |             0.634374 |
| iodized_salt                |             0.631625 |
| dpt3_penta3                 |             0.622658 |

## Top / bottom districts

### improved_sanitation

**Top 10**

| state      | district       |   improved_sanitation |
|:-----------|:---------------|----------------------:|
| Kerala     | Ernakulam      |                98.997 |
| Kerala     | Kannur         |                98.997 |
| Kerala     | Kozhikode      |                98.997 |
| Kerala     | Palakkad       |                98.997 |
| Kerala     | Malappuram     |                98.997 |
| Kerala     | Thrissur       |                98.997 |
| Kerala     | Alappuzha      |                98.997 |
| Puducherry | Mahe           |                98.997 |
| Kerala     | Pathanamthitta |                98.9   |
| Kerala     | Kasaragod      |                98.9   |

**Bottom 10**

| state        | district   |   improved_sanitation |
|:-------------|:-----------|----------------------:|
| Karnataka    | Yadgir     |                37.4   |
| Bihar        | Saran      |                37.2   |
| Chhattisgarh | Bijapur    |                36.521 |
| Bihar        | Purnia     |                36.521 |
| West Bengal  | Puruliya   |                36.521 |
| Karnataka    | Gulbarga   |                36.521 |
| Bihar        | Araria     |                36.521 |
| Bihar        | Madhepura  |                36.521 |
| Gujarat      | Dahod      |                36.521 |
| Chhattisgarh | Sukma      |                36.521 |

### women_10yr_schooling

**Top 10**

| state      | district           |   women_10yr_schooling |
|:-----------|:-------------------|-----------------------:|
| Kerala     | Thrissur           |                 79.188 |
| Kerala     | Kottayam           |                 79.188 |
| Kerala     | Kollam             |                 79.188 |
| Kerala     | Kozhikode          |                 79.188 |
| Kerala     | Ernakulam          |                 79.188 |
| Puducherry | Mahe               |                 79.188 |
| Kerala     | Thiruvananthapuram |                 79.188 |
| Kerala     | Pathanamthitta     |                 79.188 |
| Kerala     | Alappuzha          |                 78.8   |
| Tamil Nadu | Kanniyakumari      |                 77.1   |

**Bottom 10**

| state         | district        |   women_10yr_schooling |
|:--------------|:----------------|-----------------------:|
| Uttar Pradesh | Shrawasti       |                 15.9   |
| Chhattisgarh  | Sukma           |                 15.9   |
| Odisha        | Malkangiri      |                 15.803 |
| Bihar         | Kishanganj      |                 15.803 |
| Odisha        | Nabarangapur    |                 15.803 |
| Gujarat       | Devbhumi Dwarka |                 15.803 |
| Jharkhand     | Pakur           |                 15.803 |
| Tripura       | Dhalai          |                 15.803 |
| Haryana       | Mewat           |                 15.803 |
| Uttar Pradesh | Bahraich        |                 15.803 |

### clean_cooking_fuel

**Top 10**

| state     | district           |   clean_cooking_fuel |
|:----------|:-------------------|---------------------:|
| Telangana | Medchal-Malkajgiri |                 99   |
| Telangana | Hyderabad          |                 99   |
| NCT Delhi | North              |                 99   |
| NCT Delhi | West               |                 99   |
| NCT Delhi | Shahdara           |                 99   |
| NCT Delhi | South              |                 99   |
| NCT Delhi | South East         |                 99   |
| NCT Delhi | East               |                 99   |
| NCT Delhi | North East         |                 99   |
| NCT Delhi | Central            |                 98.8 |

**Bottom 10**

| state          | district               |   clean_cooking_fuel |
|:---------------|:-----------------------|---------------------:|
| Chhattisgarh   | Sukma                  |               14.4   |
| Jharkhand      | Latehar                |               14.2   |
| Nagaland       | Mon                    |               13.424 |
| Nagaland       | Longleng               |               13.424 |
| Meghalaya      | West Khasi Hills       |               13.424 |
| Chhattisgarh   | Kodagaon               |               13.424 |
| Chhattisgarh   | Balrampur              |               13.424 |
| Meghalaya      | South West Khasi Hills |               13.424 |
| Madhya Pradesh | Dindori                |               13.424 |
| Bihar          | Madhepura              |               13.424 |