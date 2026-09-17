# PCA components in plain English

*77 standardised indicators, 704 districts.*

| variance target | components kept |
|---|---|
| 80% | 18 |
| 90% | 30 |
| 95% | 41 |

The project uses the **90% rule -> 30 components** for the main pipeline; 80% and 95% are tested in Phase 7 to show the choice is not doing the work.


## PC1 - Maternal health
*variance explained:* 25.3%  
*polarity:* higher score = worse outcomes  
*dominant domains:* Maternal health (8), Population & household (2), Nutrition (1), Women's empowerment & family planning (1)

| indicator | loading |
|---|---|
| pop_below_15 | +0.19 |
| ifa_180_days | -0.18 |
| ifa_100_days | -0.18 |
| anc_4plus_visits | -0.17 |
| pnc_mother_2days | -0.17 |
| pnc_newborn_2days | -0.17 |
| death_registered | -0.17 |
| caesarean_total | -0.17 |

Positive side: pop_below_15  
Negative side: ifa_180_days, ifa_100_days, anc_4plus_visits, pnc_mother_2days, pnc_newborn_2days

## PC2 - Child health & immunisation + Maternal health
*variance explained:* 11.1%  
*polarity:* higher score = better outcomes  
*dominant domains:* Child health & immunisation (5), Maternal health (4), Women's empowerment & family planning (2), NCDs (blood sugar, BP, screening) (1)

| indicator | loading |
|---|---|
| hepb3_penta3 | +0.20 |
| measles_dose1 | +0.19 |
| oop_delivery_public_rs | -0.19 |
| fully_vaccinated_card_or_recall | +0.19 |
| tetanus_protected_birth | +0.19 |
| dpt3_penta3 | +0.19 |
| men_bp_elevated_or_medicated | -0.18 |
| bcg | +0.17 |

Positive side: hepb3_penta3, measles_dose1, fully_vaccinated_card_or_recall, tetanus_protected_birth, dpt3_penta3  
Negative side: oop_delivery_public_rs, men_bp_elevated_or_medicated

## PC3 - NCDs (blood sugar, BP, screening)
*variance explained:* 7.0%  
*polarity:* higher score = worse outcomes  
*dominant domains:* NCDs (blood sugar, BP, screening) (10), Women's empowerment & family planning (2)

| indicator | loading |
|---|---|
| women_blood_sugar_high | +0.26 |
| men_blood_sugar_high_or_medicated | +0.26 |
| women_blood_sugar_high_or_medicated | +0.24 |
| men_bp_mildly_elevated | -0.24 |
| men_blood_sugar_high | +0.23 |
| women_bp_mildly_elevated | -0.23 |
| men_blood_sugar_very_high | +0.22 |
| married_before_18 | +0.20 |

Positive side: women_blood_sugar_high, men_blood_sugar_high_or_medicated, women_blood_sugar_high_or_medicated, men_blood_sugar_high, men_blood_sugar_very_high  
Negative side: men_bp_mildly_elevated, women_bp_mildly_elevated, men_bp_moderate_severe, men_bp_elevated_or_medicated

## PC4 - Tobacco & alcohol
*variance explained:* 5.2%  
*polarity:* higher score = worse outcomes  
*dominant domains:* Tobacco & alcohol (4), Women's empowerment & family planning (2), NCDs (blood sugar, BP, screening) (2), Nutrition (1), Water & sanitation (WASH) (1), Population & household (1), Energy (1)

| indicator | loading |
|---|---|
| women_tobacco_use | +0.29 |
| men_alcohol_use | +0.26 |
| women_alcohol_use | +0.23 |
| adequate_diet_breastfed | +0.22 |
| men_tobacco_use | +0.21 |
| teen_pregnancy | +0.21 |
| men_blood_sugar_high | +0.21 |
| improved_water | -0.21 |

Positive side: women_tobacco_use, men_alcohol_use, women_alcohol_use, adequate_diet_breastfed, men_tobacco_use  
Negative side: improved_water, clean_cooking_fuel, birth_order_3plus

## PC5 - NCDs (blood sugar, BP, screening)
*variance explained:* 4.8%  
*polarity:* higher score = better outcomes  
*dominant domains:* NCDs (blood sugar, BP, screening) (6), Education (2), Nutrition (2), Population & household (1), Women's empowerment & family planning (1)

| indicator | loading |
|---|---|
| female_ever_attended_school | +0.28 |
| men_bp_moderate_severe | -0.27 |
| women_literate | +0.26 |
| women_bp_moderate_severe | -0.26 |
| men_bp_elevated_or_medicated | -0.23 |
| men_bp_mildly_elevated | -0.21 |
| women_bmi_below_normal | -0.21 |
| women_bp_elevated_or_medicated | -0.20 |

Positive side: female_ever_attended_school, women_literate, birth_registered, early_breastfeeding_1h  
Negative side: men_bp_moderate_severe, women_bp_moderate_severe, men_bp_elevated_or_medicated, men_bp_mildly_elevated, women_bmi_below_normal
