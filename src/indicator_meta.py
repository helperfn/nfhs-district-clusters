"""
Indicator metadata: the judgement layer of the project.

Every NFHS indicator gets four attributes. These drive everything downstream,
so they are declared once, here, and written out to config/indicators.csv:

  domain     which thematic block the indicator belongs to. Used for the
             domain-weighted baseline, the domain heatmaps and the radar charts.
  direction  +1 = higher is better (literacy), -1 = higher is worse (stunting).
             Needed because a composite index must add things that point the
             same way. Clustering itself does NOT use direction - standardising
             already removes scale - but the sign makes the profiles readable.
  role       feature  = used to build clusters
             outcome  = deliberately HELD OUT, never seen by any grouping, so
                        it can be used as an independent fairness test (Phase 7)
             drop     = not used (no defensible direction, or redundant)

Choice of outcomes (child stunting / wasting / underweight / overweight and
anaemia in children and women): these are the nutrition results policy actually
cares about. If clusters built only from living conditions and service use
still separate districts on these untouched outcomes, the clusters capture
something real rather than an artefact of the input columns.

Family-planning *method mix* (pill, IUD, condom, sterilisation...) is dropped:
using more condoms is not "better" or "worse" than using more IUDs, so no
honest direction can be assigned. The totals (any method / any modern method)
are kept.
"""
from __future__ import annotations

# (full_name, short_name, domain, direction, role)
INDICATORS = [
    # ---------------------------------------------- population & household ---
    ("Female population age 6 years and above who ever attended school (%)", "female_ever_attended_school", "education", +1, "feature"),
    ("Population below age 15 years (%)", "pop_below_15", "population_household", -1, "feature"),
    ("Sex ratio of the total population (females per 1,000 males)", "sex_ratio_total", "population_household", +1, "feature"),
    ("Sex ratio at birth for children born in the last five years (females per 1,000 males)", "sex_ratio_at_birth", "population_household", +1, "feature"),
    ("Children under age 5 years whose birth was registered with the civil authority (%)", "birth_registered", "population_household", +1, "feature"),
    ("Deaths in the last 3 years registered with the civil authority (%)", "death_registered", "population_household", +1, "feature"),
    ("Population living in households with electricity (%)", "electricity", "energy", +1, "feature"),
    ("Population living in households with an improved drinkingwater source (%)", "improved_water", "wash", +1, "feature"),
    ("Population living in households that use an improved sanitation facility (%)", "improved_sanitation", "wash", +1, "feature"),
    ("Households using clean fuel for cooking (%)", "clean_cooking_fuel", "energy", +1, "feature"),
    ("Households using iodized salt (%)", "iodized_salt", "nutrition", +1, "feature"),
    ("Households with any usual member covered under a health insurance/financing scheme (%)", "health_insurance", "population_household", +1, "feature"),
    ("Children age 5 years who attended preprimary school during the school year 2019-20 (%)", "preprimary_school", "education", +1, "feature"),

    # ---------------------------------------------- characteristics of women ---
    ("Women who are literate (%)", "women_literate", "education", +1, "feature"),
    ("Women with 10 or more years of schooling (%)", "women_10yr_schooling", "education", +1, "feature"),

    # --------------------------------------------------- marriage & fertility ---
    ("Women age 2024 years married before age 18 years (%)", "married_before_18", "women_empowerment", -1, "feature"),
    ("Births in the 5 years preceding the survey that are third or higher order (%)", "birth_order_3plus", "women_empowerment", -1, "feature"),
    ("Women age 15-19 years who were already mothers or pregnant at the time of the survey (%)", "teen_pregnancy", "women_empowerment", -1, "feature"),
    ("Women age 15-24 years who use hygienic methods of protection during their menstrual period (%)", "hygienic_menstrual_protection", "women_empowerment", +1, "feature"),

    # --------------------------------------- family planning (current use etc) ---
    ("Any method (%)", "fp_any_method", "women_empowerment", +1, "feature"),
    ("Any modern method (%)", "fp_any_modern_method", "women_empowerment", +1, "feature"),
    ("Female sterilization (%)", "fp_female_sterilization", "women_empowerment", -1, "drop"),
    ("Male sterilization (%)", "fp_male_sterilization", "women_empowerment", +1, "drop"),
    ("IUD/PPIUD (%)", "fp_iud", "women_empowerment", +1, "drop"),
    ("Injectables (%)", "fp_injectables", "women_empowerment", +1, "drop"),
    ("Pill (%)", "fp_pill", "women_empowerment", +1, "drop"),
    ("Condom (%)", "fp_condom", "women_empowerment", +1, "drop"),
    ("Total unmet need (%)", "fp_unmet_need_total", "women_empowerment", -1, "feature"),
    ("Unmet need for spacing (%)", "fp_unmet_need_spacing", "women_empowerment", -1, "feature"),
    ("Current users ever told about side effects of current method (%)", "fp_told_side_effects", "women_empowerment", +1, "feature"),
    ("Health worker ever talked to female non-users about family planning (%)", "fp_worker_talked_to_nonusers", "women_empowerment", +1, "feature"),

    # ------------------------------------------------------- maternal health ---
    ("Mothers who had an antenatal checkup in the first trimester (%)", "anc_first_trimester", "maternal_health", +1, "feature"),
    ("Mothers who had at least 4 antenatal care visits (%)", "anc_4plus_visits", "maternal_health", +1, "feature"),
    ("Mothers whose last birth was protected against neonatal tetanus (%)", "tetanus_protected_birth", "maternal_health", +1, "feature"),
    ("Mothers who consumed iron folic acid for 100 days or more when they were pregnant (%)", "ifa_100_days", "maternal_health", +1, "feature"),
    ("Mothers who consumed iron folic acid for 180 days or more when they were pregnant (%)", "ifa_180_days", "maternal_health", +1, "feature"),
    ("Registered pregnancies for which the mother received a Mother and Child Protection (MCP) card (%)", "mcp_card_received", "maternal_health", +1, "feature"),
    ("Mothers who received postnatal care from a doctor/nurse/LHV/ANM/midwife/other health personnel within 2 days of delivery (%)", "pnc_mother_2days", "maternal_health", +1, "feature"),
    ("Children who received postnatal care from a doctor/nurse/LHV/ANM/midwife/other health personnel within 2 days of delivery (%)", "pnc_newborn_2days", "maternal_health", +1, "feature"),
    ("Children born at home who were taken to a health facility for a checkup within 24 hours of birth (%)", "home_birth_checkup_24h", "maternal_health", +1, "feature"),
    ("Average out-of-pocket expenditure per delivery in a public health facility (Rs)", "oop_delivery_public_rs", "maternal_health", -1, "feature"),

    # ---------------------------------------------------------- delivery care ---
    ("Institutional births (%)", "institutional_births", "maternal_health", +1, "feature"),
    ("Institutional births in public facility (%)", "institutional_births_public", "maternal_health", +1, "feature"),
    ("Home births that were conducted by skilled health personnel (%)", "home_births_skilled", "maternal_health", +1, "feature"),
    ("Births attended by skilled health personnel (%)", "skilled_birth_attendance", "maternal_health", +1, "feature"),
    # Caesarean section: very high district rates signal over-medicalisation
    # (mostly private sector) rather than better care, hence direction -1.
    ("Births delivered by caesarean section (%)", "caesarean_total", "maternal_health", -1, "feature"),
    ("Births in a private health facility that were delivered by caesarean section (%)", "caesarean_private", "maternal_health", -1, "feature"),
    ("Births in a public health facility that were delivered by caesarean section (%)", "caesarean_public", "maternal_health", -1, "feature"),

    # --------------------------------------- child vaccination & vitamin A -----
    ("Children age 12-23 months fully vaccinated based on information from either vaccination card or mothers recall (%)", "fully_vaccinated_card_or_recall", "child_health_immunisation", +1, "feature"),
    ("Children age 12-23 months fully vaccinated based on information from vaccination card only (%)", "fully_vaccinated_card_only", "child_health_immunisation", +1, "feature"),
    ("Children age 12-23 months who have received 3 doses of penta or DPT vaccine (%)", "dpt3_penta3", "child_health_immunisation", +1, "feature"),
    ("Children age 12-23 months who have received 3 doses of penta or hepatitis B vaccine (%)", "hepb3_penta3", "child_health_immunisation", +1, "feature"),
    ("Children age 12-23 months who have received 3 doses of polio vaccine (%)", "polio3", "child_health_immunisation", +1, "feature"),
    ("Children age 12-23 months who have received 3 doses of rotavirus vaccine (%)", "rotavirus3", "child_health_immunisation", +1, "feature"),
    ("Children age 12-23 months who have received BCG (%)", "bcg", "child_health_immunisation", +1, "feature"),
    ("Children age 12-23 months who have received the first dose of measlescontaining vaccine (MCV) (%)", "measles_dose1", "child_health_immunisation", +1, "feature"),
    ("Children age 24-35 months who have received a second dose of measlescontaining vaccine (MCV) (%)", "measles_dose2", "child_health_immunisation", +1, "feature"),
    ("Children age 12-23 months who received most of their vaccinations in a public health facility (%)", "vaccinated_public_facility", "child_health_immunisation", +1, "feature"),
    ("Children age 12-23 months who received most of their vaccinations in a private health facility (%)", "vaccinated_private_facility", "child_health_immunisation", -1, "feature"),
    ("Children age 9-35 months who received a vitamin A dose in the last 6 months (%)", "vitamin_a_dose", "child_health_immunisation", +1, "feature"),

    # ----------------------------------------- treatment of childhood diseases --
    ("Children with diarrhoea in the 2 weeks preceding the survey taken to a health facility or health provider (%)", "diarrhoea_care_sought", "child_health_immunisation", +1, "feature"),
    ("Children with diarrhoea in the 2 weeks preceding the survey who received oral rehydration salts (ORS) (%)", "diarrhoea_ors", "child_health_immunisation", +1, "feature"),
    ("Children with diarrhoea in the 2 weeks preceding the survey who received zinc (%)", "diarrhoea_zinc", "child_health_immunisation", +1, "feature"),
    ("Children with fever or symptoms of ARI in the 2 weeks preceding the survey taken to a health facility or health provider (%)", "ari_fever_care_sought", "child_health_immunisation", +1, "feature"),
    ("Prevalence of diarrhoea in the 2 weeks preceding the survey (%)", "diarrhoea_prevalence", "child_health_immunisation", -1, "feature"),
    ("Prevalence of symptoms of acute respiratory infection (ARI) in the 2 weeks preceding the survey (%)", "ari_prevalence", "child_health_immunisation", -1, "feature"),

    # ------------------------------------------------- child feeding practices --
    ("Children under age 3 years breastfed within one hour of birth (%)", "early_breastfeeding_1h", "nutrition", +1, "feature"),
    ("Children under age 6 months exclusively breastfed (%)", "exclusive_breastfeeding", "nutrition", +1, "feature"),
    ("Children age 6-8 months receiving solid or semisolid food and breastmilk (%)", "complementary_feeding_6_8m", "nutrition", +1, "feature"),
    ("Breastfeeding children age 6-23 months receiving an adequate diet (%)", "adequate_diet_breastfed", "nutrition", +1, "feature"),
    ("Nonbreastfeeding children age 6-23 months receiving an adequate diet (%)", "adequate_diet_nonbreastfed", "nutrition", +1, "feature"),
    ("Total children age 6-23 months receiving an adequate diet (%)", "adequate_diet_total", "nutrition", +1, "feature"),

    # -------------------------------- HELD-OUT OUTCOMES: child undernutrition ---
    ("Children under 5 years who are stunted (height for age) (%)", "child_stunted", "nutrition", -1, "outcome"),
    ("Children under 5 years who are wasted (weight for height) (%)", "child_wasted", "nutrition", -1, "outcome"),
    ("Children under 5 years who are severely wasted (weight for height) (%)", "child_severely_wasted", "nutrition", -1, "outcome"),
    ("Children under 5 years who are underweight (weight for age) (%)", "child_underweight", "nutrition", -1, "outcome"),
    ("Children under 5 years who are overweight (weight fo rheight)20 (%)", "child_overweight", "nutrition", -1, "outcome"),

    # ---------------------------------------------- nutritional status of women -
    ("Women whose Body Mass Index (BMI) is below normal (BMI <18)", "women_bmi_below_normal", "nutrition", -1, "feature"),
    ("Women who are overweight or obese", "women_overweight_obese", "nutrition", -1, "feature"),
    ("Women who have high risk waisttohip ratio", "women_high_waist_hip_ratio", "nutrition", -1, "feature"),

    # ---------------------------------------------- HELD-OUT OUTCOMES: anaemia --
    ("Children age 6-59 months who are anaemic", "children_anaemic", "anaemia", -1, "outcome"),
    ("All women age 15-19 years who are anaemic (%)", "women_15_19_anaemic", "anaemia", -1, "outcome"),
    ("All women age 15-49 years who are anaemic (%)", "women_15_49_anaemic", "anaemia", -1, "outcome"),
    ("Nonpregnant women age 15-49 years who are anaemic", "nonpregnant_women_anaemic", "anaemia", -1, "outcome"),
    ("Pregnant women age 15-49 years who are anaemic", "pregnant_women_anaemic", "anaemia", -1, "outcome"),

    # ------------------------------------------------------------------ NCDs ---
    ("Female Blood sugar level  high (141-160 mg/dl) (%)", "women_blood_sugar_high", "ncd", -1, "feature"),
    ("Female Blood sugar level  very high (>160 mg/dl) (%)", "women_blood_sugar_very_high", "ncd", -1, "feature"),
    ("Female Blood sugar level  high or very high (>140 mg/dl) or taking medicine to control blood sugar level (%)", "women_blood_sugar_high_or_medicated", "ncd", -1, "feature"),
    ("Male Blood sugar level  high (141-160 mg/dl) (%)", "men_blood_sugar_high", "ncd", -1, "feature"),
    ("Male Blood sugar level  very high (>160 mg/dl) (%)", "men_blood_sugar_very_high", "ncd", -1, "feature"),
    ("Male Blood sugar level  high or very high (>140 mg/dl) or taking medicine to control blood sugar level (%)", "men_blood_sugar_high_or_medicated", "ncd", -1, "feature"),
    ("Female Mildly elevated blood pressure (Systolic 140-159 mm of Hg and/or Diastolic 90-99 mm of Hg) (%)", "women_bp_mildly_elevated", "ncd", -1, "feature"),
    ("Female Moderately or severely elevated blood pressure (%)", "women_bp_moderate_severe", "ncd", -1, "feature"),
    ("Female Elevated blood pressure or taking medicine to control blood pressure (%)", "women_bp_elevated_or_medicated", "ncd", -1, "feature"),
    ("Male Mildly elevated blood pressure (Systolic 140-159 mm of Hg and/or Diastolic 90-99 mm of Hg) (%)", "men_bp_mildly_elevated", "ncd", -1, "feature"),
    ("Male Moderately or severely elevated blood pressure (%)", "men_bp_moderate_severe", "ncd", -1, "feature"),
    ("Male Elevated blood pressure or taking medicine to control blood pressure (%)", "men_bp_elevated_or_medicated", "ncd", -1, "feature"),
    ("Ever undergone a breast examination for breast cancer (%)", "breast_cancer_exam", "ncd", +1, "feature"),
    ("Ever undergone a screening test for cervical cancer (%)", "cervical_cancer_screening", "ncd", +1, "feature"),
    ("Ever undergone an oral cavity examination for oral cancer (%)", "oral_cancer_exam", "ncd", +1, "feature"),

    # ------------------------------------------------------ tobacco & alcohol ---
    ("Men age 15 years and above who use any kind of tobacco (%)", "men_tobacco_use", "tobacco_alcohol", -1, "feature"),
    ("Men age 15 years and above who consume alcohol (%)", "men_alcohol_use", "tobacco_alcohol", -1, "feature"),
    ("Women age 15 years and above who use any kind of tobacco (%)", "women_tobacco_use", "tobacco_alcohol", -1, "feature"),
    ("Women age 15 years and above who consume alcohol (%)", "women_alcohol_use", "tobacco_alcohol", -1, "feature"),
]

# Human-readable domain labels, used in figures and in the Streamlit app.
DOMAIN_LABELS = {
    "population_household": "Population & household",
    "education": "Education",
    "wash": "Water & sanitation (WASH)",
    "energy": "Energy",
    "maternal_health": "Maternal health",
    "child_health_immunisation": "Child health & immunisation",
    "nutrition": "Nutrition",
    "anaemia": "Anaemia",
    "ncd": "NCDs (blood sugar, BP, screening)",
    "women_empowerment": "Women's empowerment & family planning",
    "tobacco_alcohol": "Tobacco & alcohol",
}
