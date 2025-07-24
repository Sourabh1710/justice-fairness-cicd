import os

# Data paths
RAW_DATA_PATH = "compas-scores-two-years.csv" 
PROCESSED_DATA_PATH = os.path.join("data", "processed", "compas_cleaned.csv")

# Model artifact path
MODEL_PATH = "recidivism_model.joblib"

# Target variable: 1 if the person re-offended within two years
TARGET = "two_year_recid"

# Sensitive attribute for fairness analysis
SENSITIVE_ATTRIBUTE = "race"

RACE_FILTERS = ['African-American', 'Caucasian']


FEATURES = [
    'sex',
    'age',
    'age_cat',
    'juv_fel_count',
    'juv_misd_count',
    'juv_other_count',
    'priors_count',
    'c_charge_degree',
    'race' 
]

# Fairness Threshold
DEMOGRAPHIC_PARITY_THRESHOLD = 0.10