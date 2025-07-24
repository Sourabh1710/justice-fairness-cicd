import pandas as pd
import config
import os

def run_preprocessing():
    """
    Loads the raw COMPAS data and applies the same filters used in the
    ProPublica analysis to create a clean, processed dataset.
    """
    print("    Starting Data Preprocessing    ")

    #  Load Raw Data
    try:
        df = pd.read_csv(config.RAW_DATA_PATH)
        print(f"Successfully loaded raw data from: {config.RAW_DATA_PATH}")
        print(f"Initial raw data shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: Raw data file not found at '{config.RAW_DATA_PATH}'.")
        print("Please download 'compas-scores-two-years.csv' and place it in the project's root directory.")
        return 

    #  Apply ProPublica's Filters
    print("Applying ProPublica's filtering criteria...")

    df_filtered = df[
        (df['days_b_screening_arrest'] <= 30) &
        (df['days_b_screening_arrest'] >= -30) &
        (df['is_recid'] != -1) &
        (df['c_charge_degree'] != 'O') &
        (df['score_text'] != 'N/A')
    ]
    print(f"Shape after initial filters: {df_filtered.shape}")

    # Further filter to focus on the primary comparison groups
    df_filtered = df_filtered[df_filtered['race'].isin(config.RACE_FILTERS)]
    print(f"Shape after filtering for race ('African-American', 'Caucasian'): {df_filtered.shape}")

    #  Select Relevant Columns
    columns_to_keep = config.FEATURES + [config.TARGET]
    df_processed = df_filtered[columns_to_keep].copy()
    print(f"Selected final features and target. Final data shape: {df_processed.shape}")

    #  Save the Processed Data
    os.makedirs(os.path.dirname(config.PROCESSED_DATA_PATH), exist_ok=True)
    
    # Save the cleaned dataframe to a new CSV file.
    df_processed.to_csv(config.PROCESSED_DATA_PATH, index=False)
    print(f"Successfully saved processed data to: {config.PROCESSED_DATA_PATH}")
    print("    Data Preprocessing Complete    ")


if __name__ == "__main__":
    run_preprocessing()