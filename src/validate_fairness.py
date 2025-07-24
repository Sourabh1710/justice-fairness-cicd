import pandas as pd
import joblib
from fairlearn.metrics import MetricFrame, demographic_parity_difference
from sklearn.model_selection import train_test_split
import config
import sys

def run_validation():
    """Loads a trained model and evaluates its fairness on the COMPAS dataset."""
    print("Starting fairness validation...")

    # Load processed data
    df = pd.read_csv(config.PROCESSED_DATA_PATH)
    
    # Isolate the original sensitive attribute column for Fairlearn's use
    sensitive_features_original = df[config.SENSITIVE_ATTRIBUTE]
    
    # Prepare the feature set X and target y
    X = df.drop(columns=[config.TARGET])
    y = df[config.TARGET]
    
    # One-hot encode the data to match the model's training format
    print("Applying one-hot encoding for validation...")
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    # Split data to get the same test set as in training
    _, X_test_encoded, _, y_test = train_test_split(
        X_encoded, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Align the original sensitive features column with the test set samples
    sensitive_features_test = sensitive_features_original.loc[y_test.index]
    
    # Load the trained model
    try:
        model = joblib.load(config.MODEL_PATH)
    except FileNotFoundError:
        print("Error: Model file not found. Run training first.")
        sys.exit(1)
        
    # Get model predictions using the correctly formatted, ENCODED test data
    y_pred = model.predict(X_test_encoded)

    #     Fairness Analysis using Fairlearn 
    metrics = {'selection_rate': lambda y_true, y_pred: y_pred.mean()}
    
    grouped_on_race = MetricFrame(metrics=metrics, y_true=y_test, y_pred=y_pred, sensitive_features=sensitive_features_test)

    # This returns a pandas Series
    dp_series = grouped_on_race.difference(method='between_groups')
    dp_diff_value = dp_series.item()
    
    # Now, use the extracted float value for printing and comparison
    print(f"Demographic Parity Difference (Recidivism Prediction Rate): {dp_diff_value:.4f}")

    #     The Fairness Gate 
    # Use the extracted float value in the condition
    if abs(dp_diff_value) > config.DEMOGRAPHIC_PARITY_THRESHOLD:
        print(f"VALIDATION FAILED: Bias ({abs(dp_diff_value):.4f}) is above the threshold ({config.DEMOGRAPHIC_PARITY_THRESHOLD}).")
        sys.exit(1) # Fail the pipeline
    else:
        print("VALIDATION PASSED: Model is within fairness thresholds.")
        sys.exit(0) # Pass the pipeline

if __name__ == "__main__":
    run_validation()