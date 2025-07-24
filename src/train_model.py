import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import config

def run_training():
    """Trains and saves the recidivism model."""
    print("Starting model training...")
    
    # Load processed data
    df = pd.read_csv(config.PROCESSED_DATA_PATH)
    
    # Define features (X) and target (y)
    X = df.drop(columns=[config.TARGET])
    y = df[config.TARGET]

    #     ONE-HOT ENCODING 
    # Convert categorical string columns into numerical format.
    # The machine learning model can only process numbers.
    print("Applying one-hot encoding to features...")
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    print("Features after encoding:")
    print(list(X_encoded.columns)) 
    #     END OF FIX 

    # Split the *encoded* data
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train a model 
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate and print accuracy
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    print(f"Model trained. Test Accuracy: {accuracy:.4f}")

    # Save the trained model
    joblib.dump(model, config.MODEL_PATH)
    print(f"Model saved to {config.MODEL_PATH}")

if __name__ == "__main__":
    run_training()