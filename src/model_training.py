import os
import sys
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple


# Load your processed data
project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

processed_path = os.path.join(project_root, "data", "processed", "telco_churn_processed.csv")
df = pd.read_csv(processed_path)

# Prepare data

def prepare_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Prepare the data for training and testing."""
    df = df.copy()  # Avoid modifying the original DataFrame
    
    # Ensure 'Churn_Yes' exists as the target variable
    if "Churn_Yes" in df.columns:
        y = df["Churn_Yes"]
    else:
        raise KeyError("The 'Churn_Yes' column does not exist in the DataFrame.")

    # Drop unnecessary columns
    X = df.drop(columns=["Churn_Yes", "customerID", "tenure_group"], errors='ignore')

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Initialize the scaler
    scaler = StandardScaler()
    
    # Fit and transform the training data, then transform the test data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrame to retain feature names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    # Return the scaled data
    return X_train_scaled, X_test_scaled, y_train, y_test


X_train, X_test, y_train, y_test = prepare_data(df)

# Train models
models = {
    'Logistic Regression': train_logistic_regression(max_iter=2000)(X_train, y_train),
    'Random Forest': train_random_forest(X_train, y_train),
    'Decision Tree': train_decision_tree(X_train, y_train),
    'XGBoost': train_xgboost(X_train, y_train)
}

# Save models for later use
models_dir = os.path.join(project_root, "data", "models")
os.makedirs(models_dir, exist_ok=True)

for name, model in models.items():
    save_model(model, os.path.join(models_dir, f"{name.lower().replace(' ', '_')}.pkl"))

print("Models trained and saved successfully.")
