import os
import sys

from src.data_preprocessing import create_data_dirs, load_data, clean_data
from src.feature_engineering import engineer_features
from src.model_training import prepare_data, train_logistic_regression, train_random_forest
from src.evaluation import evaluate_model


def main():
    # 1. Ensure directories exist and load raw data
    create_data_dirs()
    df = load_data("data/raw/Telco-Customer-Churn.csv")

    # 2. Clean and engineer features
    df = clean_data(df)
    df = engineer_features(df)

    # 3. Prepare train/test data
    X_train, X_test, y_train, y_test = prepare_data(df)

    # 4. Train models (these now return fitted estimators)
    log_model = train_logistic_regression(X_train, y_train)
    rf_model  = train_random_forest(X_train, y_train)

    # 5. Evaluate models using src.evaluation
    log_results = evaluate_model(log_model, X_test, y_test)
    rf_results  = evaluate_model(rf_model,  X_test, y_test)

    print("Logistic Regression Results:")
    print("  Accuracy :", log_results["accuracy"])
    print("  Precision:", log_results["precision"])
    print("  Recall   :", log_results["recall"])
    print("  F1       :", log_results["f1"])
    print("\nClassification report (Logistic Regression):")
    print(log_results["classification_report"])

    print("\nRandom Forest Results:")
    print("  Accuracy :", rf_results["accuracy"])
    print("  Precision:", rf_results["precision"])
    print("  Recall   :", rf_results["recall"])
    print("  F1       :", rf_results["f1"])
    print("\nClassification report (Random Forest):")
    print(rf_results["classification_report"])


if __name__ == "__main__":
    main()
