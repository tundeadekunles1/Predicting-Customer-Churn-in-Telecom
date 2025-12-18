import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from typing import Tuple, Any, Dict, Optional
from sklearn.preprocessing import StandardScaler

def prepare_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Prepare the data for training and testing."""
    df = df.copy()  # Avoid modifying the original DataFrame
    y = df["Churn_Yes"]
    X = df.drop(columns=["Churn_Yes", "customerID", "tenure_group"], errors='ignore')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    return X_train, X_test, y_train, y_test
    return X_train, X_test, y_train, y_test

def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """Train and return a Logistic Regression model."""
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train: np.ndarray, y_train: np.ndarray, n_estimators: int = 300) -> RandomForestClassifier:
    """Train and return a Random Forest model."""
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_decision_tree(X_train: np.ndarray, y_train: np.ndarray) -> DecisionTreeClassifier:
    """Train and return a Decision Tree model."""
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train: np.ndarray, y_train: np.ndarray) -> XGBClassifier:
    """Train and return an XGBoost model."""
    model = XGBClassifier(
        random_state=42,
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        use_label_encoder=False
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Optional[float]]:
    """Return common evaluation metrics for a fitted classifier."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba) if y_proba is not None else None
    }

    return metrics

def load_model(filepath: str) -> Any:
    """Load and return a persisted model."""
    return joblib.load(filepath)

def save_model(model: Any, filepath: str) -> None:
    """Persist model to filepath using joblib."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)

def plot_confusion_matrices(models: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray) -> None:
    """Plot confusion matrices for given models."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for ax, (name, model) in zip(axes, models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'{name} Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

    plt.tight_layout()
    plt.show()

def plot_roc_curves(models: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray) -> None:
    """Plot ROC curves for given models."""
    plt.figure(figsize=(8, 6))

    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

def plot_feature_importance(models: Dict[str, Any], X: pd.DataFrame) -> None:
    """Plot feature importance for tree-based models."""
    for name, model in models.items():
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            features = X.columns if hasattr(X, 'columns') else [f'Feature {i}' for i in range(X.shape[1])]
            plt.figure(figsize=(10, 6))
            plt.title(f'{name} Feature Importances')
            plt.bar(range(len(importances)), importances[indices], align='center')
            plt.xticks(range(len(importances)), [features[i] for i in indices], rotation=90)
            plt.tight_layout()
            plt.show()

