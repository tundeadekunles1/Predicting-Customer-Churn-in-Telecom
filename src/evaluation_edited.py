# from sklearn.metrics import (
#     accuracy_score,
#     precision_score,
#     recall_score,
#     f1_score,
#     confusion_matrix,
#     classification_report,
# )

# def evaluate_model(model, X_test, y_test):
#     y_pred = model.predict(X_test)
#     acc = accuracy_score(y_test, y_pred)
#     prec = precision_score(y_test, y_pred)
#     rec = recall_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)
#     cm = confusion_matrix(y_test, y_pred)
#     report = classification_report(y_test, y_pred)
#     return {
#         "accuracy": acc,
#         "precision": prec,
#         "recall": rec,
#         "f1": f1,
#         "confusion_matrix": cm,
#         "classification_report": report,
#     }

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

def evaluate_model(model, X_test, y_test, threshold=0.5):
    # Probabilities
    y_prob = model.predict_proba(X_test)[:, 1]

    # Apply custom threshold
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred),
    }

def recall_at_k(y_true, y_prob, k=0.2):
    cutoff = int(len(y_prob) * k)
    top_k_indices = np.argsort(y_prob)[::-1][:cutoff]
    return y_true.iloc[top_k_indices].sum() / y_true.sum()


def precision_at_k(y_true, y_prob, k=0.2):
    cutoff = int(len(y_prob) * k)
    top_k_indices = np.argsort(y_prob)[::-1][:cutoff]
    return y_true.iloc[top_k_indices].sum() / cutoff

def f1_at_k(y_true, y_prob, k=0.2):
    prec = precision_at_k(y_true, y_prob, k)
    rec = recall_at_k(y_true, y_prob, k)
    if prec + rec == 0:
        return 0.0
    return 2 * (prec * rec) / (prec + rec) 
 
def evaluate_model_at_k(model, X_test, y_test, k=0.2):
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "recall_at_k": recall_at_k(y_test, y_prob, k),
        "precision_at_k": precision_at_k(y_test, y_prob, k),
        "f1_at_k": f1_at_k(y_test, y_prob, k),
    }   
    