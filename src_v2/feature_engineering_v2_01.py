from __future__ import annotations

from typing import Tuple
import numpy as np
import pandas as pd

from .config import FEATURE_COLUMNS, LABEL_COL, ID_COL


_CATEGORICALS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


def _add_safe_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # charges_ratio: TotalCharges per month of tenure (robust to tenure=0)
    if "TotalCharges" in out.columns and "tenure" in out.columns:
        denom = out["tenure"].fillna(0).astype(float) + 1.0
        out["charges_ratio"] = out["TotalCharges"].fillna(0).astype(float) / denom
    else:
        out["charges_ratio"] = 0.0

    # HighSpender: top quartile of MonthlyCharges
    if "MonthlyCharges" in out.columns:
        q75 = out["MonthlyCharges"].quantile(0.75)
        out["HighSpender"] = (out["MonthlyCharges"] >= q75).astype(int)
    else:
        out["HighSpender"] = 0

    # HighChurnRisk: heuristic risk flag based ONLY on non-label features
    # (kept simple and leakage-safe)
    def _flag(row) -> int:
        tenure = float(row.get("tenure", 0) or 0)
        contract = str(row.get("Contract", "")).lower()
        pay = str(row.get("PaymentMethod", "")).lower()
        internet = str(row.get("InternetService", "")).lower()
        monthly = float(row.get("MonthlyCharges", 0) or 0)
        risk = 0
        if "month-to-month" in contract:
            risk += 1
        if "electronic check" in pay:
            risk += 1
        if tenure < 12:
            risk += 1
        if "fiber optic" in internet:
            risk += 1
        if monthly >= df["MonthlyCharges"].quantile(0.75) if "MonthlyCharges" in df.columns else False:
            risk += 1
        return 1 if risk >= 3 else 0

    out["HighChurnRisk"] = out.apply(_flag, axis=1)

    return out


def build_processed_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the exact feature schema used by the saved Logistic Regression model
    (FEATURE_COLUMNS) + ID_COL + LABEL_COL.

    Output columns:
      - customerID
      - Churn_Yes
      - 26 model features in FEATURE_COLUMNS (order preserved)
    """
    out = df.copy()

    # Label: Churn_Yes (if raw 'Churn' exists)
    if LABEL_COL not in out.columns and "Churn" in out.columns:
        out[LABEL_COL] = (out["Churn"].astype(str).str.strip().str.lower() == "yes").astype(int)

    # Engineered features
    out = _add_safe_engineered_features(out)

    # One-hot encode the specific categoricals
    present_cats = [c for c in _CATEGORICALS if c in out.columns]
    dummies = pd.get_dummies(out[present_cats], drop_first=True)

    # Numeric base
    base_cols = [c for c in ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"] if c in out.columns]
    base = out[base_cols].copy()

    # Combine all potential features
    feat = pd.concat([base, out[["charges_ratio", "HighSpender", "HighChurnRisk"]], dummies], axis=1)

    # Align to expected schema and order; fill missing with 0
    feat = feat.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    # Build final frame
    final_cols = []
    if ID_COL in out.columns:
        final_cols.append(out[ID_COL].astype(str))
    else:
        # fall back to row index as ID
        final_cols.append(pd.Series(np.arange(len(out)), name=ID_COL).astype(str))

    if LABEL_COL in out.columns:
        final_cols.append(out[LABEL_COL].astype(int))
    else:
        # if label missing, keep for scoring-only workflows
        final_cols.append(pd.Series([pd.NA]*len(out), name=LABEL_COL))

    final = pd.concat(final_cols + [feat], axis=1)
    final.columns = [ID_COL, LABEL_COL] + FEATURE_COLUMNS
    return final
