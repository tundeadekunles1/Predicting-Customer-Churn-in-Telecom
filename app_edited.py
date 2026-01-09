# app.py
# Telco Customer Churn — Retention Targeting (Top 10% / 20% by churn probability)
#

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

try:
    import joblib  # type: ignore
except Exception:
    joblib = None  # type: ignore

# Optional: SHAP for key-driver explanations (especially useful for Random Forest)
try:
    import shap  # type: ignore
except Exception:
    shap = None  # type: ignore


# ----------------------------
# App config
# ----------------------------
st.set_page_config(
    page_title="Telco Churn — Retention Targeting",
    page_icon="📉",
    layout="wide",
)


# ----------------------------
# Helpers
# ----------------------------
def project_root() -> Path:
    return Path(__file__).resolve().parent


def first_existing_path(candidates: List[Path]) -> Optional[Path]:
    for p in candidates:
        if p.exists():
            return p
    return None


def detect_customer_id_col(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "CustomerID",
        "customerID",
        "customer_id",
        "CustomerId",
        "customerId",
        "cust_id",
        "CustID",
        "subscriber_id",
    ]
    lowered = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lowered:
            return lowered[c.lower()]
    return None


def detect_label_col(df: pd.DataFrame) -> Optional[str]:
    """
    Project churn label column is explicitly: 'Churn_Yes'
    Returns 'Churn_Yes' if present (case-insensitive), else None.
    """
    lowered = {c.lower(): c for c in df.columns}
    return lowered.get("churn_yes", None)


def normalize_label(y: pd.Series) -> pd.Series:
    if y.dtype == bool:
        return y.astype(int)
    if pd.api.types.is_numeric_dtype(y):
        return y.astype(int)

    y_str = y.astype(str).str.strip().str.lower()
    mapping = {"yes": 1, "y": 1, "true": 1, "1": 1, "no": 0, "n": 0, "false": 0, "0": 0}
    mapped = y_str.map(mapping)
    if mapped.isna().any():
        return pd.to_numeric(y_str, errors="coerce").fillna(0).astype(int)
    return mapped.astype(int)


@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    if joblib is None:
        raise RuntimeError("joblib is not available. Please `pip install joblib`.")
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found: {p}")
    return joblib.load(p)


@st.cache_data(show_spinner=False)
def load_default_dataset() -> Tuple[Optional[pd.DataFrame], Optional[Path]]:
    root = project_root()
    p = root / "data" / "processed" / "telco_churn_processed.csv"
    if not p.exists():
        return None, p
    try:
        return pd.read_csv(p), p
    except Exception:
        return None, p


def extract_preprocessor_and_estimator(model):
    preprocessor = None
    estimator = model
    if hasattr(model, "named_steps"):
        steps = model.named_steps
        if "preprocessor" in steps:
            preprocessor = steps["preprocessor"]
        try:
            estimator = list(steps.values())[-1]
        except Exception:
            estimator = model
    return preprocessor, estimator


def expected_input_columns(model) -> Optional[List[str]]:
    """
    Return the exact feature columns the model was fitted with (when available).
    This prevents 'feature names unseen at fit time' errors.

    Priority:
    1) model.feature_names_in_
    2) model.named_steps["preprocessor"].feature_names_in_ (if pipeline)
    """
    try:
        if hasattr(model, "feature_names_in_"):
            cols = list(getattr(model, "feature_names_in_"))
            return cols if len(cols) > 0 else None
    except Exception:
        pass

    try:
        if hasattr(model, "named_steps") and "preprocessor" in model.named_steps:
            prep = model.named_steps["preprocessor"]
            if hasattr(prep, "feature_names_in_"):
                cols = list(getattr(prep, "feature_names_in_"))
                return cols if len(cols) > 0 else None
    except Exception:
        pass

    return None


def align_X_to_model(
    model, df: pd.DataFrame
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Align dataframe columns to what the model expects.
    Returns:
      X_aligned, missing_cols, extra_cols
    """
    exp_cols = expected_input_columns(model)
    if not exp_cols:
        # No reliable schema; use as-is.
        return df, [], []

    extra = sorted(set(df.columns) - set(exp_cols))
    missing = sorted(set(exp_cols) - set(df.columns))
    X_aligned = df.reindex(columns=exp_cols, fill_value=0)
    return X_aligned, missing, extra


def score_proba(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return proba[:, 1] if proba.ndim == 2 and proba.shape[1] >= 2 else proba.ravel()
    if hasattr(model, "decision_function"):
        z = model.decision_function(X)
        return 1 / (1 + np.exp(-z))
    raise RuntimeError("Model does not support probability scoring.")


def precision_recall_lift_at_k(
    y_true: np.ndarray, y_score: np.ndarray, k_frac: float
) -> Dict[str, float]:
    n = len(y_true)
    k = max(1, int(np.ceil(k_frac * n)))
    order = np.argsort(-y_score)
    topk = order[:k]
    baseline = float(np.mean(y_true))
    precision_k = float(np.mean(y_true[topk])) if k > 0 else 0.0
    positives = float(np.sum(y_true))
    recall_k = float(np.sum(y_true[topk]) / positives) if positives > 0 else 0.0
    lift_k = float(precision_k / baseline) if baseline > 0 else 0.0
    return {
        "baseline": baseline,
        "precision_k": precision_k,
        "recall_k": recall_k,
        "lift_k": lift_k,
        "k": k,
    }


def choose_action(prob: float) -> str:
    if prob >= 0.70:
        return "Call"
    if prob >= 0.40:
        return "Offer discount"
    return "Bundle offer"


def get_feature_names(preprocessor, X: pd.DataFrame) -> List[str]:
    if preprocessor is None:
        return list(X.columns)
    if hasattr(preprocessor, "get_feature_names_out"):
        try:
            return list(preprocessor.get_feature_names_out())
        except Exception:
            pass
    return list(X.columns)


def key_drivers_for_rows(
    model, X_raw: pd.DataFrame, row_positions: np.ndarray, max_features: int = 3
) -> List[str]:
    """
    Best-effort per-row drivers.
    - Tree models: SHAP TreeExplainer (if shap installed).
    - Logistic regression: coefficient contributions (approx).
    """
    drivers = [""] * len(row_positions)
    if len(row_positions) == 0:
        return drivers

    preprocessor, estimator = extract_preprocessor_and_estimator(model)

    # Transform for explanation if preprocessor exists
    Xt_all = None
    if preprocessor is not None and hasattr(preprocessor, "transform"):
        try:
            Xt_all = preprocessor.transform(X_raw)
        except Exception:
            Xt_all = None

    # Tree + SHAP
    if shap is not None:
        try:
            est_name = estimator.__class__.__name__.lower()
            if any(k in est_name for k in ["forest", "xgb", "gb", "tree"]):
                Xt = Xt_all if Xt_all is not None else X_raw.to_numpy()
                explainer = shap.TreeExplainer(estimator)
                sv = explainer.shap_values(Xt)
                if isinstance(sv, list):
                    sv = sv[1] if len(sv) > 1 else sv[0]
                sv = np.asarray(sv)

                feature_names = get_feature_names(preprocessor, X_raw)
                for i, ridx in enumerate(row_positions):
                    contrib = sv[ridx]
                    top = np.argsort(-np.abs(contrib))[:max_features]
                    parts = []
                    for j in top:
                        fname = feature_names[j] if j < len(feature_names) else f"f{j}"
                        parts.append(f"{'+' if contrib[j] >= 0 else '-'}{fname}")
                    drivers[i] = ", ".join(parts)
                return drivers
        except Exception:
            pass

    # Logistic regression contributions
    try:
        if hasattr(estimator, "coef_") and Xt_all is not None:
            coef = np.asarray(estimator.coef_).ravel()
            feature_names = get_feature_names(preprocessor, X_raw)

            for i, ridx in enumerate(row_positions):
                row = Xt_all[ridx]
                try:
                    row_dense = np.asarray(row.todense()).ravel()
                except Exception:
                    row_dense = np.asarray(row).ravel()

                if row_dense.shape[0] != coef.shape[0]:
                    continue

                contrib = row_dense * coef
                top = np.argsort(-np.abs(contrib))[:max_features]
                parts = []
                for j in top:
                    fname = feature_names[j] if j < len(feature_names) else f"f{j}"
                    parts.append(f"{'+' if contrib[j] >= 0 else '-'}{fname}")
                drivers[i] = ", ".join(parts)
            return drivers
    except Exception:
        pass

    return drivers


# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("Controls")
root = project_root()

model_choice = st.sidebar.selectbox(
    "Model",
    options=[
        (
            "Logistic Regression",
            str(root / "data" / "models" / "logistic_regression.pkl"),
        ),
        ("Random Forest", str(root / "data" / "models" / "random_forest.pkl")),
    ],
    format_func=lambda x: x[0],
)
model_label, model_path = model_choice

use_uploaded = st.sidebar.checkbox(
    "Upload a CSV instead of using default dataset", value=False
)
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV", type=["csv"], disabled=not use_uploaded
)

st.sidebar.markdown("---")
target_tier = st.sidebar.selectbox("Target tier", options=["Top 10%", "Top 20%"])
tier_frac = 0.10 if target_tier == "Top 10%" else 0.20

min_prob = st.sidebar.slider(
    "Minimum probability threshold (optional)",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.01,
)

max_rows_show = st.sidebar.slider("Max rows to show in table", 50, 1000, 200, 50)

explain_key_drivers = st.sidebar.checkbox(
    "Compute key drivers for targeted customers (may be slower)",
    value=True,
)

# ----------------------------
# Load model
# ----------------------------
st.title("Telco Customer Churn — Retention Targeting")

with st.spinner("Loading model..."):
    try:
        model = load_model(model_path)
    except Exception as e:
        st.error(f"Could not load model from '{model_path}'. Error: {e}")
        st.stop()

# ----------------------------
# Load data
# ----------------------------
if use_uploaded:
    if uploaded_file is None:
        st.info(
            "Upload a CSV to score customers, or uncheck upload to use the default dataset path."
        )
        st.stop()
    df = pd.read_csv(uploaded_file)
    data_path_display = "Uploaded CSV"
else:
    df_default, default_path = load_default_dataset()
    if df_default is None:
        st.warning(
            "Default dataset not found/readable.\n\n"
            "Expected one of:\n"
            "- data/processed/telco_churn_processed.csv\n"
            "- data/processed/telco_cleaned.csv\n"
            "- data/raw/Telco-Customer-Churn.csv\n\n"
            "Please upload a CSV using the sidebar."
        )
        st.stop()
    df = df_default
    data_path_display = str(default_path)

st.caption(f"Dataset source: {data_path_display}")
st.write(f"Rows: {len(df):,} | Columns: {len(df.columns):,}")

# ----------------------------
# Column selection (fixes the CustomerID issue you saw)
# ----------------------------
auto_id = detect_customer_id_col(df)
id_col = st.sidebar.selectbox(
    "Customer ID column",
    options=["(use row index)"] + list(df.columns),
    index=0 if auto_id is None else (1 + list(df.columns).index(auto_id)),
)

auto_label = detect_label_col(df)
label_col = st.sidebar.selectbox(
    "Churn label column (for metrics)",
    options=["(none)"] + list(df.columns),
    index=0 if auto_label is None else (1 + list(df.columns).index(auto_label)),
)

# Normalize selected columns
id_col = None if id_col == "(use row index)" else id_col
label_col = None if label_col == "(none)" else label_col

# Create a stable CustomerID field
df_work = df.copy()
if id_col is None:
    df_work["CustomerID"] = np.arange(len(df_work))
    id_col_used = "CustomerID"
else:
    id_col_used = id_col

# Build y for metrics (if provided)
y_true = None
if label_col is not None:
    y_true = normalize_label(df_work[label_col])

# ----------------------------
# Scoring block
# ----------------------------
st.header("Scoring")

# Start from feature table (remove ID + label if present)
drop_cols = [id_col_used]
if label_col is not None:
    drop_cols.append(label_col)

X_raw = df_work.drop(columns=drop_cols, errors="ignore").copy()

# Align to model schema (THIS fixes your "feature names unseen at fit time" error)
X, missing_cols, extra_cols = align_X_to_model(model, X_raw)

with st.expander("Input feature alignment diagnostics"):
    st.write(
        "The app aligns your dataset columns to the exact columns the model was trained with (when available)."
    )
    st.write(f"Raw feature columns provided: **{len(X_raw.columns)}**")
    if expected_input_columns(model):
        st.write(f"Model expected columns: **{len(expected_input_columns(model))}**")
        st.write(f"Extra columns dropped: **{len(extra_cols)}**")
        if extra_cols:
            st.code(", ".join(extra_cols))
        st.write(f"Missing columns filled with 0: **{len(missing_cols)}**")
        if missing_cols:
            st.code(", ".join(missing_cols))
    else:
        st.info(
            "This model does not expose feature_names_in_. No alignment performed (using columns as-is)."
        )

with st.spinner("Scoring customers..."):
    try:
        churn_prob = score_proba(model, X)
    except Exception as e:
        st.error(
            "Scoring failed.\n\n"
            "Most common causes:\n"
            "- You are passing a processed/encoded dataset but the model expects raw features (or vice versa)\n"
            "- A preprocessing pipeline step is missing from the saved pickle\n\n"
            f"Error: {e}"
        )
        st.stop()

# Use the *actual* selected ID column name in outputs (e.g., 'customerID')
id_display_col = id_col_used

scored = pd.DataFrame(
    {
        id_display_col: df_work[id_col_used].astype(str),
        "churn_probability": churn_prob,
    }
)
scored["Action"] = scored["churn_probability"].apply(choose_action)
if y_true is not None:
    scored["actual_churn"] = y_true.values

# ----------------------------
# Retention Targeting
# ----------------------------
st.header("Retention Targeting")

n_total = len(scored)
k = max(1, int(np.ceil(tier_frac * n_total)))

scored_sorted = scored.sort_values("churn_probability", ascending=False).reset_index(
    drop=True
)
targeted = scored_sorted.head(k)
if min_prob > 0.0:
    targeted = targeted[targeted["churn_probability"] >= min_prob]

# Per-row key drivers (optional)
targeted = targeted.copy()
targeted["key_drivers"] = ""
if explain_key_drivers:
    with st.spinner("Computing key drivers for targeted customers..."):
        try:
            order = np.argsort(
                -churn_prob
            )  # positions in original df_work order (aligned to X_raw)
            top_positions = order[:k]
            if min_prob > 0.0:
                top_positions = top_positions[churn_prob[top_positions] >= min_prob]
            drivers = key_drivers_for_rows(model, X_raw, top_positions, max_features=3)
            targeted["key_drivers"] = drivers
        except Exception:
            targeted["key_drivers"] = ""

# Metrics (requires label)
metric_cols = st.columns(4)
if y_true is not None:
    metrics = precision_recall_lift_at_k(
        y_true.to_numpy(dtype=int),
        churn_prob.astype(float),
        tier_frac,
    )
    metric_cols[0].metric("Baseline (Churn Rate)", f"{metrics['baseline']:.3f}")
    metric_cols[1].metric(f"Precision@{target_tier}", f"{metrics['precision_k']:.3f}")
    metric_cols[2].metric(f"Recall@{target_tier}", f"{metrics['recall_k']:.3f}")
    metric_cols[3].metric(f"Lift@{target_tier}", f"{metrics['lift_k']:.2f}×")
else:
    metric_cols[0].metric("Baseline (Churn Rate)", "—")
    metric_cols[1].metric(f"Precision@{target_tier}", "—")
    metric_cols[2].metric(f"Recall@{target_tier}", "—")
    metric_cols[3].metric(f"Lift@{target_tier}", "—")
    st.info(
        "Select a churn label column in the sidebar to compute Baseline / Precision@k / Recall@k / Lift@k."
    )

st.subheader("Top Segment List")
st.write(
    f"Target tier: **{target_tier}** (top **{k:,}** customers by churn probability) "
    f"{'with' if min_prob > 0 else 'without'} minimum probability threshold."
)

display_cols = [id_display_col, "churn_probability", "Action", "key_drivers"]
targeted_display = targeted[display_cols].head(max_rows_show).copy()
targeted_display["churn_probability"] = targeted_display["churn_probability"].round(4)

st.dataframe(targeted_display, use_container_width=True)

csv_bytes = targeted_display.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download targeted customers (CSV)",
    data=csv_bytes,
    file_name=f"retention_targeting_{target_tier.replace(' ', '').lower()}_{model_label.replace(' ', '_').lower()}.csv",
    mime="text/csv",
)

# ----------------------------
# Model insights
# ----------------------------
with st.expander("Model insights (optional)"):
    preprocessor, estimator = extract_preprocessor_and_estimator(model)
    st.write(f"Loaded model: **{model_label}**")
    st.write(f"Estimator type: `{estimator.__class__.__name__}`")
    exp_cols = expected_input_columns(model)
    if exp_cols:
        st.write(f"Model expected input columns: {len(exp_cols)}")

    # quick global importance
    try:
        if hasattr(estimator, "feature_importances_"):
            importances = np.asarray(estimator.feature_importances_)
            feature_names = get_feature_names(preprocessor, X_raw)
            top = np.argsort(-importances)[:20]
            imp_df = pd.DataFrame(
                {
                    "feature": [
                        feature_names[i] if i < len(feature_names) else f"f{i}"
                        for i in top
                    ],
                    "importance": importances[top],
                }
            )
            st.caption("Top 20 features by model importance (global).")
            st.dataframe(imp_df, use_container_width=True)
        elif hasattr(estimator, "coef_"):
            coef = np.asarray(estimator.coef_).ravel()
            feature_names = get_feature_names(preprocessor, X_raw)
            top_pos = np.argsort(-coef)[:10]
            top_neg = np.argsort(coef)[:10]
            coef_df = pd.DataFrame(
                {
                    "feature": (
                        [
                            feature_names[i] if i < len(feature_names) else f"f{i}"
                            for i in top_pos
                        ]
                        + [
                            feature_names[i] if i < len(feature_names) else f"f{i}"
                            for i in top_neg
                        ]
                    ),
                    "coefficient": np.concatenate([coef[top_pos], coef[top_neg]]),
                }
            )
            st.caption("Top positive/negative coefficients (global).")
            st.dataframe(coef_df, use_container_width=True)
        else:
            st.caption("No global feature importance available for this estimator.")
    except Exception:
        st.caption("Could not compute global feature importance for this model.")

st.markdown("---")
st.caption(
    "Implementation notes:\n"
    "- Top 10% / 20% targeting is ranking by churn_probability.\n"
    "- Lift@k is Precision@k divided by baseline churn rate.\n"
    "- If you want metrics to match your notebook exactly, ensure the app scores the same evaluation split used in the notebook."
)
