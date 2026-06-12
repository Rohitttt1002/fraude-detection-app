# streamlit_app_pro.py
"""
Fraud Detection — Polished Streamlit App (pro)
Features:
- Colorful, attractive GUI with custom CSS
- Upload data / use sample dataset
- Full training pipeline (XGBoost baseline) with progress & metrics
- Inference UI with threshold tuning and per-row results
- Model persistence (models/), scaler persistence (models/)
- Evaluation: ROC, Precision-Recall, Confusion Matrix, Classification Report
- SHAP explainability (summary + per row waterfall)
- Evidently report generation (optional)
- Download predictions and model
- Robust error handling and helpful messages

Place data/creditcard.csv in project root or upload a CSV with 'Class' to train.
"""

import os
import io
import base64
import time
import joblib
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, confusion_matrix, classification_report
)

import xgboost as xgb

# Optional features
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, ClassificationPerformancePreset
    EVIDENTLY_AVAILABLE = True
except Exception:
    EVIDENTLY_AVAILABLE = False

# ----------------------
# Config / Paths
# ----------------------
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "fraud_xgb_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"

DATA_DIR = Path("data")
SAMPLE_CSV = DATA_DIR / "creditcard.csv"

st.set_page_config(page_title="Fraud Detection — Pro App", layout="wide", initial_sidebar_state="expanded")

# ----------------------
# Custom CSS for colors & UI
# ----------------------
st.markdown(
    """
    <style>
    /* background gradient */
    .reportview-container {
        background: linear-gradient(135deg, #0f172a 0%, #001219 100%);
        color: #e6eef8;
    }
    .css-1v3fvcr {  /* main background override for some streamlit versions */
        background: linear-gradient(135deg, #0f172a 0%, #001219 100%);
    }
    /* Card style for metric panels */
    .card {
        background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
        border-radius: 12px;
        padding: 12px;
        box-shadow: 0 6px 18px rgba(2,6,23,0.5);
        color: #e6eef8;
    }
    .small-muted { color: #9fb3d5; font-size:12px; }
    .stButton>button {
        border-radius: 10px;
        background: linear-gradient(90deg,#00c2ff,#7b61ff);
        color: white;
    }
    </style>
    """, unsafe_allow_html=True
)

# ----------------------
# Utility functions
# ----------------------
@st.cache_data(ttl=3600)
def load_sample_data():
    if SAMPLE_CSV.exists():
        try:
            df = pd.read_csv(SAMPLE_CSV)
            return df
        except Exception as e:
            st.warning(f"Sample CSV exists but failed to load: {e}")
    # fallback synthetic data if no file
    rng = np.random.RandomState(42)
    n = 10000
    X = rng.normal(size=(n, 15))
    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 16)])
    df["Amount"] = np.abs(rng.normal(scale=50, size=n))
    df["Class"] = rng.choice([0, 1], size=n, p=[0.995, 0.005])
    return df

def save_model_and_scaler(model, scaler):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

@st.cache_data
def load_model():
    try:
        if MODEL_PATH.exists():
            return joblib.load(MODEL_PATH)
    except Exception:
        return None
    return None

@st.cache_data
def load_scaler():
    try:
        if SCALER_PATH.exists():
            return joblib.load(SCALER_PATH)
    except Exception:
        return None
    return None

def download_file(obj, filename, text):
    """Return a anchor link to download pandas DataFrame or bytes"""
    if isinstance(obj, pd.DataFrame):
        csv = obj.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
        return href
    else:
        buf = io.BytesIO()
        joblib.dump(obj, buf)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{text}</a>'
        return href

def prepare_features(df):
    """Keep numeric columns, drop constant columns"""
    X = df.select_dtypes(include=[np.number]).copy()
    X = X.loc[:, X.nunique() > 1]
    return X

def train_xgb(X_train, y_train, X_val=None, y_val=None, early_stopping_rounds=25):
    clf = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
    if X_val is not None and y_val is not None:
        clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=early_stopping_rounds,
            verbose=False
        )
    else:
        clf.fit(X_train, y_train, verbose=False)
    return clf

def plot_roc_pr(y_true, y_proba, container):
    auc = roc_auc_score(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)

    with container:
        fig1 = px.area(
            x=fpr, y=tpr,
            title=f"<b>ROC Curve</b> — AUC={auc:.4f}",
            labels=dict(x="False Positive Rate", y="True Positive Rate"),
            width=700, height=420
        )
        fig1.add_shape(type="line", x0=0, x1=1, y0=0, y1=1, line=dict(dash="dash"))
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.area(
            x=recall, y=precision,
            title=f"<b>Precision-Recall Curve</b> — AP={ap:.4f}",
            labels=dict(x="Recall", y="Precision"),
            width=700, height=420
        )
        st.plotly_chart(fig2, use_container_width=True)

    return auc, ap

def plot_confusion(y_true, y_pred, ax=None):
    cm = confusion_matrix(y_true, y_pred)
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    return ax.get_figure()

# ----------------------
# Sidebar controls
# ----------------------
with st.sidebar:
    st.title("Controls")
    ds_choice = st.radio("Data Source", options=["Sample dataset", "Upload CSV"], index=0)
    uploaded_file = st.file_uploader("Upload CSV (for training/inference)", type=["csv"])
    st.markdown("---")
    st.write("Model actions:")
    train_btn = st.button("Train XGBoost Model", key="train")
    retrain_btn = st.button("Retrain using uploaded data", key="retrain")
    load_model_btn = st.button("Load existing model", key="load")
    download_model_btn = st.button("Download saved model", key="download_model")
    st.markdown("---")
    st.write("Misc:")
    show_shap = st.checkbox("Enable SHAP explanations (may be slow)", value=False)
    show_evidently = st.checkbox("Enable Evidently reports (optional)", value=False)
    st.markdown("App by — Fraud Detection Pro")

# ----------------------
# Main app header (two-column)
# ----------------------
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("<h1 style='color:#e6eef8'>Fraud Detection — Pro Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<div class='small-muted'>Upload your transaction data, train a model, inspect metrics, and explain predictions.</div>", unsafe_allow_html=True)
with col2:
    st.image("https://raw.githubusercontent.com/plotly/datasets/master/logo-plotly.png", width=80)

st.markdown("---")

# ----------------------
# Load data
# ----------------------
if ds_choice == "Sample dataset":
    data = load_sample_data()
    st.success(f"Loaded sample dataset: {data.shape[0]} rows × {data.shape[1]} columns")
else:
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.success(f"Uploaded dataset loaded: {data.shape[0]} rows × {data.shape[1]} columns")
        except Exception as e:
            st.error(f"Failed to load uploaded CSV: {e}")
            data = load_sample_data()
            st.info("Falling back to sample dataset.")
    else:
        st.info("No file uploaded — using sample dataset.")
        data = load_sample_data()

# Quick data info panel
with st.expander("Dataset Snapshot & Info", expanded=True):
    st.write("First rows:")
    st.dataframe(data.head(6))
    st.write("Column types:")
    col_info = pd.DataFrame({"col": data.columns, "dtype": data.dtypes.astype(str), "nunique": data.nunique().values})
    st.dataframe(col_info)
    # class distribution if exists
    if "Class" in data.columns:
        st.markdown("**Class distribution**")
        class_counts = data["Class"].value_counts().reset_index()
        class_counts.columns = ["Class", "Count"]
        fig_bar = px.bar(class_counts, x="Class", y="Count", title="Class distribution", width=700, height=350)
        st.plotly_chart(fig_bar)

# ----------------------
# Tabs: EDA | Train | Inference | Explain | Monitor
# ----------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["EDA", "Train", "Inference", "Explain", "Monitor"])

# ----------------------
# EDA Tab
# ----------------------
with tab1:
    st.header("Exploratory Data Analysis (EDA)")
    numeric_cols = list(data.select_dtypes(include=[np.number]).columns)
    if len(numeric_cols) == 0:
        st.warning("No numeric columns available for EDA.")
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Top correlations")
            corr = data[numeric_cols].corr().abs().unstack().sort_values(ascending=False).drop_duplicates()
            top_corr = corr[corr < 1].head(20)
            st.dataframe(top_corr.reset_index().rename(columns={"level_0":"feature1","level_1":"feature2",0:"abs_corr"}))
        with col_b:
            st.subheader("Distribution sampler")
            feat = st.selectbox("Select feature for histogram", options=numeric_cols, index=0)
            bins = st.slider("Bins", 20, 200, 80)
            fig = px.histogram(data, x=feat, nbins=bins, title=f"Distribution of {feat}", width=700, height=400)
            st.plotly_chart(fig)

# ----------------------
# Train Tab
# ----------------------
with tab2:
    st.header("Training & Model Management")
    model = load_model()
    scaler = load_scaler()

    # Show existing model info
    if model is not None:
        st.success("Saved model loaded.")
        st.write("Model object:")
        st.write(model)
    else:
        st.info("No saved model found. Train a new model.")

    # Training workflow
    st.subheader("Train XGBoost baseline model (recommended workflow)")

    # Prepare dataset for training
    can_train = "Class" in data.columns
    if not can_train:
        st.warning("Training requires a 'Class' column — upload labeled dataset.")
    else:
        test_size = st.slider("Test set fraction", 0.05, 0.4, 0.2)
        val_frac = st.slider("Validation fraction (of train)", 0.0, 0.3, 0.1)
        sampling = st.radio("Imbalance handling", options=["None", "Random Undersampling", "SMOTE"], index=0)

        if train_btn or retrain_btn:
            try:
                with st.spinner("Preparing data..."):
                    X = prepare_features(data.drop(columns=["Class"]))
                    y = data["Class"].loc[X.index]
                    # Align lengths
                    if X.shape[0] != y.shape[0]:
                        y = data["Class"].reset_index(drop=True).loc[X.index]

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
                    if val_frac > 0.0:
                        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_frac, stratify=y_train, random_state=42)
                    else:
                        X_val = y_val = None

                    # scale
                    scaler = StandardScaler()
                    X_train_sc = scaler.fit_transform(X_train)
                    X_test_sc = scaler.transform(X_test)
                    X_val_sc = scaler.transform(X_val) if X_val is not None else None

                    # imbalance handling
                    if sampling == "Random Undersampling":
                        from imblearn.under_sampling import RandomUnderSampler
                        rus = RandomUnderSampler(random_state=42)
                        X_train_sc, y_train = rus.fit_resample(X_train_sc, y_train)
                    elif sampling == "SMOTE":
                        from imblearn.over_sampling import SMOTE
                        sm = SMOTE(random_state=42)
                        X_train_sc, y_train = sm.fit_resample(X_train_sc, y_train)

                # Training (with progress indicator)
                st.info("Training XGBoost (this may take a while)...")
                progress = st.progress(0)
                tic = time.time()
                clf = train_xgb(X_train_sc, y_train, X_val_sc, y_val, early_stopping_rounds=30) if X_val is not None else train_xgb(X_train_sc, y_train)
                elapsed = time.time() - tic
                progress.progress(80)
                # Save model & scaler
                save_model_and_scaler(clf, scaler)
                progress.progress(100)
                st.success(f"Training complete — time: {elapsed:.1f}s. Model saved to {MODEL_PATH}")
                model = clf

                # Evaluate on test
                y_proba_test = clf.predict_proba(X_test_sc)[:,1]
                auc, ap = roc_auc_score(y_test, y_proba_test), average_precision_score(y_test, y_proba_test)
                st.metric("Test ROC-AUC", f"{auc:.4f}")
                st.metric("Test AP (PR AUC)", f"{ap:.4f}")
                st.write("Classification report (0.5 threshold):")
                y_pred_test_05 = (y_proba_test >= 0.5).astype(int)
                st.text(classification_report(y_test, y_pred_test_05))

                # Show ROC/PR
                plot_roc_pr(y_test, y_proba_test, st.container())

                # Save artifacts in session
                # (scaler saved by save_model_and_scaler)
            except Exception as e:
                st.error(f"Training failed: {e}")

    if load_model_btn:
        model = load_model()
        scaler = load_scaler()
        if model is None:
            st.warning("No saved model to load.")
        else:
            st.success("Model loaded from disk.")

    if download_model_btn:
        model = load_model()
        if model is None:
            st.warning("No model to download.")
        else:
            href = download_file(model, "fraud_model.pkl", "Click to download model (.pkl)")
            st.markdown(href, unsafe_allow_html=True)

# ----------------------
# Inference Tab
# ----------------------
with tab3:
    st.header("Inference — Predict on new data")
    model = load_model()
    scaler = load_scaler()
    if model is None:
        st.warning("No model found. Train or load a model first in the 'Train' tab.")
    else:
        st.success("Model loaded for inference.")

        st.subheader("Upload features CSV (no 'Class' needed) or use sample rows")
        pred_file = st.file_uploader("Upload CSV for prediction", type=["csv"], key="predict_file")
        if pred_file is not None:
            try:
                to_predict = pd.read_csv(pred_file)
            except Exception as e:
                st.error(f"Failed to read uploaded file: {e}")
                to_predict = None
        else:
            to_predict = data.drop(columns=["Class"]) if "Class" in data.columns else data.copy()
            to_predict = to_predict.head(200)

        if to_predict is None or to_predict.shape[0] == 0:
            st.warning("No rows available for prediction.")
        else:
            st.write("Preview of input features:")
            st.dataframe(to_predict.head(8))

            # Prepare features
            X_pred = prepare_features(to_predict)
            # align scaler
            if scaler is not None:
                try:
                    X_pred_sc = scaler.transform(X_pred)
                except Exception:
                    st.warning("Scaler mismatch: transforming using fresh StandardScaler (results may be unreliable).")
                    X_pred_sc = StandardScaler().fit_transform(X_pred)
            else:
                X_pred_sc = StandardScaler().fit_transform(X_pred)

            # Predict probabilities
            proba = model.predict_proba(X_pred_sc)[:,1]
            default_thresh = 0.5
            thresh = st.slider("Decision threshold", 0.0, 1.0, float(default_thresh), 0.01)
            preds = (proba >= thresh).astype(int)

            results = to_predict.reset_index(drop=True).copy()
            results["fraud_probability"] = proba
            results["predicted_label"] = preds

            st.write("Top predictions (sorted by fraud probability):")
            st.dataframe(results.sort_values("fraud_probability", ascending=False).head(20))

            # Download predictions
            href = download_file(results, "predictions.csv", "Download predictions as CSV")
            st.markdown(href, unsafe_allow_html=True)

# ----------------------
# Explain Tab (SHAP)
# ----------------------
with tab4:
    st.header("Explainability")
    model = load_model()
    scaler = load_scaler()

    if not SHAP_AVAILABLE or not show_shap:
        st.info("SHAP is not available or disabled. Enable SHAP in the sidebar and install `shap` to use it.")
    else:
        st.success("SHAP available")
        try:
            # Prepare sample for SHAP (load sample data)
            X_all = prepare_features(data.drop(columns=["Class"])) if "Class" in data.columns else prepare_features(data)
            if scaler is not None:
                X_all_sc = scaler.transform(X_all)
            else:
                X_all_sc = StandardScaler().fit_transform(X_all)

            sample_size = st.slider("SHAP sample size (for speed)", 50, min(1000, X_all_sc.shape[0]), 200)
            sample_idx = np.random.choice(np.arange(X_all_sc.shape[0]), size=sample_size, replace=False)
            X_sample = X_all_sc[sample_idx, :]

            explainer = shap.Explainer(model)
            with st.spinner("Computing SHAP values (may take time)..."):
                shap_values = explainer(X_sample)

            st.subheader("SHAP summary (bar)")
            fig_shap = shap.plots.bar(shap_values, show=False)
            st.pyplot(bbox_inches="tight")

            st.subheader("Per-row explanation")
            row_id = st.number_input("Row index in sample (0-based)", min_value=0, max_value=max(0, sample_size-1), value=0)
            st.write("SHAP waterfall for selected row")
            shap.plots.waterfall(shap_values[row_id], show=False)
            st.pyplot(bbox_inches="tight")
        except Exception as e:
            st.error(f"SHAP explainability failed: {e}")

# ----------------------
# Monitor Tab (Evidently)
# ----------------------
with tab5:
    st.header("Monitoring & Reports")
    if not EVIDENTLY_AVAILABLE or not show_evidently:
        st.info("Evidently not enabled or not installed. Toggle Evidently in sidebar and install `evidently` to use.")
    else:
        st.success("Evidently available")
        st.write("Generate a Data Drift + Classification Performance report.")
        ref_file = st.file_uploader("Reference dataset CSV (older data)", type=["csv"], key="ref")
        cur_file = st.file_uploader("Current dataset CSV (newer data)", type=["csv"], key="cur")
        if ref_file is not None and cur_file is not None:
            try:
                ref_df = pd.read_csv(ref_file)
                cur_df = pd.read_csv(cur_file)
                report = Report(metrics=[DataDriftPreset(), ClassificationPerformancePreset()])
                with st.spinner("Running Evidently report (may take a minute)..."):
                    report.run(reference_data=ref_df, current_data=cur_df)
                    html = report.as_html()
                    st.components.v1.html(html, height=800, scrolling=True)
                    b64 = base64.b64encode(html.encode()).decode()
                    st.markdown(f'<a href="data:text/html;base64,{b64}" download="evidently_report.html">Download Evidently report</a>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Evidently report generation failed: {e}")

# ----------------------
# Footer
# ----------------------
st.markdown("---")
st.markdown("<div class='small-muted'>Built with ❤️ — Streamlit · XGBoost · SHAP · Evidently</div>", unsafe_allow_html=True)
