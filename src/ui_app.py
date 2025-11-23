import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
from pathlib import Path

# Import custom tools from your source code
from src.utils.common import read_params
from src.utils.transforms import HFSE_REE_Ratios, PivotILRTransformer
from src.utils.predict_raw import prepare_data_for_pipeline

import __main__

# -------------------------------------------------------------------
# Hack: Make custom transformers visible during joblib.load
# -------------------------------------------------------------------
__main__.HFSE_REE_Ratios = HFSE_REE_Ratios
__main__.PivotILRTransformer = PivotILRTransformer

# -------------------------------------------------------------------
# Streamlit page config
# -------------------------------------------------------------------
st.set_page_config(page_title="Lithium Leucogranite UI", layout="wide")
st.title("Lithium Leucogranite ML – Streamlit Interface")

# -------------------------------------------------------------------
# Global Constants & Setup
# -------------------------------------------------------------------
CONFIG_PATH = Path("Config/params.yaml")
params = read_params(CONFIG_PATH)
MODEL_PATH = Path(params.get("api", {}).get("model_path", "models/final_pipeline.joblib"))

# Define Class Mapping
CLASS_MAPPING = {
    0: "Poor (0)",
    1: "Moderate (1)",
    2: "Enriched (2)"
}

# Instantiate transformers globally
ratio_transformer = HFSE_REE_Ratios()
ilr_transformer = PivotILRTransformer()

# -------------------------------------------------------------------
# Load Model Logic
# -------------------------------------------------------------------
pipeline = None
label_encoder = None
loaded_from = None

PIPE_PATHS = [
    MODEL_PATH,
    Path("models/best_model_tuned.joblib"),
    Path("models/pipeline.joblib"),
]

for p in PIPE_PATHS:
    if p.exists():
        try:
            obj = joblib.load(p)
            if isinstance(obj, dict):
                if "pipeline" in obj:
                    pipeline = obj["pipeline"]
                    label_encoder = obj.get("label_encoder", None)
                elif "model" in obj:
                    pipeline = obj["model"]
                    label_encoder = obj.get("label_encoder", None)
            else:
                # plain sklearn Pipeline or estimator
                if hasattr(obj, "predict"):
                    pipeline = obj
            loaded_from = p
            break
        except Exception as e:
            st.sidebar.error(f"Failed to load {p}: {e}")

if pipeline is None:
    st.sidebar.warning("No pipeline detected. Please ensure models/final_pipeline.joblib exists.")
else:
    st.sidebar.success(f"Loaded pipeline from {loaded_from}")


# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------

def get_class_names(pipeline, label_encoder):
    """Return ordered class labels for probability columns."""
    if label_encoder is not None:
        return list(label_encoder.classes_)
    if hasattr(pipeline, "classes_"):
        return list(pipeline.classes_)
    if hasattr(pipeline, "named_steps"):
        # Search last step for classes_
        for name, step in reversed(list(pipeline.named_steps.items())):
            if hasattr(step, "classes_"):
                return list(step.classes_)
    return []

def read_uploaded_file(uploaded_file):
    """Reads CSV or Excel into a Pandas DataFrame."""
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(uploaded_file)
    else:
        return pd.read_csv(uploaded_file)

def align_features(df, model):
    """
    Aligns the dataframe columns to exactly match what the model expects.
    1. Adds missing columns (filling with 0).
    2. Removes extra columns.
    3. Reorders columns.
    """
    expected_features = None
    
    # Attempt to find the feature names from the model object
    if hasattr(model, "feature_names_in_"):
        expected_features = model.feature_names_in_
    elif hasattr(model, "named_steps"):
        # Check the last step (estimator)
        if hasattr(model.steps[-1][1], "feature_names_in_"):
            expected_features = model.steps[-1][1].feature_names_in_
            
    if expected_features is not None:
        # 1. Ensure all expected columns exist
        for col in expected_features:
            if col not in df.columns:
                if col == "Source_Sheet":
                    df[col] = 0 # Numeric placeholder for median imputer
                else:
                    df[col] = 0.0
        
        # 2. Select ONLY the expected columns
        df_aligned = df[expected_features].copy()
        return df_aligned
    else:
        return df

def apply_preprocessing(raw_df, model):
    """
    Applies transformations and aligns features to the model.
    """
    df = raw_df.copy()
    
    # 1. Clean column names
    df.columns = [c.strip() for c in df.columns]

    # 2. Handle Missing Values in Raw Data
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(1e-6)

    # 3. Apply Standard Preparation
    try:
        df = prepare_data_for_pipeline(df)
    except Exception:
        pass 

    # 4. Explicitly Calculate Ratios
    if 'La' in df.columns and 'Yb' in df.columns:
        df['La_Yb'] = df['La'] / df['Yb']
    if 'Rb' in df.columns and 'Cs' in df.columns:
        df['Rb_Cs'] = df['Rb'] / df['Cs']
    if 'Rb' in df.columns and 'Sr' in df.columns:
        df['Rb_Sr'] = df['Rb'] / df['Sr']
    if 'Sn' in df.columns and 'W' in df.columns:
        w_safe = df['W'].replace(0, 1e-6)
        df['Sn_W'] = df['Sn'] / w_safe

    # 5. Apply Custom Transformers
    try:
        df = ratio_transformer.fit_transform(df)
    except Exception:
        pass

    try:
        df = ilr_transformer.transform(df)
    except Exception:
        pass

    # 6. ALIGN FEATURES
    df_final = align_features(df, model)

    return df_final

# -------------------------------------------------------------------
# Layout
# -------------------------------------------------------------------
col1, col2 = st.columns([2, 1])

# ==============================
# Left column: Batch Prediction
# ==============================
with col1:
    uploaded_files = st.file_uploader(
        "Upload CSV / Excel files (Raw Geochemistry)",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Predict Uploaded Files"):
        if pipeline is None:
            st.error("No pipeline available.")
        else:
            for uploaded_file in uploaded_files:
                st.markdown(f"### File: {uploaded_file.name}")
                try:
                    df_raw = read_uploaded_file(uploaded_file)
                    st.write(f"Input shape: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
                except Exception as e:
                    st.error(f"Failed to read {uploaded_file.name}: {e}")
                    continue

                try:
                    # --- PROCESS & ALIGN DATA ---
                    features_df = apply_preprocessing(df_raw, pipeline)

                    # Predict
                    preds = pipeline.predict(features_df)
                    probs = pipeline.predict_proba(features_df) if hasattr(pipeline, "predict_proba") else None

                    # Handle Label Decoding (0->Poor, 1->Moderate, 2->Enriched)
                    if label_encoder is not None:
                        preds_decoded = label_encoder.inverse_transform(preds)
                    else:
                        # Map integers to strings using global mapping
                        preds_decoded = [CLASS_MAPPING.get(int(p), p) for p in preds]

                    class_names = get_class_names(pipeline, label_encoder)

                    # Prepare Output DataFrame
                    out_df = df_raw.copy().reset_index(drop=True)
                    out_df["Predicted_Class"] = preds_decoded

                    if probs is not None and class_names:
                        # Map prob columns to meaningful names if they are just 0,1,2
                        clean_class_names = [CLASS_MAPPING.get(int(c), c) if isinstance(c, (int, np.integer)) else c for c in class_names]
                        
                        probs_df = pd.DataFrame(probs, columns=clean_class_names)
                        out_df = pd.concat([out_df, probs_df.reset_index(drop=True)], axis=1)
                        out_df["Confidence"] = probs.max(axis=1)

                    st.dataframe(out_df.head(20))

                    # Download
                    csv_bytes = out_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label=f"Download predictions for {uploaded_file.name}",
                        data=csv_bytes,
                        file_name=f"predictions_{uploaded_file.name}.csv",
                        mime="text/csv",
                    )
                    st.success("Prediction completed.")

                except Exception as e:
                    st.error(f"Prediction failed for {uploaded_file.name}")
                    st.error(f"Error details: {str(e)}")
                    
# =================================
# Right column: Single Sample Input
# =================================
with col2:
    st.subheader("Single sample input")
    st.markdown(
        "Enter one sample manually.\n\n"
        "- **Key=Value lines**: `SiO2=73.5`\n"
        "- **Single-line CSV**: Header + 1 data row"
    )

    manual_mode = st.radio("Input mode:", ["Key=Value lines", "Single-line CSV"])
    sample_input = st.text_area(
        "Paste sample here",
        height=200,
        placeholder="Example:\nSiO2=73.5\nAl2O3=14.2\nRb=500\nCs=50\nLa=20\nYb=2\nSn=10\nW=5\n...",
    )

    if st.button("Predict single sample"):
        if pipeline is None:
            st.error("No pipeline available.")
        else:
            sample_df_raw = None
            try:
                if manual_mode == "Key=Value lines":
                    lines = [L.strip() for L in sample_input.splitlines() if L.strip()]
                    data_dict = {}
                    for L in lines:
                        if "=" in L:
                            k, v = L.split("=", 1)
                            data_dict[k.strip()] = float(v.strip())
                    sample_df_raw = pd.DataFrame([data_dict])
                else:
                    txt = sample_input.strip()
                    sample_df_raw = pd.read_csv(io.StringIO(txt))
            except Exception as e:
                st.error(f"Could not parse input: {e}")

            if sample_df_raw is not None:
                try:
                    features_df = apply_preprocessing(sample_df_raw, pipeline)

                    preds = pipeline.predict(features_df)
                    probs = pipeline.predict_proba(features_df) if hasattr(pipeline, "predict_proba") else None

                    # Decode Label
                    raw_pred = preds[0]
                    if label_encoder is not None:
                        pred_label = label_encoder.inverse_transform([raw_pred])[0]
                    else:
                        # Apply mapping
                        pred_label = CLASS_MAPPING.get(int(raw_pred), raw_pred)

                    st.metric("Predicted Class", f"{pred_label}")
                    
                    class_names = get_class_names(pipeline, label_encoder)
                    if probs is not None and class_names:
                        # Clean class names for table
                        clean_class_names = [CLASS_MAPPING.get(int(c), c) if isinstance(c, (int, np.integer)) else c for c in class_names]
                        prob_series = pd.Series(probs[0], index=clean_class_names)
                        st.table(prob_series.sort_values(ascending=False).to_frame("Probability"))

                except Exception as e:
                    st.error(f"Prediction failed: {e}")

st.markdown("---")
st.markdown("**Legend:** 0 = Poor, 1 = Moderate, 2 = Enriched")