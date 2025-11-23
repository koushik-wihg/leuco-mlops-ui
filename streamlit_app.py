# streamlit_app.py – Lithium Leucogranite ML (direct model, no API)

import io
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.utils.common import read_params
from src.utils.transforms import HFSE_REE_Ratios, PivotILRTransformer
import __main__

# Make custom transformers visible for joblib (safety)
__main__.HFSE_REE_Ratios = HFSE_REE_Ratios
__main__.PivotILRTransformer = PivotILRTransformer

# ==========================
# CONFIG & MODEL LOAD
# ==========================

# Try reading model path from params.yaml (same as api.py)
try:
    CONFIG_PATH = Path("Config/params.yaml")
    params = read_params(CONFIG_PATH)
    MODEL_PATH = Path(params.get("api", {}).get("model_path", "models/final_pipeline.joblib"))
except Exception:
    MODEL_PATH = Path("models/final_pipeline.joblib")


def load_model_artifact(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found at {path}")
    obj = joblib.load(path)
    # Same logic as api.py
    if isinstance(obj, dict):
        if "pipeline" in obj:
            pipeline = obj["pipeline"]
            le = obj.get("label_encoder", None)
            return pipeline, le
        elif "model" in obj:
            model = obj["model"]
            le = obj.get("label_encoder", None)
            return model, le
    if hasattr(obj, "predict"):
        return obj, None
    raise ValueError("Unrecognized model artifact format.")


try:
    model, le = load_model_artifact(MODEL_PATH)
    model_loaded = True
except Exception as e:
    model = None
    le = None
    model_loaded = False
    load_error = str(e)

# Get expected features if available (used for alignment + ratios)
EXPECTED_FEATURES = list(getattr(model, "feature_names_in_", [])) if model_loaded else []

# Li class mapping
LI_CLASS_MAP = {
    0: "poor (Li < 300 ppm)",
    1: "moderate (300–1000 ppm)",
    2: "enriched (>1000 ppm)",
}


def get_li_label(cls_value):
    """
    Map model output (int or '0'/'1'/'2') to human-readable Li class label.
    """
    try:
        key = int(cls_value)
        return LI_CLASS_MAP.get(key, "unknown")
    except Exception:
        # If labels are non-numeric words, just return as-is
        return str(cls_value)


# ==========================
# PREPROCESSING HELPERS
# (mirror api.py logic)
# ==========================
def preprocess_df_for_model(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # Clean column names
    df.columns = df.columns.astype(str).str.strip()

    # Drop Li (leakage prevention)
    if "Li" in df.columns:
        df = df.drop(columns=["Li"])

    # DO NOT drop LOI – model may expect it; if absent we'll handle later

    # Convert everything to numeric; non-numeric -> NaN
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.fillna(0.0)

    # Align to model.feature_names_in_ and create ratio features if needed
    expected = EXPECTED_FEATURES
    if expected:
        # First pass: add ratio features A_B if both A and B exist
        missing = [f for f in expected if f not in df.columns]
        for fname in missing:
            if "_" in fname:
                num, denom = fname.split("_", 1)
                if (num in df.columns) and (denom in df.columns):
                    # create ratio; avoid div-by-zero issues
                    denom_vals = df[denom].replace({0.0: pd.NA})
                    df[fname] = df[num] / denom_vals

        # Second pass: fill any still-missing (including LOI if not present) with 0.0
        missing = [f for f in expected if f not in df.columns]
        for fname in missing:
            df[fname] = 0.0

        # Reorder to exactly match training feature order
        df = df[expected]

        # Cleanup inf / NaN from ratios
        df = df.replace([float("inf"), float("-inf")], pd.NA)
        df = df.fillna(0.0)

    return df


def run_model(df_raw: pd.DataFrame):
    if not model_loaded:
        raise RuntimeError("Model not loaded.")
    df_pre = preprocess_df_for_model(df_raw)
    preds_encoded = model.predict(df_pre)
    probs = model.predict_proba(df_pre) if hasattr(model, "predict_proba") else None

    if le is not None:
        preds_decoded = le.inverse_transform(preds_encoded)
        class_names = list(le.classes_)
    else:
        preds_decoded = preds_encoded
        class_names = [str(c) for c in sorted(set(preds_encoded))]

    results = []
    for i, enc in enumerate(preds_encoded):
        cls_val = preds_decoded[i]
        label = get_li_label(cls_val)
        row = {
            "Predicted_Class": cls_val,
            "Class_Label": label,
        }
        if probs is not None:
            prob_row = probs[i].tolist()
            for j, cname in enumerate(class_names):
                row[f"Prob_{cname}"] = prob_row[j]
            row["Confidence"] = max(prob_row)
        results.append(row)

    return pd.DataFrame(results)


# ==========================
# PARSERS FOR SINGLE-SAMPLE INPUT
# ==========================
def parse_key_value_block(text: str) -> dict:
    rec = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        k, v = line.split("=", 1)
        rec[k.strip()] = v.strip()
    return rec


def parse_single_line_csv_block(text: str) -> pd.DataFrame:
    buf = io.StringIO(text)
    return pd.read_csv(buf)


def read_uploaded_file(f) -> pd.DataFrame:
    name = f.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(f)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(f)
    else:
        raise ValueError("Unsupported file type. Use CSV / XLSX / XLS.")


# ==========================
# LI CLASS LEGEND (UI TEXT)
# ==========================
LI_LEGEND_MD = """
### 🔍 Li Class Legend

| Class | Threshold (ppm) | Interpretation |
|-------|------------------|----------------|
| **0** | Li < 300         | **poor** |
| **1** | 300–1000         | **moderate** |
| **2** | >1000            | **enriched** |

*This model works best for **leucogranite** and **pegmatite** compositions.*
"""


# ==========================
# STREAMLIT LAYOUT (light theme)
# ==========================
st.set_page_config(
    page_title="Lithium Leucogranite ML – Direct Streamlit App",
    layout="wide",
)

# Light background override
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f7f7f9 !important;
        color: #111827 !important;
    }
    .main {
        background-color: #f7f7f9 !important;
    }
    body {
        background-color: #f7f7f9 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

col_left, col_mid, col_right = st.columns([1.3, 3.0, 2.2])

# ----- LEFT: MODEL INFO -----
with col_left:
    st.markdown("### Model Info")

    st.markdown(
        f"""
        <div style="
            border-radius: 0.75rem;
            padding: 0.9rem 1rem;
            background-color: #DBEAFE;
            border: 1px solid #3B82F6;
            font-size: 0.9rem;
            word-break: break-all;">
            <strong>Model path:</strong><br>
            {MODEL_PATH}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")

    if model_loaded:
        n_feats = len(EXPECTED_FEATURES) if EXPECTED_FEATURES else "unknown"
        st.success(f"Model loaded ✅  (features: {n_feats})")
    else:
        st.error("Model failed to load ❌")
        st.caption(load_error if "load_error" in globals() else "Check MODEL_PATH.")

    st.markdown("---")
    st.caption(
        "This app loads the trained pipeline directly (no API). "
        "Raw geochemical data are preprocessed in the same way as in the FastAPI backend."
    )

# ----- MIDDLE: BATCH PREDICTION -----
with col_mid:
    st.markdown("## Lithium Leucogranite ML – Streamlit Interface")
    st.markdown(
        "Upload CSV / Excel files with whole-rock geochemistry to obtain Li class predictions. "
        "Values like `bdl` or blanks are tolerated; they will be coerced safely."
    )

    st.markdown("### Upload CSV / Excel for batch prediction")

    uploaded_files = st.file_uploader(
        "Drag and drop files here",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
        help="Limit 200MB per file • CSV, XLSX, XLS",
    )

    if uploaded_files:
        all_dfs = []
        meta = []

        for f in uploaded_files:
            try:
                df_u = read_uploaded_file(f)
            except Exception as e:
                st.error(f"Failed to read `{f.name}`: {e}")
                continue
            df_u.columns = df_u.columns.astype(str)
            all_dfs.append(df_u)
            meta.append((f.name, df_u.shape))

        if not all_dfs:
            st.stop()

        st.markdown("### Uploaded files")
        for fname, (n_rows, n_cols) in zip([m[0] for m in meta], meta):
            st.markdown(f"**File:** `{fname}` &nbsp;&nbsp; Shape: `{n_rows} × {n_cols}`")

        st.info(
            "Columns `Li` and `LOI` can be present. "
            "`Li` will be excluded to avoid data leakage. "
            "Engineered ratios (e.g. `Ce_Yb`, `Rb_Sr`) are derived automatically where needed."
        )

        df_all = pd.concat(all_dfs, axis=0, ignore_index=True)
        max_rows = len(df_all)

        n_rows_total = st.number_input(
            "Number of rows (total) to run",
            min_value=1,
            max_value=max_rows,
            value=min(50, max_rows),
            step=1,
        )
        df_subset = df_all.iloc[:n_rows_total].copy()

        st.markdown("#### Preview of input")
        st.dataframe(df_subset.head(), use_container_width=True)

        if st.button("🚀 Run batch predictions", type="primary"):
            if not model_loaded:
                st.error("Model not loaded. Check MODEL_PATH.")
            else:
                with st.spinner("Running model on uploaded data..."):
                    try:
                        preds_df = run_model(df_subset)
                    except Exception as e:
                        st.error("Error during prediction:")
                        st.error(str(e))
                    else:
                        combined = pd.concat(
                            [df_subset.reset_index(drop=True), preds_df], axis=1
                        )
                        st.markdown("### 🧠 Batch predictions")
                        st.dataframe(combined, use_container_width=True)

                        st.markdown("---")
                        st.markdown(LI_LEGEND_MD)

                        # Download options
                        csv_bytes = combined.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "⬇️ Download predictions as CSV",
                            data=csv_bytes,
                            file_name="leucogranite_predictions.csv",
                            mime="text/csv",
                        )

                        excel_buf = io.BytesIO()
                        with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
                            combined.to_excel(writer, index=False, sheet_name="Predictions")
                        st.download_button(
                            "⬇️ Download predictions as Excel (.xlsx)",
                            data=excel_buf.getvalue(),
                            file_name="leucogranite_predictions.xlsx",
                            mime=(
                                "application/vnd.openxmlformats-officedocument."
                                "spreadsheetml.sheet"
                            ),
                        )

# ----- RIGHT: SINGLE-SAMPLE INPUT -----
with col_right:
    st.markdown("### Single-sample input")
    st.caption(
        "Enter one sample manually, either as key=value lines or as a single-row CSV. "
        "Column names should match your training schema (SiO2, TiO2, ..., Li, REEs, etc.). "
        "Non-numeric entries like `bdl` are allowed."
    )

    st.markdown(
        """
        - **Key=Value lines:** `SiO2=73.5`  
        - **Single-line CSV:** header line + one data row
        """
    )

    mode = st.radio(
        "Input mode:",
        ["Key=Value lines", "Single-line CSV"],
        index=0,
    )

    if mode == "Key=Value lines":
        example_text = (
            "SiO2=73.5\n"
            "TiO2=0.30\n"
            "Al2O3=14.2\n"
            "Fe2O3=2.0\n"
            "FeO=2.3\n"
            "MgO=0.7\n"
            "CaO=1.8\n"
            "Na2O=3.8\n"
            "K2O=3.0\n"
            "P2O5=0.08\n"
            "MnO=0.04\n"
            "LOI=0.9\n"
            "Li=bdl\n"
            "Rb=120\n"
            "Cs=6\n"
            "Be=3\n"
            "Ta=0.9\n"
            "Nb=8\n"
            "Sn=4\n"
            "W=\n"
            "Ba=530\n"
            "Sr=290\n"
            "Y=9\n"
            "Zr=105\n"
            "Hf=2.8\n"
            "Th=10.2\n"
            "U=2.6\n"
            "La=23\n"
            "Ce=42\n"
            "Pr=4.6\n"
            "Nd=16.5\n"
            "Sm=3.1\n"
            "Eu=0.75\n"
            "Gd=2.6\n"
            "Tb=0.36\n"
            "Dy=2.0\n"
            "Ho=0.33\n"
            "Er=0.83\n"
            "Tm=0.11\n"
            "Yb=0.70\n"
            "Lu=0.10\n"
        )

        text = st.text_area(
            "Paste sample here",
            value=example_text,
            height=340,
        )

        if st.button("Predict single sample", key="single_kv"):
            if not model_loaded:
                st.error("Model not loaded. Check MODEL_PATH.")
            else:
                rec = parse_key_value_block(text)
                if not rec:
                    st.error("No valid `key=value` lines detected.")
                else:
                    df_single = pd.DataFrame([rec])
                    try:
                        preds_df = run_model(df_single)
                    except Exception as e:
                        st.error("Error during prediction:")
                        st.error(str(e))
                    else:
                        st.markdown("#### Prediction result")
                        st.json(preds_df.iloc[0].to_dict())
                        st.markdown("---")
                        st.markdown(LI_LEGEND_MD)

    else:  # Single-line CSV
        example_csv = (
            "SiO2,TiO2,Al2O3,Fe2O3,FeO,MgO,CaO,Na2O,K2O,P2O5,MnO,LOI,Li,Rb,Cs,Be,Ta,Nb,Sn,W,Ba,Sr,Y,Zr,Hf,Th,U,"
            "La,Ce,Pr,Nd,Sm,Eu,Gd,Tb,Dy,Ho,Er,Tm,Yb,Lu\n"
            "73.5,0.30,14.2,2.0,2.3,0.7,1.8,3.8,3.0,0.08,0.04,0.9,bdl,120,6,3,0.9,8,4,,530,290,9,105,2.8,10.2,2.6,"
            "23,42,4.6,16.5,3.1,0.75,2.6,0.36,2.0,0.33,0.83,0.11,0.70,0.10\n"
        )

        text = st.text_area(
            "Paste header + single data row",
            value=example_csv,
            height=280,
        )

        if st.button("Predict single sample", key="single_csv"):
            if not model_loaded:
                st.error("Model not loaded. Check MODEL_PATH.")
            else:
                try:
                    df_single = parse_single_line_csv_block(text)
                except Exception as e:
                    st.error(f"Could not parse CSV block: {e}")
                else:
                    try:
                        preds_df = run_model(df_single)
                    except Exception as e:
                        st.error("Error during prediction:")
                        st.error(str(e))
                    else:
                        st.markdown("#### Prediction result")
                        st.json(preds_df.iloc[0].to_dict())
                        st.markdown("---")
                        st.markdown(LI_LEGEND_MD)

# Footer
st.write("---")
st.caption(
    "Li class legend: 0 = poor (Li < 300 ppm), 1 = moderate (300–1000 ppm), "
    "2 = enriched (>1000 ppm). This model works best for leucogranite and pegmatite compositions."
)
