# streamlit_app.py – Lithium Leucogranite ML (API Mode, light theme)

import io
import requests
import pandas as pd
import streamlit as st

# ==========================
# CONFIG
# ==========================
API_BASE = "http://127.0.0.1:8000"
HEALTH_ENDPOINT = f"{API_BASE}/health"
PREDICT_ENDPOINT = f"{API_BASE}/predict"

st.set_page_config(
    page_title="Lithium Leucogranite ML – Streamlit Interface (API Mode)",
    layout="wide",
)

# --- Force light-style background even if global theme is dark ---
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

# ==========================
# HELPERS
# ==========================
def check_api_health():
    try:
        r = requests.get(HEALTH_ENDPOINT, timeout=5)
        if r.status_code == 200:
            data = r.json()
            return data.get("model_exists", False), None
        return False, f"Health endpoint returned status {r.status_code}"
    except Exception as e:
        return False, str(e)


def call_predict_api(records):
    """
    records: list[dict] – df.to_dict(orient="records")
    FastAPI /predict expects a bare list of GeochemRecord objects.
    """
    try:
        resp = requests.post(PREDICT_ENDPOINT, json=records, timeout=60)
        if resp.status_code == 200:
            return resp.json(), None
        else:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            return None, f"API Error: Status {resp.status_code}. Response: {detail}"
    except Exception as e:
        return None, str(e)


def read_uploaded_file(f):
    name = f.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(f)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(f)
    else:
        raise ValueError("Unsupported file type. Please upload CSV, XLSX, or XLS.")


def parse_key_value_block(text: str) -> dict:
    """
    Parse lines like:
      SiO2=73.5
      Al2O3=14.2
      Rb=500
      Cs=50
    into a dict.
    """
    rec = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        k, v = line.split("=", 1)
        rec[k.strip()] = v.strip()
    return rec


def parse_single_line_csv_block(text: str) -> pd.DataFrame:
    """
    Expect:
      SiO2,Al2O3,Rb,Cs,...
      73.5,14.2,500,50,...
    """
    buf = io.StringIO(text)
    return pd.read_csv(buf)


# ==========================
# LAYOUT
# ==========================
col_left, col_mid, col_right = st.columns([1.2, 3.0, 2.0])

# ----- LEFT: API TARGET / STATUS -----
with col_left:
    st.markdown("### API Target")

    st.markdown(
        f"""
        <div style="
            border-radius: 0.75rem;
            padding: 0.9rem 1rem;
            background-color: #DCFCE7;
            border: 1px solid #4ADE80;
            font-size: 0.9rem;
            word-break: break-all;">
            <strong>API Target:</strong><br>
            <a href="{PREDICT_ENDPOINT}" target="_blank">{PREDICT_ENDPOINT}</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")

    model_ok, health_err = check_api_health()
    if model_ok:
        st.success("API is up and model is loaded ✅")
    else:
        st.error("API / model not available ❌")
        if health_err:
            st.caption(f"Health details: {health_err}")

    st.markdown("---")
    st.caption(
        "This interface sends geochemical data to the FastAPI backend "
        "for prediction. Make sure `uvicorn src.api:app --reload` (or your "
        "Docker container) is running on port 8000."
    )

# ----- MIDDLE: UPLOAD & BATCH PREDICTION -----
with col_mid:
    st.markdown("## Lithium Leucogranite ML – Streamlit Interface (API Mode)")
    st.markdown(
        "Upload CSV / Excel files containing raw whole-rock geochemistry. "
        "The app will call the `/predict` API and display class probabilities."
    )

    st.markdown("### Upload CSV / Excel files for API Prediction")

    uploaded_files = st.file_uploader(
        "Drag and drop files here",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
        help="Limit 200MB per file • CSV, XLSX, XLS",
    )

    if uploaded_files:
        st.markdown(
            """
            <div style="
                margin-top:0.5rem;
                padding:0.75rem;
                border-radius:0.5rem;
                background-color:#FEF9C3;
                border:1px solid #FACC15;
                font-size:0.9rem;">
            Using default local API endpoint. Ensure your FastAPI service is running!
            </div>
            """,
            unsafe_allow_html=True,
        )

        all_dfs = []
        file_meta = []

        for f in uploaded_files:
            try:
                df = read_uploaded_file(f)
            except Exception as e:
                st.error(f"Failed to read `{f.name}`: {e}")
                continue

            df.columns = df.columns.astype(str)
            all_dfs.append(df)
            file_meta.append((f.name, df.shape))

        if not all_dfs:
            st.stop()

        st.markdown("### Uploaded Files")
        for (fname, shape) in file_meta:
            n_rows, n_cols = shape
            st.markdown(f"**File:** `{fname}` &nbsp;&nbsp; Shape: `{n_rows} rows × {n_cols} columns`")

        st.info(
            "Columns `Li` and `LOI` may be present in your CSV; the API backend "
            "handles leakage prevention and engineered features (ratios like Ce_Yb, Rb_Sr) automatically. "
            "Values like `bdl` or blanks are tolerated."
        )

        # Combine all into one DF for prediction
        df_all = pd.concat(all_dfs, axis=0, ignore_index=True)
        max_rows = len(df_all)

        n_rows_total = st.number_input(
            "Number of rows (total) to send to API",
            min_value=1,
            max_value=max_rows,
            value=min(50, max_rows),
            step=1,
        )

        if n_rows_total < max_rows:
            df_subset = df_all.iloc[:n_rows_total].copy()
        else:
            df_subset = df_all.copy()

        st.markdown("#### Preview of input sent to API")
        st.dataframe(df_subset.head(), use_container_width=True)

        if st.button("Predict Uploaded Files via API", type="primary"):
            if not model_ok:
                st.error("API / model not available. Start FastAPI first.")
            else:
                with st.spinner("Calling API and running model..."):
                    records = df_subset.to_dict(orient="records")
                    result, err = call_predict_api(records)

                if err:
                    st.error(err)
                else:
                    preds = result.get("predictions", [])
                    if not preds:
                        st.warning("API returned no predictions.")
                    else:
                        preds_df = pd.DataFrame(preds)
                        combined = pd.concat(
                            [df_subset.reset_index(drop=True), preds_df], axis=1
                        )

                        st.markdown("### 🧠 Model Predictions (Batch)")
                        st.dataframe(combined, use_container_width=True)

                        # Download buttons
                        csv_bytes = combined.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "⬇️ Download predictions as CSV",
                            data=csv_bytes,
                            file_name="leucogranite_predictions.csv",
                            mime="text/csv",
                        )

                        excel_buffer = io.BytesIO()
                        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                            combined.to_excel(writer, index=False, sheet_name="Predictions")
                        st.download_button(
                            "⬇️ Download predictions as Excel (.xlsx)",
                            data=excel_buffer.getvalue(),
                            file_name="leucogranite_predictions.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        )

                        if "Predicted_Class" in preds_df.columns:
                            st.markdown("#### Predicted Class Distribution")
                            st.bar_chart(preds_df["Predicted_Class"].value_counts())

    else:
        st.info("Upload one or more CSV/XLSX files to run batch predictions via the API.")

# ----- RIGHT: SINGLE SAMPLE INPUT -----
with col_right:
    st.markdown("### Single sample input")
    st.caption(
        "Enter one sample manually, either as key=value lines or a single-row CSV.\n"
        "Column names must match the training schema (SiO2, TiO2, ..., Li, REEs, etc.). "
        "Non-numeric entries like `bdl` are fine; the API will coerce them."
    )

    st.markdown(
        """
        - **Key=Value lines:** `SiO2=73.5`  
        - **Single-line CSV:** header line + one data row
        """
    )

    input_mode = st.radio(
        "Input mode:",
        options=["Key=Value lines", "Single-line CSV"],
        index=0,
    )

    if input_mode == "Key=Value lines":
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
            "Nb=8\n"
            "Ba=530\n"
            "Sr=290\n"
            "Y=9\n"
            "Zr=105\n"
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
            height=320,
        )

        if st.button("Predict Single Sample", key="single_kv"):
            if not model_ok:
                st.error("API / model not available. Start FastAPI first.")
            else:
                rec = parse_key_value_block(text)
                if not rec:
                    st.error("No valid `key=value` lines detected.")
                else:
                    with st.spinner("Calling API and running model..."):
                        result, err = call_predict_api([rec])

                    if err:
                        st.error(err)
                    else:
                        preds = result.get("predictions", [])
                        if not preds:
                            st.warning("API returned no predictions.")
                        else:
                            st.markdown("#### Prediction Result")
                            st.json(preds[0])

    else:  # Single-line CSV mode
        example_csv = (
            "SiO2,TiO2,Al2O3,Fe2O3,FeO,MgO,CaO,Na2O,K2O,P2O5,MnO,LOI,Li,Rb,Cs,Nb,Ba,Sr,Y,Zr,U,La,Ce,Pr,Nd,Sm,Eu,Gd,Tb,Dy,Ho,Er,Tm,Yb,Lu\n"
            "73.5,0.30,14.2,2.0,2.3,0.7,1.8,3.8,3.0,0.08,0.04,0.9,bdl,120,6,8,530,290,9,105,2.6,23,42,4.6,16.5,3.1,0.75,2.6,0.36,2.0,0.33,0.83,0.11,0.70,0.10\n"
        )

        text = st.text_area(
            "Paste header + single data row",
            value=example_csv,
            height=280,
        )

        if st.button("Predict Single Sample", key="single_csv"):
            if not model_ok:
                st.error("API / model not available. Start FastAPI first.")
            else:
                try:
                    df_single = parse_single_line_csv_block(text)
                except Exception as e:
                    st.error(f"Could not parse CSV block: {e}")
                else:
                    if df_single.shape[0] != 1:
                        st.warning(
                            f"Expected exactly 1 data row, but found {df_single.shape[0]}."
                        )
                    with st.spinner("Calling API and running model..."):
                        records = df_single.to_dict(orient="records")
                        result, err = call_predict_api(records)

                    if err:
                        st.error(err)
                    else:
                        preds = result.get("predictions", [])
                        if not preds:
                            st.warning("API returned no predictions.")
                        else:
                            st.markdown("#### Prediction Result")
                            st.json(preds[0])
