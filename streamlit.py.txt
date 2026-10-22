"""
HimLithium: Machine-Learning Exploration Screener for Lithium Fertility
in Himalayan-Tibetan Leucogranites
=============================================================================
Wadia Institute of Himalayan Geology (WIHG), Dehradun, India
Zenodo Archive: https://doi.org/10.5281/zenodo.23417284
"""

import io
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="HimLithium | Geochemical Screener",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# CONSTANTS & SBP SPECIFICATIONS
# -----------------------------------------------------------------------------
CODA_OXIDES = ['SiO2', 'Al2O3', 'K2O', 'Na2O', 'CaO', 'Fe2O3T', 'MgO', 'MnO', 'TiO2', 'P2O5']

# Orthonormal Sequential Binary Partition (SBP) Matrix
SBP_DICT = {
    'SiO2':    [+1, +1,  0,  0,  0,  0,  0,  0,  0],
    'Al2O3':   [+1, -1, +1,  0,  0,  0,  0,  0,  0],
    'K2O':     [+1, -1, -1, +1,  0,  0,  0,  0,  0],
    'Na2O':    [+1, -1, -1, -1,  0,  0,  0,  0,  0],
    'CaO':     [-1,  0,  0,  0, +1, +1,  0,  0,  0],
    'Fe2O3T':  [-1,  0,  0,  0, +1, -1, +1,  0,  0],
    'MgO':     [-1,  0,  0,  0, +1, -1, -1, +1,  0],
    'MnO':     [-1,  0,  0,  0, +1, -1, -1, -1,  0],
    'TiO2':    [-1,  0,  0,  0, -1,  0,  0,  0, +1],
    'P2O5':    [-1,  0,  0,  0, -1,  0,  0,  0, -1]
}
SBP_DF = pd.DataFrame(SBP_DICT, index=[f"ILR_z{i+1}" for i in range(9)])

TRACES_LIST = ['Rb', 'Sr', 'Ba', 'Be', 'Zr', 'Hf', 'Nb', 'Ta', 'Th', 'U', 'Y']
RATIOS_LIST = ['Rb_Sr', 'Zr_Hf', 'Nb_Ta', 'Th_U', 'Ce_Yb', 'La_Yb', 'La_Sm']
MODEL_FEATURES = [f"ILR_z{i+1}" for i in range(9)] + TRACES_LIST + RATIOS_LIST

CLASS_LABELS = {0: "Li-poor (<300 ppm)", 1: "Li-moderate (300-1000 ppm)", 2: "Li-enriched (>1000 ppm)"}
CLASS_COLORS = {0: "#1f77b4", 1: "#ff7f0e", 2: "#d62728"}

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def clean_col_name(c):
    return re.sub(r'[\(_].*?[\)]?', '', str(c)).strip()

def find_col(target_regex, cols):
    for c in cols:
        core = clean_col_name(c)
        if re.fullmatch(target_regex, core, re.IGNORECASE):
            return c
    return None

def safe_ratio(num, den, eps=1e-5):
    return num / np.maximum(den, eps)

def transform_coda_ilr(df_in):
    """Compute 9 orthonormal ILR balances from 10 major oxides."""
    X = df_in[CODA_OXIDES].clip(lower=1e-5)
    X_norm = X.div(X.sum(axis=1), axis=0)
    
    ilr_df = pd.DataFrame(index=df_in.index)
    for balance_name, row in SBP_DF.iterrows():
        pos_parts = row[row == 1].index.tolist()
        neg_parts = row[row == -1].index.tolist()
        r, s = len(pos_parts), len(neg_parts)
        g_pos = np.exp(np.log(X_norm[pos_parts]).mean(axis=1))
        g_neg = np.exp(np.log(X_norm[neg_parts]).mean(axis=1))
        coef = np.sqrt((r * s) / (r + s))
        ilr_df[balance_name] = coef * np.log(g_pos / g_neg)
    return ilr_df

@st.cache_resource
def load_artifacts():
    """Load serialized XGBoost classifier and median preprocessor imputer."""
    try:
        model = joblib.load("best_model_step2_xgb.joblib")
        imputer = joblib.load("preprocessor_imputer.joblib")
        return model, imputer, None
    except Exception as e:
        return None, None, str(e)

# -----------------------------------------------------------------------------
# UI HEADER & SIDEBAR
# -----------------------------------------------------------------------------
st.title("⛏️ HimLithium: Leucogranite Li-Fertility Screener")
st.markdown(
    """
    **Automated Machine Learning & Compositional Data Analysis (CoDA) Pipeline**  
    *Trained on Himalayan–Tibetan collision-zone whole-rock geochemical suites.*
    """
)

with st.sidebar:
    st.header("⚙️ Configuration")
    decision_mode = st.radio(
        "Decision Policy",
        options=["Standard Argmax (Out-of-Fold / Scientific)", "Bayesian Cost-Sensitive (Target Screening)"],
        help="Bayesian Cost-Sensitive minimizes false-negative risk by heavily penalizing misclassifying Li-enriched targets as Li-poor."
    )
    st.markdown("---")
    st.markdown("📌 **Citation:** [DOI: 10.5281/zenodo.23417284](https://doi.org/10.5281/zenodo.23417284)")
    st.markdown("🏢 *Wadia Institute of Himalayan Geology (WIHG), Dehradun*")

# -----------------------------------------------------------------------------
# FILE INGESTION & DATA PROCESSING
# -----------------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Geochemical Spreadsheet (.xlsx, .csv)", 
    type=["xlsx", "csv"],
    help="Requires standard whole-rock major oxides (SiO2, Al2O3, etc.) and trace elements (Rb, Sr, Ba, Zr, Hf, Nb, Ta, etc.)"
)

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".xlsx"):
            df_upload = pd.read_excel(uploaded_file)
        else:
            df_upload = pd.read_csv(uploaded_file)
            
        st.success(f"Successfully uploaded: **{len(df_upload)} samples**")
        cols = list(df_upload.columns)
        
        # Total iron standardization
        fe2o3_c = find_col(r'(Fe2O3|TFe2O3|Fe2O3T)', cols)
        feo_c   = find_col(r'(FeO|FeOT|TFeO|FeO\*)', cols)
        
        df_proc = df_upload.copy()
        if fe2o3_c and feo_c:
            df_proc['Fe2O3T'] = pd.to_numeric(df_upload[fe2o3_c], errors='coerce').fillna(
                pd.to_numeric(df_upload[feo_c], errors='coerce') * 1.11134
            )
        elif fe2o3_c:
            df_proc['Fe2O3T'] = pd.to_numeric(df_upload[fe2o3_c], errors='coerce')
        elif feo_c:
            df_proc['Fe2O3T'] = pd.to_numeric(df_upload[feo_c], errors='coerce') * 1.11134
        else:
            df_proc['Fe2O3T'] = np.nan
            
        # Other major oxides
        for ox in ['SiO2', 'Al2O3', 'K2O', 'Na2O', 'CaO', 'MgO', 'MnO', 'TiO2', 'P2O5']:
            c_ox = find_col(rf'{ox}', cols)
            df_proc[ox] = pd.to_numeric(df_upload[c_ox], errors='coerce') if c_ox else np.nan
            
        # Traces
        for tr in TRACES_LIST + ['La', 'Ce', 'Sm', 'Yb']:
            c_tr = find_col(rf'{tr}', cols)
            df_proc[tr] = pd.to_numeric(df_upload[c_tr], errors='coerce') if c_tr else np.nan
            
        # Compute ratios
        df_proc['Rb_Sr'] = safe_ratio(df_proc['Rb'], df_proc['Sr'])
        df_proc['Zr_Hf'] = safe_ratio(df_proc['Zr'], df_proc['Hf'])
        df_proc['Nb_Ta'] = safe_ratio(df_proc['Nb'], df_proc['Ta'])
        df_proc['Th_U']  = safe_ratio(df_proc['Th'], df_proc['U'])
        df_proc['Ce_Yb'] = safe_ratio(df_proc['Ce'], df_proc['Yb'])
        df_proc['La_Yb'] = safe_ratio(df_proc['La'], df_proc['Yb'])
        df_proc['La_Sm'] = safe_ratio(df_proc['La'], df_proc['Sm'])
        
        # Compute CoDA ILR
        ilr_df = transform_coda_ilr(df_proc)
        
        # Assemble 27-feature matrix
        X_mat = pd.concat([ilr_df, df_proc[TRACES_LIST], df_proc[RATIOS_LIST]], axis=1)[MODEL_FEATURES]
        
        # Inference
        model, imputer, err = load_artifacts()
        if err or model is None:
            st.error(f"Model artifact error: {err}. Please ensure 'best_model_step2_xgb.joblib' and 'preprocessor_imputer.joblib' are present.")
        else:
            X_imp = pd.DataFrame(imputer.transform(X_mat), columns=MODEL_FEATURES)
            probs = model.predict_proba(X_imp)
            
            if "Bayesian" in decision_mode:
                # Calibrated exploration cost matrix
                C = np.array([
                    [ 0.0, 1.2, 2.5 ],
                    [ 2.0, 0.0, 1.0 ],
                    [ 15.0, 1.0, 0.0 ]
                ])
                expected_risk = probs @ C
                preds = np.argmin(expected_risk, axis=1)
            else:
                preds = np.argmax(probs, axis=1)
                
            # Attach predictions
            df_upload['Pred_Class_Code'] = preds
            df_upload['Pred_Fertility'] = [CLASS_LABELS[p] for p in preds]
            df_upload['Prob_Poor'] = np.round(probs[:, 0], 4)
            df_upload['Prob_Moderate'] = np.round(probs[:, 1], 4)
            df_upload['Prob_Enriched'] = np.round(probs[:, 2], 4)
            
            # -----------------------------------------------------------------
            # METRIC SUMMARY CARDS
            # -----------------------------------------------------------------
            st.markdown("### 📊 Screening Summary")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Analyzed", len(df_upload))
            c2.metric("Li-poor Targets", (preds == 0).sum())
            c3.metric("Li-moderate Targets", (preds == 1).sum())
            c4.metric("Li-enriched Targets", (preds == 2).sum())
            
            # -----------------------------------------------------------------
            # TERNARY PROBABILITY VISUALIZATION
            # -----------------------------------------------------------------
            st.markdown("### 🔺 Ternary Confidence Simplex")
            
            sample_ids = df_upload.iloc[:, 0].astype(str) if len(df_upload.columns) > 0 else df_upload.index.astype(str)
            
            fig = px.scatter_ternary(
                df_upload,
                a="Prob_Poor",
                b="Prob_Moderate",
                c="Prob_Enriched",
                color="Pred_Fertility",
                color_discrete_map={
                    CLASS_LABELS[0]: CLASS_COLORS[0],
                    CLASS_LABELS[1]: CLASS_COLORS[1],
                    CLASS_LABELS[2]: CLASS_COLORS[2]
                },
                hover_name=sample_ids,
                hover_data={"Prob_Poor": ':.3f', "Prob_Moderate": ':.3f', "Prob_Enriched": ':.3f'},
                labels={"Prob_Poor": "P(Poor)", "Prob_Moderate": "P(Mod)", "Prob_Enriched": "P(Enriched)"},
                title="Model Posterior Probability Simplex"
            )
            fig.update_layout(
                ternary=dict(
                    sum=1,
                    aaxis=dict(title='Li-Poor'),
                    baxis=dict(title='Li-Moderate'),
                    caxis=dict(title='Li-Enriched')
                )
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # -----------------------------------------------------------------
            # RESULTS TABLE & DOWNLOAD
            # -----------------------------------------------------------------
            st.markdown("### 📋 Classification Results")
            st.dataframe(df_upload)
            
            csv_buf = io.StringIO()
            df_upload.to_csv(csv_buf, index=False)
            st.download_button(
                label="📥 Download Full Predictions CSV",
                data=csv_buf.getvalue(),
                file_name="HimLithium_Predicted_Results.csv",
                mime="text/csv"
            )
            
    except Exception as ex:
        st.error(f"Error processing dataset: {ex}")
else:
    st.info("👆 Upload an Excel or CSV file above to begin exploration screening.")