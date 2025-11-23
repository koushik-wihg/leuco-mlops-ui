import streamlit as st
import pandas as pd
import numpy as np
import io
import json
import requests
from pathlib import Path

# --- Removed: Local model imports ---
# from src.utils.common import read_params
# from src.utils.transforms import HFSE_REE_Ratios, PivotILRTransformer
# from src.utils.predict_raw import prepare_data_for_pipeline

# -------------------------------------------------------------------
# Configuration: API Endpoint for MLOps
# -------------------------------------------------------------------
# NOTE: Update this URL if your API is deployed remotely or on a different port/IP.
API_ENDPOINT = "http://127.0.0.1:8000/predict"

# -------------------------------------------------------------------
# Streamlit page config
# -------------------------------------------------------------------
st.set_page_config(page_title="Lithium Leucogranite UI", layout="wide")
st.title("Lithium Leucogranite ML – Streamlit Interface (API Mode)")
st.sidebar.success(f"API Target: {API_ENDPOINT}")

# -------------------------------------------------------------------
# Global Constants
# -------------------------------------------------------------------
# Define Class Mapping (Matches expected API output labels)
CLASS_MAPPING = {
    0: "Poor (0)",
    1: "Moderate (1)",
    2: "Enriched (2)"
}

# -------------------------------------------------------------------
# Helper Function for API Communication
# -------------------------------------------------------------------
def get_predictions_from_api(raw_data_df):
    """
    Sends raw DataFrame records to the deployed API and returns predictions.
    The API must handle all preprocessing and model inference.
    """
    try:
        # Convert DataFrame to JSON format expected by the API (orient='records')
        data_json = raw_data_df.to_json(orient='records')
        
        # Send POST request
        response = requests.post(
            API_ENDPOINT, 
            json=json.loads(data_json),
            # Set a timeout in seconds (optional)
            timeout=30 
        )
        
        # Check for successful response
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: Status {response.status_code}. Response: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        st.error(f"Connection Error: Could not reach API endpoint. Is the API running at {API_ENDPOINT}?")
        return None
    except requests.exceptions.Timeout:
        st.error(f"API Timeout: The request took too long ({30} seconds) to complete.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during API call: {e}")
        return None

def read_uploaded_file(uploaded_file):
    """Reads CSV or Excel into a Pandas DataFrame."""
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(uploaded_file)
    else:
        return pd.read_csv(uploaded_file)


# -------------------------------------------------------------------
# Layout
# -------------------------------------------------------------------
col1, col2 = st.columns([2, 1])

# ==============================
# Left column: Batch Prediction
# ==============================
with col1:
    uploaded_files = st.file_uploader(
        "Upload CSV / Excel files (Raw Geochemistry) for API Prediction",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Predict Uploaded Files via API"):
        
        if API_ENDPOINT == "http://127.0.0.1:8000/predict" and not Path(API_ENDPOINT).exists():
             st.warning("Using default local API endpoint. Ensure your service is running!")

        for uploaded_file in uploaded_files:
            st.markdown(f"### File: {uploaded_file.name}")
            try:
                df_raw = read_uploaded_file(uploaded_file)
                st.write(f"Input shape: {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
            except Exception as e:
                st.error(f"Failed to read {uploaded_file.name}: {e}")
                continue

            try:
                # --- CALL THE API ---
                api_result = get_predictions_from_api(df_raw)

                if api_result and 'predictions' in api_result:
                    
                    # Prepare Output DataFrame
                    out_df = df_raw.copy().reset_index(drop=True)
                    
                    # The API returns a list of dictionaries, convert it back to a DataFrame
                    predictions_df = pd.DataFrame(api_result['predictions'])
                    
                    # Concatenate raw data with prediction results
                    out_df = pd.concat([out_df, predictions_df.reset_index(drop=True)], axis=1)

                    st.dataframe(out_df.head(20))

                    # Download
                    csv_bytes = out_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label=f"Download predictions for {uploaded_file.name}",
                        data=csv_bytes,
                        file_name=f"predictions_API_{uploaded_file.name}.csv",
                        mime="text/csv",
                    )
                    st.success("Prediction completed via API.")

                elif api_result:
                     st.error("API response was successful but did not contain the expected 'predictions' key.")

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
        placeholder="Example:\nSiO2=73.5\nAl2O3=14.2\nRb=500\nCs=50\n...",
    )

    if st.button("Predict single sample via API"):
        
        if API_ENDPOINT == "http://127.0.0.1:8000/predict" and not Path(API_ENDPOINT).exists():
             st.warning("Using default local API endpoint. Ensure your service is running!")
             
        sample_df_raw = None
        try:
            if manual_mode == "Key=Value lines":
                lines = [L.strip() for L in sample_input.splitlines() if L.strip()]
                data_dict = {}
                for L in lines:
                    if "=" in L:
                        k, v = L.split("=", 1)
                        try:
                            data_dict[k.strip()] = float(v.strip())
                        except ValueError:
                            data_dict[k.strip()] = v.strip()
                sample_df_raw = pd.DataFrame([data_dict])
            else:
                txt = sample_input.strip()
                sample_df_raw = pd.read_csv(io.StringIO(txt))
        except Exception as e:
            st.error(f"Could not parse input: {e}")

        if sample_df_raw is not None:
            try:
                # --- CALL THE API ---
                api_result = get_predictions_from_api(sample_df_raw)

                if api_result and 'predictions' in api_result and api_result['predictions']:
                    
                    # Single prediction result is the first element in the list
                    pred = api_result['predictions'][0]
                    
                    pred_label = pred.get("Predicted_Class", "N/A")
                    st.metric("Predicted Class", f"{pred_label}")
                    
                    # Display probabilities/confidences
                    prob_data = {k: v for k, v in pred.items() if k not in ["Predicted_Class", "Confidence"]}
                    if prob_data:
                        prob_series = pd.Series(prob_data)
                        st.table(prob_series.sort_values(ascending=False).to_frame("Probability"))
                        
                elif api_result:
                    st.error("API response was successful but no prediction data was returned.")

            except Exception as e:
                st.error(f"Prediction failed: {e}")

st.markdown("---")
st.markdown("**MLOps Compliance:** UI calls the API for prediction, separating presentation from ML inference.")