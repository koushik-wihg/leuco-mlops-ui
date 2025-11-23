from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import pandas as pd, joblib, os, io
from pathlib import Path
from src.utils.common import read_params
from typing import Optional, List, Union

from src.utils.transforms import HFSE_REE_Ratios, PivotILRTransformer
import __main__
__main__.HFSE_REE_Ratios = HFSE_REE_Ratios
__main__.PivotILRTransformer = PivotILRTransformer


app = FastAPI(title="Lithium Leucogranite ML API", version="0.1")
CONFIG_PATH = Path("Config/params.yaml")
params = read_params(CONFIG_PATH)
MODEL_PATH = params.get("api", {}).get("model_path", "models/final_pipeline.joblib")


def load_model_artifact(path):
    if not Path(path).exists():
        raise FileNotFoundError(f"Model artifact not found at {path}")
    obj = joblib.load(path)
    if isinstance(obj, dict):
        if 'pipeline' in obj:
            pipeline = obj['pipeline']
            le = obj.get('label_encoder', None)
            return pipeline, le
        elif 'model' in obj:
            model = obj['model']
            le = obj.get('label_encoder', None)
            return model, le
    if hasattr(obj, "predict"):
        return obj, None
    raise ValueError("Unrecognized model artifact format.")


try:
    model, le = load_model_artifact(MODEL_PATH)
except Exception as e:
    model = None
    le = None
    print(f"FATAL: Could not load model artifact: {e}")


@app.get("/health")
def health():
    return {"status": "ok", "model_exists": model is not None}


class GeochemRecord(BaseModel):
    # Main Oxides
    SiO2: Optional[Union[float, str]]
    TiO2: Optional[Union[float, str]]
    Al2O3: Optional[Union[float, str]]
    Fe2O3: Optional[Union[float, str]]
    FeO: Optional[Union[float, str]]
    MgO: Optional[Union[float, str]]
    CaO: Optional[Union[float, str]]
    Na2O: Optional[Union[float, str]]
    K2O: Optional[Union[float, str]]
    P2O5: Optional[Union[float, str]]
    MnO: Optional[Union[float, str]]

    # Trace Elements – allow float, string ("bdl" etc.), or null
    Li: Optional[Union[float, str]] = None
    Rb: Optional[Union[float, str]]
    Cs: Optional[Union[float, str]] = None
    Be: Optional[Union[float, str]] = None
    Ta: Optional[Union[float, str]] = None
    Nb: Optional[Union[float, str]]
    Sn: Optional[Union[float, str]] = None
    W: Optional[Union[float, str]] = None 
    Ba: Optional[Union[float, str]]
    Sr: Optional[Union[float, str]]
    Y: Optional[Union[float, str]]
    Zr: Optional[Union[float, str]]
    Hf: Optional[Union[float, str]] = None
    Th: Optional[Union[float, str]] = None
    U: Optional[Union[float, str]]

    # REEs
    La: Optional[Union[float, str]]
    Ce: Optional[Union[float, str]]
    Pr: Optional[Union[float, str]]
    Nd: Optional[Union[float, str]]
    Sm: Optional[Union[float, str]]
    Eu: Optional[Union[float, str]]
    Gd: Optional[Union[float, str]]
    Tb: Optional[Union[float, str]]
    Dy: Optional[Union[float, str]]
    Ho: Optional[Union[float, str]]
    Er: Optional[Union[float, str]]
    Tm: Optional[Union[float, str]]
    Yb: Optional[Union[float, str]]
    Lu: Optional[Union[float, str]]

    class Config:
        extra = "allow"


@app.post("/predict")
def predict(records: List[GeochemRecord]):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not available or trained yet.")
    
    # Convert list of records to DataFrame
    df = pd.DataFrame([r.dict(exclude_unset=True) for r in records])

    # Clean column names (remove hidden whitespace)
    df.columns = df.columns.str.strip()

    # Drop Li to prevent leakage
    if "Li" in df.columns:
        df = df.drop(columns=["Li"])

    # DO NOT drop LOI here; if the model expects it, we'll handle it below.

    # Ensure numeric + simple imputation
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.fillna(0.0)

    # Align features to model training features
    expected_features = getattr(model, "feature_names_in_", None)

    if expected_features is not None:
        expected_features = list(expected_features)

        # Try to construct ratio features like Ce_Yb, Rb_Cs, etc.
        missing = [f for f in expected_features if f not in df.columns]
        for fname in missing:
            if "_" in fname:
                num, denom = fname.split("_", 1)
                if (num in df.columns) and (denom in df.columns):
                    df[fname] = df[num] / df[denom]

        # Recompute missing after adding ratios
        missing = [f for f in expected_features if f not in df.columns]

        # For remaining missing (including LOI if not present), fill with 0.0
        for fname in missing:
            df[fname] = 0.0

        # Order columns exactly as in training
        df = df[expected_features]

        # Clean up inf/NaNs from ratios
        df = df.replace([float("inf"), float("-inf")], pd.NA)
        df = df.fillna(0.0)

    try:
        preds_encoded = model.predict(df)
        preds_decoded = (
            le.inverse_transform(preds_encoded).tolist()
            if le is not None
            else preds_encoded.tolist()
        )
        probs = model.predict_proba(df) if hasattr(model, "predict_proba") else None

        results = []
        class_names = le.classes_.tolist() if le is not None else None

        for i, _ in enumerate(preds_encoded):
            result = {"Predicted_Class": preds_decoded[i]}

            if probs is not None and class_names is not None:
                prob_row = probs[i].tolist()
                for j, class_name in enumerate(class_names):
                    result[f"Prob_{class_name}"] = prob_row[j]
                result["Confidence"] = max(prob_row)

            results.append(result)

        return {"predictions": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
