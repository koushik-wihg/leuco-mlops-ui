# src/utils/predict_raw.py
import re
import numpy as np
import pandas as pd

OXIDE_REGEX = re.compile(
    r"^(sio2|tio2|al2o3|fe2o3|feo|mno|mgo|cao|na2o|k2o|p2o5|so3|loi)$", re.I
)

REE_LIST = [
    "La","Ce","Pr","Nd","Sm","Eu","Gd","Tb","Dy","Ho",
    "Er","Tm","Yb","Lu","Y","Sc"
]

DROP_COLS = [
    "Li",
    "Li_Class", "Li_Label",
    "Pred_Class", "Pred_Label",
    "Conf_Max", "Prob_poor", "Prob_moderate", "Prob_enriched",
]


def prepare_data_for_pipeline(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Reproduce the notebook behaviour:
      - strip column names
      - drop Li / label / prediction columns if present
      - keep only numeric majors + trace/REE + any other numeric columns
      - DO NOT require Ce_Yb or ilr_* from the user
    The pipeline itself will build ratios + ILR internally.
    """
    df = df_in.copy()
    df.columns = [c.strip() for c in df.columns]

    # drop targets / helper columns if present
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

    # numeric only
    X_num = df.apply(pd.to_numeric, errors="coerce")

    majors, traces, others = [], [], []
    for col in X_num.columns:
        if OXIDE_REGEX.match(col):
            majors.append(col)
        elif col.split(".")[0] in REE_LIST or any(
            h in col.lower() for h in ["ppm", "trace"]
        ):
            traces.append(col)
        else:
            others.append(col)

    transform_input_cols = majors + traces + others
    if not transform_input_cols:
        raise RuntimeError("No usable geochemical columns found in upload.")

    X_ordered = X_num[transform_input_cols].copy()
    return X_ordered
