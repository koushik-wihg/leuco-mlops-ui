import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.utils.common import read_params
import joblib
import numpy as np


def clr_transform(X):
    X = np.asarray(X, dtype=float)
    offset = 1e-9
    X = X + offset
    logX = np.log(X)
    gm = np.mean(logX, axis=1, keepdims=True)
    clr = logX - gm
    return clr


def get_preprocessor(params, numerical_cols):
    impute_strategy = params['data_processing']['impute_strategy']
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy=impute_strategy)),
        ('clr', FunctionTransformer(clr_transform, validate=False)),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=params['data_processing']['pca_variance_threshold']))
    ])
    return ColumnTransformer([('num', numerical_pipeline, numerical_cols)], remainder='passthrough')


def main():
    CONFIG_PATH = Path("Config/params.yaml")
    params = read_params(CONFIG_PATH)
    target_col = params['base']['target_column']
    raw_data_path = Path(params['data_ingestion']['output_file'])
    passthrough = params['data_processing']['passthrough_features']

    processed_data_path = Path("data/processed/data_processed.pkl")
    pipeline_path = Path("models/preprocessing_pipeline.joblib")
    feature_list_path = Path("models/feature_names.pkl")

    try:
        data = pd.read_csv(raw_data_path)
    except FileNotFoundError:
        print("Error: Raw data not found. Run data_ingestion first.")
        return

    X = data.drop(columns=[target_col], errors='ignore')
    y = data[target_col] if target_col in data.columns else pd.Series([None]*len(data))
    num_cols = [c for c in X.columns if X[c].dtype in ['int64', 'float64'] and c not in passthrough]
    if not num_cols:
        print("Error: No numerical columns found for preprocessing/PCA.")
        return

    preprocessor = get_preprocessor(params, num_cols)
    X_trans = preprocessor.fit_transform(X)

    # Extract PCA and ensure numeric dtype
    pca = preprocessor.named_transformers_['num'].named_steps['pca']
    pca_n = pca.n_components_
    pca_cols = [f'PCA_{i+1}' for i in range(pca_n)]

    X_arr = np.asarray(X_trans)
    pca_array = X_arr[:, :pca_n]
    passthrough_array = X_arr[:, pca_n:]

    pca_df = pd.DataFrame(pca_array, columns=pca_cols).astype(float)
    remainder_cols = [c for c in X.columns if c not in num_cols]
    passthrough_df = pd.DataFrame(passthrough_array, columns=remainder_cols)

    final = pd.concat([pca_df.reset_index(drop=True), passthrough_df.reset_index(drop=True), pd.Series(y).reset_index(drop=True)], axis=1)

    # Defensive: ensure PCA cols are numeric
    for c in pca_cols:
        final[c] = pd.to_numeric(final[c], errors='raise')

    processed_data_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, pipeline_path)
    final.to_pickle(processed_data_path)
    joblib.dump({'numerical_cols': num_cols, 'pca_cols': pca_cols, 'remainder_cols': remainder_cols}, feature_list_path)

    print(f"Data processed and saved. PCA components found: {pca_n}")

if __name__ == "__main__":
    main()
