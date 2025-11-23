import joblib, optuna, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from src.utils.common import read_params
from sklearn.pipeline import Pipeline
from src.utils.transforms import HFSE_REE_Ratios, PivotILRTransformer
import warnings; warnings.filterwarnings('ignore')


def main():
    CONFIG_PATH = Path("Config/params.yaml")
    params = read_params(CONFIG_PATH)
    target_col = params['base']['target_column']

    processed_data_path = Path("data/processed/data_processed.pkl")
    if not processed_data_path.exists():
        print(f"Error: Processed data not found at {processed_data_path}. Please run data_processing first.")
        return

    data = pd.read_pickle(processed_data_path)

    X = data.drop(columns=[target_col] + params['data_processing']['passthrough_features'], errors='ignore')
    y = data[target_col]

    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y), name=target_col)
    print(f"Target labels encoded: {list(le.classes_)} -> {list(range(len(le.classes_)))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=params['data_ingestion']['test_size'],
        random_state=params['data_ingestion']['random_state'],
        stratify=y
    )

    xgb_default = params['model_trainer'].get('xgb_default', {})

    def objective(trial):
        optuna_params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 800),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        }
        full_params = {**optuna_params, **xgb_default}

        m = XGBClassifier(use_label_encoder=False, **full_params)
        m.fit(X_train, y_train)
        return m.score(X_test, y_test)

    try:
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=params['model_trainer']['n_trials'])
    except Exception as e:
        print(f"Optuna failed during optimization: {e}")
        study = type('', (object,), {'best_params': {}})()
        study.best_params = params['model_trainer'].get('xgb_default', {})
        print("Falling back to default model parameters due to Optuna failure.")

    print("Best parameters found:", study.best_params)

    final_model_params = {**study.best_params, **xgb_default}
    model = XGBClassifier(use_label_encoder=False, **final_model_params)
    model.fit(X_train, y_train)

    Path("models").mkdir(exist_ok=True)
    preproc_path = Path("models/preprocessing_pipeline.joblib")
    preprocessor = None
    if preproc_path.exists():
        try:
            preprocessor = joblib.load(preproc_path)
            print(f"Loaded preprocessor from {preproc_path}")
        except Exception as e:
            print(f"Warning: could not load preprocessor: {e}")
            preprocessor = None
    else:
        print("Warning: preprocessing_pipeline.joblib not found. Ensure data_processing.py was run.")

    if preprocessor is not None:
        full_pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", model)])
    else:
        full_pipeline = Pipeline([("classifier", model)])
        print("Saved pipeline contains only the classifier because preprocessor was not found.")

    joblib.dump({"pipeline": full_pipeline, "label_encoder": le}, "models/final_pipeline.joblib")
    joblib.dump({'model': model, 'label_encoder': le}, "models/best_model_tuned.joblib")
    print("Saved models/final_pipeline.joblib (pipeline + label_encoder) and models/best_model_tuned.joblib (compat).")

    # Quick verification print
    try:
        obj = joblib.load("models/final_pipeline.joblib")
        pipe = obj.get('pipeline') if isinstance(obj, dict) else obj
        print('final_pipeline contains preprocessor?', 'preprocessor' in getattr(pipe, 'named_steps', {}))
    except Exception as e:
        print('Verification load failed:', e)

if __name__ == "__main__":
    main()
