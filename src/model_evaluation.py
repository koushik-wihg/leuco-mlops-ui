import pandas as pd, joblib, json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns, matplotlib.pyplot as plt
from src.utils.common import read_params

def main():
    CONFIG_PATH = Path("Config/params.yaml")
    params = read_params(CONFIG_PATH)
    ycol = params['base']['target_column']
    df = pd.read_pickle("data/processed/data_processed.pkl")
    X = df.drop(columns=[ycol] + params['data_processing']['passthrough_features'])
    y = df[ycol]

    try:
        model_and_le = joblib.load("models/best_model_tuned.joblib")
        le = model_and_le['label_encoder']
    except Exception:
        print("Error: Could not load model or LabelEncoder. Ensure model_trainer ran successfully.")
        return

    y_encoded = le.transform(y)

    X_train, X_test, y_train_orig, y_test_orig = train_test_split(X, y_encoded, test_size=params['data_ingestion']['test_size'],
                                                             random_state=params['data_ingestion']['random_state'], stratify=y_encoded)

    model = model_and_le['model']
    y_pred_encoded = model.predict(X_test)

    acc = accuracy_score(y_test_orig, y_pred_encoded)
    p, r, f1, _ = precision_recall_fscore_support(y_test_orig, y_pred_encoded, average='macro')
    res = dict(accuracy=acc, precision=p, recall=r, f1=f1)

    Path("metrics").mkdir(exist_ok=True)
    with open("metrics/final_metrics.json", "w") as f: json.dump(res, f, indent=2)

    cm = confusion_matrix(y_test_orig, y_pred_encoded)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title("Confusion Matrix")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("metrics/confusion_matrix.png")
    plt.close()
    print("Evaluation complete.")

if __name__ == "__main__":
    main()
