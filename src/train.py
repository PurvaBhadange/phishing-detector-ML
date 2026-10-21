import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.features import FEATURE_COLUMNS, build_feature_table
from src.predict import MODEL_DIR, model_path

DATA_PATH = "data/PhiUSIIL_Phishing_URL_Dataset.csv"
METRICS = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]


def get_importance(model):
    # logistic regression sits inside a pipeline, trees expose feature_importances_
    final = model[-1] if isinstance(model, Pipeline) else model
    if hasattr(final, "coef_"):
        values = abs(final.coef_[0])
    else:
        values = final.feature_importances_
    return values / values.sum()


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH, usecols=["URL", "label"])
    df = df.dropna().drop_duplicates(subset="URL").reset_index(drop=True)

    # in this dataset label 0 = phishing and 1 = legitimate, so flip it: phishing = 1
    y = (df["label"] == 0).astype(int)
    print("Rows:", len(df))
    print("Phishing:", y.sum(), "| Legitimate:", (y == 0).sum())

    print("Extracting features (takes a minute or two)...")
    X = build_feature_table(df["URL"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "Decision Tree": DecisionTreeClassifier(max_depth=12, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                 eval_metric="logloss", random_state=42),
    }

    rows = []
    importance = pd.DataFrame({"feature": FEATURE_COLUMNS})
    for name, model in models.items():
        print("Training", name)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred),
            "F1": f1_score(y_test, pred),
            "ROC-AUC": roc_auc_score(y_test, prob),
        })
        importance[name] = get_importance(model)
        joblib.dump(model, model_path(name))

    results = pd.DataFrame(rows).round(4)
    print("\n", results.to_string(index=False))

    best_name = results.sort_values("F1", ascending=False).iloc[0]["Model"]
    best_model = models[best_name]
    print("\nBest model:", best_name)

    results.to_csv(os.path.join(MODEL_DIR, "results.csv"), index=False)
    importance.to_csv(os.path.join(MODEL_DIR, "feature_importance.csv"), index=False)
    with open(os.path.join(MODEL_DIR, "best_model.txt"), "w") as f:
        f.write(best_name)

    # chart 1: model comparison
    low = results[METRICS].min().min()
    results.set_index("Model")[METRICS].plot(kind="bar", figsize=(10, 5), ylim=(max(0, low - 0.05), 1))
    plt.title("Model comparison on test data")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "model_comparison.png"))
    plt.close()

    # chart 2: confusion matrix of the best model
    cm = confusion_matrix(y_test, best_model.predict(X_test))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legitimate", "Phishing"], yticklabels=["Legitimate", "Phishing"])
    plt.title(f"Confusion matrix - {best_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"))
    plt.close()

    # chart 3: top features of the best model
    top = importance[["feature", best_name]].sort_values(best_name).tail(10)
    plt.barh(top["feature"], top[best_name])
    plt.title(f"Top 10 features - {best_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "top_features.png"))
    plt.close()

    print("Saved models and charts in the models/ folder")


if __name__ == "__main__":
    main()