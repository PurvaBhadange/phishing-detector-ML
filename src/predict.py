import os
import sys

import joblib
import pandas as pd

from src.features import build_feature_table

MODEL_DIR = "models"


def model_path(name):
    return os.path.join(MODEL_DIR, name.lower().replace(" ", "_") + ".joblib")


def load_model(name):
    return joblib.load(model_path(name))


def best_model_name():
    with open(os.path.join(MODEL_DIR, "best_model.txt")) as f:
        return f.read().strip()


def predict_urls(urls, model):
    features = build_feature_table(urls)
    prob = model.predict_proba(features)[:, 1]
    labels = ["Phishing" if p >= 0.5 else "Legitimate" for p in prob]
    result = pd.DataFrame({
        "URL": list(urls),
        "Prediction": labels,
        "Phishing probability": prob.round(3),
    })
    return result, features


if __name__ == "__main__":
    default_urls = [
        "https://www.google.com",
        "https://www.wikipedia.org/wiki/Machine_learning",
        "https://github.com/scikit-learn/scikit-learn",
        "https://www.amazon.in",
        "http://secure-login.paypal.com.verify-id.xyz/update?id=123",
        "http://192.168.10.5/bank/login.php",
        "http://account-verify-sbi-online.tk/confirm/password",
        "http://paypal-billing-update.top/signin",
    ]
    urls = sys.argv[1:] or default_urls

    name = best_model_name()
    print("Model:", name, "\n")
    result, _ = predict_urls(urls, load_model(name))
    print(result.to_string(index=False))