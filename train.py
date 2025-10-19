# train.py
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from packaging import version

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import sklearn

DATA_PATH = "data/heart.csv"
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True, parents=True)

FEATURES = [
    "age","sex","cp","trestbps","chol","fbs","restecg",
    "thalach","exang","oldpeak","slope","ca","thal"
]
TARGET = "target"
MODEL_VERSION = "v1.0"

def load_data():
    df = pd.read_csv(DATA_PATH)
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")
    df.replace("?", np.nan, inplace=True)
    df[FEATURES] = df[FEATURES].astype("float64")
    df[TARGET] = df[TARGET].astype(int)
    return df

def make_preprocessor():
    # No Pipeline, no FunctionTransformer, no lambda — JUST an imputer.
    return SimpleImputer(strategy="median")

def train_models(X_train, y_train, X_val, y_val):
    lr = LogisticRegression(max_iter=1000, n_jobs=None, random_state=42)
    lr.fit(X_train, y_train)
    lr_auc = roc_auc_score(y_val, lr.predict_proba(X_val)[:, 1])

    xgb = XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        tree_method="hist", nthread=4, random_state=42, reg_lambda=1.0
    )
    xgb.fit(X_train, y_train)
    xgb_auc = roc_auc_score(y_val, xgb.predict_proba(X_val)[:, 1])

    return ("xgb", xgb, xgb_auc) if xgb_auc >= lr_auc else ("lr", lr, lr_auc)

def make_calibrator(model, X_val, y_val):
    if version.parse(sklearn.__version__) >= version.parse("1.2"):
        cal = CalibratedClassifierCV(estimator=model, method="sigmoid", cv="prefit")
    else:
        cal = CalibratedClassifierCV(base_estimator=model, method="sigmoid", cv="prefit")
    cal.fit(X_val, y_val)
    return cal

def main():
    print(">> loading data")
    df = load_data()
    X = df[FEATURES].values
    y = df[TARGET].values

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    print(">> preprocessing")
    pre = make_preprocessor()
    X_train_p = pre.fit_transform(X_train)
    X_val_p   = pre.transform(X_val)
    X_test_p  = pre.transform(X_test)

    print(">> training models")
    model_name, model, val_auc = train_models(X_train_p, y_train, X_val_p, y_val)
    print(f">> best model: {model_name} (Val AUROC={val_auc:.3f})")

    print(">> calibrating")
    calibrator = make_calibrator(model, X_val_p, y_val)

    test_prob = calibrator.predict_proba(X_test_p)[:, 1]
    auc = roc_auc_score(y_test, test_prob)
    brier = brier_score_loss(y_test, test_prob)
    print(f"TEST AUROC={auc:.3f} | Brier={brier:.3f}")

    print(">> saving artifacts")
    joblib.dump({
        "model_version": MODEL_VERSION,
        "model_name": model_name,
        "preproc": pre,           # <- SimpleImputer (pickle-safe)
        "model": model,           # <- LR or XGB
        "calibrator": calibrator, # <- CalibratedClassifierCV
        "features": FEATURES
    }, MODELS_DIR / "model.joblib")

    # SHAP optional — don’t crash if not available
    try:
        import shap
        rng = np.random.RandomState(42)
        bg_idx = rng.choice(X_train_p.shape[0], size=min(500, X_train_p.shape[0]), replace=False)
        background = X_train_p[bg_idx]
        if model_name == "xgb":
            explainer = shap.TreeExplainer(model, data=background, feature_perturbation="interventional")
        else:
            explainer = shap.LinearExplainer(model, background)
        joblib.dump(explainer, MODELS_DIR / "explainer.joblib")
        print("Saved explainer.joblib")
    except Exception as e:
        print(f"(Skipping SHAP explainer: {e})")

    print("Saved model.joblib")

if __name__ == "__main__":
    main()
