import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix,
)
from xgboost import XGBClassifier

from src.data_prep import load_raw_data, engineer_features, get_model_frame
from src.pipeline import build_preprocessor, split_data


def train_baseline(X_train, y_train, preprocessor):
    baseline = Pipeline([
        ("preprocessor", preprocessor),
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=2000)),
    ])
    baseline.fit(X_train, y_train)
    return baseline


def train_xgboost(X_train, y_train, preprocessor):
    neg, pos = np.bincount(y_train)
    scale_pos_weight = neg / pos  # standard imbalance correction for XGBoost

    xgb_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    param_distributions = {
        "clf__n_estimators": [100, 200, 300, 400],
        "clf__max_depth": [3, 4, 5, 6, 8],
        "clf__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "clf__subsample": [0.6, 0.8, 1.0],
        "clf__colsample_bytree": [0.6, 0.8, 1.0],
        "clf__min_child_weight": [1, 3, 5],
        "clf__gamma": [0, 0.1, 0.3],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        xgb_pipeline,
        param_distributions=param_distributions,
        n_iter=40,
        scoring="average_precision",   # PR-AUC — correct metric for rare positives
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_


def evaluate_model(model, X_test, y_test, name="model"):
    proba = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)

    print(f"\n=== {name} ===")
    print(f"ROC-AUC:  {roc_auc_score(y_test, proba):.4f}")
    print(f"PR-AUC:   {average_precision_score(y_test, proba):.4f}")
    print("\nClassification report (threshold=0.5):")
    print(classification_report(y_test, preds, digits=3))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, preds))

    return proba, preds


if __name__ == "__main__":
    df = load_raw_data()
    df = engineer_features(df)
    X, y = get_model_frame(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    preprocessor = build_preprocessor()

    print("Training baseline (Logistic Regression)...")
    baseline = train_baseline(X_train, y_train, preprocessor)
    evaluate_model(baseline, X_test, y_test, name="Baseline (Logistic Regression)")

    print("\nTuning XGBoost (this will take a few minutes)...")
    best_xgb, best_params, best_cv_score = train_xgboost(X_train, y_train, preprocessor)
    print(f"\nBest CV PR-AUC: {best_cv_score:.4f}")
    print(f"Best params: {best_params}")
    evaluate_model(best_xgb, X_test, y_test, name="Tuned XGBoost")

    joblib.dump(best_xgb, "models/xgb_churn_model.pkl")
    print("\nSaved pipeline to models/xgb_churn_model.pkl")