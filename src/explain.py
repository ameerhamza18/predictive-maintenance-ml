import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_prep import load_raw_data, engineer_features, get_model_frame
from src.pipeline import split_data


def get_feature_names(preprocessor):

    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_names = cat_encoder.get_feature_names_out(["type", "tool_wear_bucket"])
    num_names = preprocessor.transformers_[1][2]  # the passthrough numeric list
    return list(cat_names) + list(num_names)


def build_explainer(model_pipeline, X_background):
   
    preprocessor = model_pipeline.named_steps["preprocessor"]
    clf = model_pipeline.named_steps["clf"]

    X_transformed = preprocessor.transform(X_background)
    feature_names = get_feature_names(preprocessor)

    explainer = shap.TreeExplainer(clf)
    return explainer, feature_names, X_transformed


def plot_global_summary(explainer, X_transformed, feature_names, save_path="assets/shap_summary.png"):
    shap_values = explainer.shap_values(X_transformed)

    plt.figure()
    shap.summary_plot(
        shap_values, X_transformed,
        feature_names=feature_names,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved global SHAP summary to {save_path}")
    return shap_values


def plot_dependence(explainer, shap_values, X_transformed, feature_names, feature, save_path=None):
    plt.figure()
    shap.dependence_plot(
        feature, shap_values, X_transformed,
        feature_names=feature_names,
        show=False,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved dependence plot for {feature} to {save_path}")
    plt.close()


def explain_single_prediction(explainer, shap_values, X_transformed, feature_names, row_idx, expected_value):
   
    shap.force_plot(
        expected_value,
        shap_values[row_idx],
        X_transformed[row_idx],
        feature_names=feature_names,
        matplotlib=True,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(f"assets/force_plot_row{row_idx}.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    df = load_raw_data()
    df = engineer_features(df)
    X, y = get_model_frame(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = joblib.load("models/xgb_churn_model.pkl")

    explainer, feature_names, X_test_transformed = build_explainer(model, X_test)

    shap_values = plot_global_summary(explainer, X_test_transformed, feature_names)

    for feat in ["overstrain_ratio", "temp_diff", "power_w"]:
        plot_dependence(
            explainer, shap_values, X_test_transformed, feature_names,
            feature=feat, save_path=f"assets/dependence_{feat}.png"
        )

    
    failure_idx = np.where(y_test.values == 1)[0]
    if len(failure_idx) > 0:
        idx = failure_idx[0]
        explain_single_prediction(
            explainer, shap_values, X_test_transformed, feature_names,
            row_idx=idx, expected_value=explainer.expected_value
        )
        print(f"Saved local force plot for test row {idx} (a true failure case)")