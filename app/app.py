import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import gradio as gr

from src.data_prep import engineer_features
from src.explain import get_feature_names

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "xgb_churn_model.pkl")
THRESHOLD = 0.5 

model = joblib.load(MODEL_PATH)
preprocessor = model.named_steps["preprocessor"]
clf = model.named_steps["clf"]
explainer = shap.TreeExplainer(clf)
feature_names = get_feature_names(preprocessor)


def build_input_row(machine_type, air_temp, process_temp, rot_speed, torque, tool_wear):
    """Takes raw UI inputs, applies the SAME feature engineering used in
    training, so the app can never drift out of sync with the model."""
    raw = pd.DataFrame([{
        "type": machine_type,
        "air_temp": air_temp,
        "process_temp": process_temp,
        "rot_speed": rot_speed,
        "torque": torque,
        "tool_wear": tool_wear,
    }])
    engineered = engineer_features(raw)
    engineered = engineered.drop(columns=["overstrain_threshold"])
    return engineered


def predict_and_explain(machine_type, air_temp, process_temp, rot_speed, torque, tool_wear):
    X_row = build_input_row(machine_type, air_temp, process_temp, rot_speed, torque, tool_wear)

    proba = model.predict_proba(X_row)[0, 1]
    label = "⚠️ FAILURE RISK" if proba >= THRESHOLD else "✅ Normal operation"

    X_transformed = preprocessor.transform(X_row)
    shap_values_row = explainer.shap_values(X_transformed)

    exp = shap.Explanation(
        values=shap_values_row[0],
        base_values=explainer.expected_value,
        data=X_transformed[0],
        feature_names=feature_names,
    )

    plt.figure()
    shap.plots.waterfall(exp, show=False, max_display=10)
    fig = plt.gcf()
    fig.tight_layout()

    result_text = f"{label}\n\nFailure probability: {proba:.1%}"
    return result_text, fig


with gr.Blocks(title="Predictive Maintenance: Machine Failure Risk") as demo:
    gr.Markdown(
        "# 🏭 Predictive Maintenance: Machine Failure Risk\n"
        "Enter live sensor readings to get a failure-risk prediction, "
        "explained via SHAP - showing exactly which readings drive the score, "
        "not just a black-box number."
    )

    with gr.Row():
        with gr.Column():
            machine_type = gr.Radio(["L", "M", "H"], value="M", label="Product Quality Variant")
            air_temp = gr.Slider(295.0, 305.0, value=300.0, label="Air Temperature (K)")
            process_temp = gr.Slider(305.0, 315.0, value=310.0, label="Process Temperature (K)")
            rot_speed = gr.Slider(1150, 2900, value=1500, label="Rotational Speed (rpm)")
            torque = gr.Slider(3.0, 77.0, value=40.0, label="Torque (Nm)")
            tool_wear = gr.Slider(0, 253, value=100, label="Tool Wear (min)")
            predict_btn = gr.Button("Predict", variant="primary")

        with gr.Column():
            result_output = gr.Textbox(label="Prediction", lines=3)
            shap_output = gr.Plot(label="Why? (SHAP explanation)")

    predict_btn.click(
        predict_and_explain,
        inputs=[machine_type, air_temp, process_temp, rot_speed, torque, tool_wear],
        outputs=[result_output, shap_output],
    )

    gr.Examples(
        label="Try example scenarios",
        examples=[
            ["L", 298.1, 308.6, 1551, 42.8, 0],     # normal, fresh tool
            ["L", 300.9, 311.2, 1350, 65.0, 220],   # high wear + high torque -> overstrain risk
            ["M", 300.5, 302.0, 1400, 30.0, 100],   # small temp_diff -> heat dissipation risk
        ],
        inputs=[machine_type, air_temp, process_temp, rot_speed, torque, tool_wear],
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)