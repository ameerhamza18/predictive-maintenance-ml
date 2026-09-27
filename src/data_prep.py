import os
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo

RENAME_MAP = {
    "UDI": "uid",
    "Product ID": "product_id",
    "Type": "type",
    "Air temperature [K]": "air_temp",
    "Process temperature [K]": "process_temp",
    "Rotational speed [rpm]": "rot_speed",
    "Torque [Nm]": "torque",
    "Tool wear [min]": "tool_wear",
    "Machine failure": "machine_failure",
    "TWF": "twf",
    "HDF": "hdf",
    "PWF": "pwf",
    "OSF": "osf",
    "RNF": "rnf",
}

# These directly encode the target . So, they must never be used as model features.
LEAKAGE_COLS = ["twf", "hdf", "pwf", "osf", "rnf"]

# Overstrain failure as discussed in notebook differs by product variant
# for low , meduim , high
OVERSTRAIN_THRESHOLD = {"L": 11000, "M": 12000, "H": 13000}


def load_raw_data(save_path="data/raw/ai4i2020.csv"):
    if os.path.exists(save_path):
        df = pd.read_csv(save_path)
    else:
        dataset = fetch_ucirepo(id=601)
        X = dataset.data.features
        y = dataset.data.targets
        df = pd.concat([X,y], axis=1)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
    df = df.rename(columns=RENAME_MAP)
    return df




def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds domain-driven features from the physical failure-mode definitions,
    instead of relying on XGBoost to rediscover them from raw sensors alone.
    """
    df = df.copy()

    # --- Heat dissipation signal ---
    # HDF triggers when (process_temp - air_temp) < 8.6K AND rot_speed < 1380 rpm.
    # Give the model the exact quantity the rule depends on, directly.
    df["temp_diff"] = df["process_temp"] - df["air_temp"]

    # Power signal(PWF triggers when torque * rotational speed (rad/s) falls outside [3500, 9000] W.
    df["power_w"] = df["torque"] * (df["rot_speed"] * 2 * np.pi / 60)

    # Overstrain signal, adjusted for product variant
    # OSF triggers when tool_wear * torque exceeds a threshold that DEPENDS on type.
    df["wear_torque_product"] = df["tool_wear"] * df["torque"]
    df["overstrain_threshold"] = df["type"].map(OVERSTRAIN_THRESHOLD)
    df["overstrain_ratio"] = df["wear_torque_product"] / df["overstrain_threshold"]

    # Tool wear failure signal :::TWF is randomly assigned in a wear window (200-240 min)
    df["tool_wear_bucket"] = pd.cut(
        df["tool_wear"],
        bins=[-1, 50, 100, 150, 200, 250, 300],
        labels=["0-50", "51-100", "101-150", "151-200", "201-250", "251-300"],
    )

    return df


def get_model_frame(df: pd.DataFrame):

    y = df["machine_failure"]
    drop_cols = ["uid", "product_id", "machine_failure", "overstrain_threshold"] + LEAKAGE_COLS
    X = df.drop(columns=drop_cols)
    return X, y


if __name__ == "__main__":
    df = load_raw_data()
    df = engineer_features(df)
    print(df.shape)
    print(df.dtypes)
    print(df[["temp_diff", "power_w", "overstrain_ratio"]].describe())