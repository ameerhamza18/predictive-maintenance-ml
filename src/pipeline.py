from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

CATEGORICAL_COLS = ["type", "tool_wear_bucket"]
NUMERICAL_COLS = [
    "air_temp", "process_temp", "rot_speed", "torque",
    "tool_wear", "temp_diff", "power_w", "wear_torque_product",
    "overstrain_ratio",
]


def build_preprocessor():
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
            ("num", "passthrough", NUMERICAL_COLS)
        ]
    )
    return preprocessor

def split_data(X,y, test_size=0.2, random_stat=42):
    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_stat
    )