import pandas as pd
import pytest

from src.data_prep import engineer_features, OVERSTRAIN_THRESHOLD


@pytest.fixture
def sample_row():
    return pd.DataFrame([{
        "type": "M",
        "air_temp": 300.0,
        "process_temp": 310.0,
        "rot_speed": 1500,
        "torque": 40.0,
        "tool_wear": 100,
    }])


def test_temp_diff_is_correct(sample_row):
    result = engineer_features(sample_row)
    assert result["temp_diff"].iloc[0] == pytest.approx(10.0)


def test_power_w_is_correct(sample_row):
    import numpy as np
    result = engineer_features(sample_row)
    expected = 40.0 * (1500 * 2 * np.pi / 60)
    assert result["power_w"].iloc[0] == pytest.approx(expected)


def test_overstrain_ratio_uses_correct_threshold_per_type(sample_row):
    result = engineer_features(sample_row)
    expected_product = 100 * 40.0
    expected_ratio = expected_product / OVERSTRAIN_THRESHOLD["M"]
    assert result["overstrain_ratio"].iloc[0] == pytest.approx(expected_ratio)


def test_overstrain_ratio_differs_by_type():
    """The same wear/torque values should give a DIFFERENT overstrain_ratio
    depending on product type - this is the whole point of the feature."""
    rows = pd.DataFrame([
        {"type": "L", "air_temp": 300.0, "process_temp": 310.0,
         "rot_speed": 1500, "torque": 40.0, "tool_wear": 100},
        {"type": "H", "air_temp": 300.0, "process_temp": 310.0,
         "rot_speed": 1500, "torque": 40.0, "tool_wear": 100},
    ])
    result = engineer_features(rows)
    ratio_L = result[result["type"] == "L"]["overstrain_ratio"].iloc[0]
    ratio_H = result[result["type"] == "H"]["overstrain_ratio"].iloc[0]
    assert ratio_L > ratio_H  # L has the LOWEST threshold, so same wear*torque -> higher ratio


def test_tool_wear_bucket_assigns_correct_bin(sample_row):
    result = engineer_features(sample_row)
    assert result["tool_wear_bucket"].iloc[0] == "51-100"
