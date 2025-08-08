import os

import numpy as np
import pytest

from matio import load_from_mat

params = [
    (
        {},
        "map1",
    ),
    (
        {
            1: np.array(["a"]),
            2: np.array(["b"]),
        },
        "map2",
    ),
    (
        {
            "a": np.array([[1]]),
            "b": np.array([[2]]),
        },
        "map3",
    ),
    (
        {
            "a": np.array([[1]]),
            "b": np.array([[2]]),
        },
        "map4",
    ),
]


@pytest.mark.parametrize(
    "expected_array, var_name",
    params,
    ids=[
        "map-empty-v7",
        "map-numeric-key-v7",
        "map-char-key-v7",
        "map-string-key-v7",
    ],
)
def test_containermap_load_v7(expected_array, var_name):
    file_path_v7 = os.path.join(os.path.dirname(__file__), "test_map_v7.mat")
    matdict = load_from_mat(file_path_v7, raw_data=False)

    assert var_name in matdict
    for key, val in matdict[var_name].items():
        assert key in expected_array
        assert np.array_equal(val, expected_array[key])


@pytest.mark.parametrize(
    "expected_array, var_name",
    params,
    ids=[
        "map-empty-v7.3",
        "map-numeric-key-v7.3",
        "map-char-key-v7.3",
        "map-string-key-v7.3",
    ],
)
def test_containermap_load_v73(expected_array, var_name):
    file_path_v73 = os.path.join(os.path.dirname(__file__), "test_map_v73.mat")
    matdict = load_from_mat(file_path_v73, raw_data=False)

    assert var_name in matdict
    for key, val in matdict[var_name].items():
        assert key in expected_array
        assert np.array_equal(val, expected_array[key])
