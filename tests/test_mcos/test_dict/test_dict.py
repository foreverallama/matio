import os

import numpy as np
import pytest

from matio import load_from_mat

params = [
    (
        (
            np.array([[1.0, 2.0, 3.0]]).reshape(-1, 1),
            np.array(["apple", "banana", "cherry"]).reshape(-1, 1),
        ),
        "dict1",
    ),
    (
        (
            np.array(["x", "y", "z"]).reshape(-1, 1),
            np.array([10.0, 20.0, 30.0]).reshape(-1, 1),
        ),
        "dict2",
    ),
    (
        (
            np.array([["name", "age"]]).reshape(-1, 1),
            np.array([np.array(["Alice"]), np.array([25.0])], dtype=object).reshape(
                -1, 1
            ),
        ),
        "dict3",
    ),
    (
        (
            np.array([[1, 2, 3]]).reshape(-1, 1),
            np.array(["one", "two", "three"]).reshape(-1, 1),
        ),
        "dict4",
    ),
]


@pytest.mark.parametrize(
    "expected_array, var_name",
    params,
    ids=[
        "dict-numeric-key-v7",
        "dict-char-key-v7",
        "dict-mixed-val-v7",
        "dict-cell-key-v7",
    ],
)
def test_dict_load_v7(expected_array, var_name):
    file_path_v7 = os.path.join(os.path.dirname(__file__), "test_dict_v7.mat")
    matdict = load_from_mat(file_path_v7, raw_data=False)

    assert var_name in matdict
    assert np.array_equal(matdict[var_name][0], expected_array[0])
    assert np.array_equal(matdict[var_name][1], expected_array[1])


@pytest.mark.parametrize(
    "expected_array, var_name",
    params,
    ids=[
        "dict-numeric-key-v7.3",
        "dict-char-key-v7.3",
        "dict-mixed-val-v7.3",
        "dict-cell-key-v7.3",
    ],
)
def test_containermap_load_v73(expected_array, var_name):
    file_path_v73 = os.path.join(os.path.dirname(__file__), "test_dict_v73.mat")
    matdict = load_from_mat(file_path_v73, raw_data=False)

    assert var_name in matdict
    assert np.array_equal(matdict[var_name][0], expected_array[0])
    assert np.array_equal(matdict[var_name][1], expected_array[1])
