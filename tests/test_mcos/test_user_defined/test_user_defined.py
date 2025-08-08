import os

import numpy as np
import pytest

from matio import load_from_mat
from matio.mat_opaque_tools import MatioOpaque

params_base = [
    (
        {
            "a": np.array([]).reshape(0, 0),
            "b": np.array([]).reshape(0, 0),
            "c": np.array([]).reshape(0, 0),
        },
        "obj1",
    ),
    (
        {
            "a": np.array([10]).reshape(1, 1),
            "b": np.array([20]).reshape(1, 1),
            "c": np.array([30]).reshape(1, 1),
        },
        "obj2",
    ),
    (
        {
            "a": np.array([]).reshape(0, 0),
            "b": np.array([10]).reshape(1, 1),
            "c": np.array([30]).reshape(1, 1),
        },
        "obj3",
    ),
    (
        np.tile(
            np.array(
                [
                    {
                        "a": np.array([10]).reshape(1, 1),
                        "b": np.array([20]).reshape(1, 1),
                        "c": np.array([30]).reshape(1, 1),
                    }
                ],
                dtype=object,
            ),
            (2, 3),
        ),
        "obj6",
    ),
    (
        {
            "a": np.array(["Default String"]).reshape(1, 1),
            "b": np.array([10]).reshape(1, 1),
            "c": np.array([30]).reshape(1, 1),
        },
        "obj7",
    ),
]


@pytest.mark.parametrize(
    "expected_array, var_name",
    params_base,
    ids=[
        "object-without-constructor-v7",
        "object-with-constructor-v7",
        "object-with-default-v7",
        "object-array-v7",
        "object-in-default-property-v7",
    ],
)
def test_user_defined_load_v7(expected_array, var_name):
    file_path = os.path.join(os.path.dirname(__file__), "test_user_defined_v7.mat")
    matdict = load_from_mat(file_path, raw_data=False)

    assert var_name in matdict
    if isinstance(matdict[var_name], MatioOpaque):
        for key, value in matdict[var_name].properties.items():
            np.testing.assert_array_equal(value, expected_array[key])

    if isinstance(matdict[var_name], np.ndarray):
        np.testing.assert_array_equal(matdict[var_name], expected_array)


@pytest.mark.parametrize(
    "expected_array, var_name",
    params_base,
    ids=[
        "object-without-constructor-v7.3",
        "object-with-constructor-v7.3",
        "object-with-default-v7.3",
        "object-array-v7.3",
        "object-in-default-property-v7.3",
    ],
)
def test_user_defined_load_v73(expected_array, var_name):
    file_path = os.path.join(os.path.dirname(__file__), "test_user_defined_v73.mat")
    matdict = load_from_mat(file_path, raw_data=False)

    assert var_name in matdict
    if isinstance(matdict[var_name], MatioOpaque):
        for key, value in matdict[var_name].properties.items():
            np.testing.assert_array_equal(value, expected_array[key])

    if isinstance(matdict[var_name], np.ndarray):
        np.testing.assert_array_equal(matdict[var_name], expected_array)
