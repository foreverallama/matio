import os
import tempfile

import numpy as np
import pytest

from matio import load_from_mat, save_to_mat
from matio.utils.matclass import MatlabEnumerationArray, MatlabOpaque

files = [("test_enum_v7.mat", "v7"), ("test_enum_v73.mat", "v7.3")]
namespace = "TestClasses"


@pytest.mark.parametrize("filename, version", files)
class TestLoadMatlabEnum:

    def test_enum_scalar(self, filename, version):
        """Test reading enum scalar from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["enum_scalar"])
        assert "enum_scalar" in mdict
        assert isinstance(mdict["enum_scalar"], MatlabEnumerationArray)
        assert mdict["enum_scalar"].classname == f"{namespace}.EnumClass"
        assert mdict["enum_scalar"].type_system == "MCOS"
        assert mdict["enum_scalar"].shape == (1, 1)
        assert mdict["enum_scalar"].dtype == object

        assert mdict["enum_scalar"][0, 0].name == "enum1"
        val_dict = {
            "val": np.array([[1]], dtype=np.float64),
        }
        for key, val in val_dict.items():
            assert key in mdict["enum_scalar"][0, 0].value
            np.testing.assert_array_equal(mdict["enum_scalar"][0, 0].value[key], val)

    def test_enum_uint32(self, filename, version):
        """Test reading enum array from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["enum_uint32"])
        assert "enum_uint32" in mdict
        assert isinstance(mdict["enum_uint32"], MatlabEnumerationArray)
        assert mdict["enum_uint32"].classname == f"{namespace}.EnumClassWithBase"
        assert mdict["enum_uint32"].type_system == "MCOS"
        assert mdict["enum_uint32"].shape == (1, 1)
        assert mdict["enum_uint32"].dtype == object

        assert mdict["enum_uint32"][0, 0].name == "enum1"
        val_dict = {
            "uint32.Data": np.array([[1]], dtype=np.uint32),
        }
        for key, val in val_dict.items():
            assert key in mdict["enum_uint32"][0, 0].value
            np.testing.assert_array_equal(mdict["enum_uint32"][0, 0].value[key], val)

    def test_enum_array(self, filename, version):
        """Test reading datetime array data from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["enum_array"])
        assert "enum_array" in mdict
        assert isinstance(mdict["enum_array"], MatlabEnumerationArray)
        assert mdict["enum_array"].classname == f"{namespace}.EnumClass"
        assert mdict["enum_array"].type_system == "MCOS"
        assert mdict["enum_array"].shape == (2, 3)
        assert mdict["enum_array"].dtype == object

        expected_names = np.array(
            [["enum1", "enum3", "enum5"], ["enum2", "enum4", "enum6"]], dtype=np.str_
        ).reshape((2, 3), order="F")

        expected_vals = np.array([[1, 3, 5], [2, 4, 6]], dtype=np.float64).reshape(
            (2, 3), order="F"
        )

        for idx in np.ndindex(mdict["enum_array"].shape):
            enum_obj = mdict["enum_array"][idx]
            assert enum_obj.name == expected_names[idx]
            assert "val" in enum_obj.value
            np.testing.assert_array_equal(
                enum_obj.value["val"],
                np.array([[expected_vals[idx]]], dtype=np.float64),
            )

    def test_load_enum_nested(self, filename, version):
        """Test reading nested enum data from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["enum_nested"])
        assert "enum_nested" in mdict
        assert isinstance(mdict["enum_nested"], MatlabOpaque)
        assert mdict["enum_nested"].classname == f"{namespace}.BasicClass"
        assert mdict["enum_nested"].type_system == "MCOS"

        assert isinstance(mdict["enum_nested"].properties["a"], MatlabEnumerationArray)
        assert (
            mdict["enum_nested"].properties["a"].classname == f"{namespace}.EnumClass"
        )
        assert mdict["enum_nested"].properties["a"].type_system == "MCOS"
        assert mdict["enum_nested"].properties["a"].shape == (1, 1)
        assert mdict["enum_nested"].properties["a"].dtype == object
        assert mdict["enum_nested"].properties["a"][0, 0].name == "enum1"

        assert isinstance(mdict["enum_nested"].properties["b"], np.ndarray)
        assert isinstance(
            mdict["enum_nested"].properties["b"][0, 0], MatlabEnumerationArray
        )
        assert (
            mdict["enum_nested"].properties["b"][0, 0].classname
            == f"{namespace}.EnumClass"
        )
        assert mdict["enum_nested"].properties["b"][0, 0].type_system == "MCOS"
        assert mdict["enum_nested"].properties["b"][0, 0].shape == (1, 1)
        assert mdict["enum_nested"].properties["b"][0, 0].dtype == object
        assert mdict["enum_nested"].properties["b"][0, 0][0, 0].name == "enum2"

        assert isinstance(mdict["enum_nested"].properties["c"], np.ndarray)
        assert isinstance(
            mdict["enum_nested"].properties["c"][0, 0]["InnerProp"],
            MatlabEnumerationArray,
        )
        assert (
            mdict["enum_nested"].properties["c"][0, 0]["InnerProp"].classname
            == f"{namespace}.EnumClass"
        )
        assert (
            mdict["enum_nested"].properties["c"][0, 0]["InnerProp"].type_system
            == "MCOS"
        )
        assert mdict["enum_nested"].properties["c"][0, 0]["InnerProp"].shape == (1, 1)
        assert mdict["enum_nested"].properties["c"][0, 0]["InnerProp"].dtype == object
        assert (
            mdict["enum_nested"].properties["c"][0, 0]["InnerProp"][0, 0].name
            == "enum3"
        )
