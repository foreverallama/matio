import os
import tempfile

import numpy as np
import pytest

from matio import load_from_mat, save_to_mat
from matio.utils.matclass import MatlabOpaque, MatlabOpaqueArray

files = [("test_user_defined_v7.mat", "v7"), ("test_user_defined_v73.mat", "v7.3")]
namespace = "TestClasses"


@pytest.mark.parametrize("filename, version", files)
class TestLoadMatlabUserDefined:

    def test_obj_no_vals(self, filename, version):
        """Test reading datetime scalar data from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["obj_no_vals"])
        assert "obj_no_vals" in mdict
        assert isinstance(mdict["obj_no_vals"], MatlabOpaque)
        assert mdict["obj_no_vals"].classname == f"{namespace}.BasicClass"
        assert mdict["obj_no_vals"].type_system == "MCOS"

        prop_dict = {
            "a": np.empty((0, 0), dtype=np.float64),
            "b": np.empty((0, 0), dtype=np.float64),
            "c": np.empty((0, 0), dtype=np.float64),
        }

        for key, val in prop_dict.items():
            assert key in mdict["obj_no_vals"].properties
            np.testing.assert_array_equal(
                mdict["obj_no_vals"].properties[key], val, strict=True
            )

    def test_obj_with_vals(self, filename, version):
        """Test reading datetime array data from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["obj_with_vals"])
        assert "obj_with_vals" in mdict
        assert isinstance(mdict["obj_with_vals"], MatlabOpaque)
        assert mdict["obj_with_vals"].classname == f"{namespace}.BasicClass"
        assert mdict["obj_with_vals"].type_system == "MCOS"

        prop_dict = {
            "a": np.array([[10]], dtype=np.float64),
            "b": np.empty((0, 0), dtype=np.float64),
            "c": np.empty((0, 0), dtype=np.float64),
        }

        for key, val in prop_dict.items():
            assert key in mdict["obj_with_vals"].properties
            np.testing.assert_array_equal(
                mdict["obj_with_vals"].properties[key], val, strict=True
            )

    def test_obj_with_default_val(self, filename, version):
        """Test reading datetime with format data from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["obj_with_default_val"])
        assert "obj_with_default_val" in mdict
        assert isinstance(mdict["obj_with_default_val"], MatlabOpaque)
        assert mdict["obj_with_default_val"].classname == f"{namespace}.DefaultClass"
        assert mdict["obj_with_default_val"].type_system == "MCOS"

        prop_dict = {
            "a": np.array([["Default String"]], dtype=np.dtypes.StringDType()),
            "b": np.array([[10]], dtype=np.float64),
        }

        assert "a" in mdict["obj_with_default_val"].properties
        np.testing.assert_array_equal(
            mdict["obj_with_default_val"].properties["a"], prop_dict["a"], strict=True
        )

        assert "b" in mdict["obj_with_default_val"].properties
        np.testing.assert_array_equal(
            mdict["obj_with_default_val"].properties["b"], prop_dict["b"], strict=True
        )

    def test_obj_array(self, filename, version):
        """Test reading datetime with timezone data from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["obj_array"])
        assert "obj_array" in mdict
        assert isinstance(mdict["obj_array"], MatlabOpaqueArray)
        assert mdict["obj_array"].classname == f"{namespace}.BasicClass"
        assert mdict["obj_array"].type_system == "MCOS"
        assert mdict["obj_array"].shape == (2, 2)
        assert mdict["obj_array"].dtype == object
        assert all(isinstance(x, MatlabOpaque) for x in mdict["obj_array"].flat)

        np.testing.assert_array_equal(
            mdict["obj_array"][0, 0].properties["a"],
            np.array([[1]], dtype=np.float64),
            strict=True,
        )
        np.testing.assert_array_equal(
            mdict["obj_array"][0, 1].properties["a"],
            np.array([[2]], dtype=np.float64),
            strict=True,
        )
        np.testing.assert_array_equal(
            mdict["obj_array"][1, 0].properties["a"],
            np.array([[3]], dtype=np.float64),
            strict=True,
        )
        np.testing.assert_array_equal(
            mdict["obj_array"][1, 1].properties["a"],
            np.array([[4]], dtype=np.float64),
            strict=True,
        )

    def test_obj_with_nested_props(self, filename, version):
        """Test reading datetime with timezone data from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["obj_with_nested_props"])
        assert "obj_with_nested_props" in mdict
        assert isinstance(mdict["obj_with_nested_props"], MatlabOpaque)
        assert mdict["obj_with_nested_props"].classname == f"{namespace}.BasicClass"
        assert mdict["obj_with_nested_props"].type_system == "MCOS"

        assert "a" in mdict["obj_with_nested_props"].properties
        assert isinstance(mdict["obj_with_nested_props"].properties["a"], MatlabOpaque)
        assert (
            mdict["obj_with_nested_props"].properties["a"].classname
            == f"{namespace}.BasicClass"
        )
        prop_a = {
            "a": np.array([[1]], dtype=np.float64),
            "b": np.array(["Obj1"], dtype=np.str_),
            "c": np.empty((0, 0), dtype=np.float64),
        }
        for key, val in prop_a.items():
            assert key in mdict["obj_with_nested_props"].properties["a"].properties
            np.testing.assert_array_equal(
                mdict["obj_with_nested_props"].properties["a"].properties[key],
                val,
                strict=True,
            )

        assert "b" in mdict["obj_with_nested_props"].properties
        assert isinstance(mdict["obj_with_nested_props"].properties["b"], np.ndarray)
        assert mdict["obj_with_nested_props"].properties["b"].dtype == object
        assert mdict["obj_with_nested_props"].properties["b"].shape == (1, 1)
        assert isinstance(
            mdict["obj_with_nested_props"].properties["b"][0, 0], MatlabOpaque
        )
        assert (
            mdict["obj_with_nested_props"].properties["b"][0, 0].classname
            == f"{namespace}.BasicClass"
        )
        for key, val in prop_a.items():
            assert (
                key in mdict["obj_with_nested_props"].properties["b"][0, 0].properties
            )
            np.testing.assert_array_equal(
                mdict["obj_with_nested_props"].properties["b"][0, 0].properties[key],
                val,
                strict=True,
            )

        assert "c" in mdict["obj_with_nested_props"].properties
        assert isinstance(mdict["obj_with_nested_props"].properties["c"], np.ndarray)
        assert set(mdict["obj_with_nested_props"].properties["c"].dtype.names) == {
            "InnerProp"
        }
        assert mdict["obj_with_nested_props"].properties["c"].shape == (1, 1)
        assert isinstance(
            mdict["obj_with_nested_props"].properties["c"][0, 0]["InnerProp"],
            MatlabOpaque,
        )
        assert (
            mdict["obj_with_nested_props"].properties["c"][0, 0]["InnerProp"].classname
            == f"{namespace}.BasicClass"
        )
        prop_c = {
            "a": np.array([[2]], dtype=np.float64),
            "b": np.array(["Obj2"], dtype=np.str_),
            "c": np.empty((0, 0), dtype=np.float64),
        }
        for key, val in prop_c.items():
            assert (
                key
                in mdict["obj_with_nested_props"]
                .properties["c"][0, 0]["InnerProp"]
                .properties
            )
            np.testing.assert_array_equal(
                mdict["obj_with_nested_props"]
                .properties["c"][0, 0]["InnerProp"]
                .properties[key],
                val,
                strict=True,
            )

    def test_handle_class_obj(self, filename, version):
        """Test reading datetime with timezone data from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(
            file_path, variable_names=["obj_handle_1", "obj_handle_2"]
        )
        assert "obj_handle_1" in mdict
        assert "obj_handle_2" in mdict
        assert isinstance(mdict["obj_handle_1"], MatlabOpaque)
        assert isinstance(mdict["obj_handle_2"], MatlabOpaque)
        assert mdict["obj_handle_1"] is mdict["obj_handle_2"]

        assert mdict["obj_handle_1"].classname == f"{namespace}.HandleClass"
        assert mdict["obj_handle_1"].type_system == "MCOS"

        val = np.array([[20]], dtype=np.float64)
        np.testing.assert_array_equal(
            mdict["obj_handle_1"].properties["a"], val, strict=True
        )
