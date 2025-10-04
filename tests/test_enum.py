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


@pytest.mark.parametrize("filename, version", files)
class TestSaveMatlabEnum:

    def test_enum_scalar(self, filename, version):
        """Test writing enum scalar to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["enum_scalar"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["enum_scalar"])
            assert "enum_scalar" in mload
            assert isinstance(mload["enum_scalar"], MatlabEnumerationArray)
            assert mload["enum_scalar"].classname == mdict["enum_scalar"].classname
            assert mload["enum_scalar"].type_system == mdict["enum_scalar"].type_system
            assert mload["enum_scalar"].shape == mdict["enum_scalar"].shape
            assert mload["enum_scalar"].dtype == mdict["enum_scalar"].dtype

            assert mload["enum_scalar"][0, 0].name == mdict["enum_scalar"][0, 0].name
            for key in mdict["enum_scalar"][0, 0].value:
                assert key in mload["enum_scalar"][0, 0].value
                np.testing.assert_array_equal(
                    mload["enum_scalar"][0, 0].value[key],
                    mdict["enum_scalar"][0, 0].value[key],
                )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_enum_uint32(self, filename, version):
        """Test writing enum array to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["enum_uint32"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["enum_uint32"])
            assert "enum_uint32" in mload
            assert isinstance(mload["enum_uint32"], MatlabEnumerationArray)
            assert mload["enum_uint32"].classname == mdict["enum_uint32"].classname
            assert mload["enum_uint32"].type_system == mdict["enum_uint32"].type_system
            assert mload["enum_uint32"].shape == mdict["enum_uint32"].shape
            assert mload["enum_uint32"].dtype == mdict["enum_uint32"].dtype

            assert mload["enum_uint32"][0, 0].name == mdict["enum_uint32"][0, 0].name
            for key in mdict["enum_uint32"][0, 0].value:
                assert key in mload["enum_uint32"][0, 0].value
                np.testing.assert_array_equal(
                    mload["enum_uint32"][0, 0].value[key],
                    mdict["enum_uint32"][0, 0].value[key],
                )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_enum_array(self, filename, version):
        """Test writing enum array data to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["enum_array"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["enum_array"])
            assert "enum_array" in mload
            assert isinstance(mload["enum_array"], MatlabEnumerationArray)
            assert mload["enum_array"].classname == mdict["enum_array"].classname
            assert mload["enum_array"].type_system == mdict["enum_array"].type_system
            assert mload["enum_array"].shape == mdict["enum_array"].shape
            assert mload["enum_array"].dtype == mdict["enum_array"].dtype

            for idx in np.ndindex(mdict["enum_array"].shape):
                enum_obj_orig = mdict["enum_array"][idx]
                enum_obj_load = mload["enum_array"][idx]
                assert enum_obj_load.name == enum_obj_orig.name
                for key in enum_obj_orig.value:
                    assert key in enum_obj_load.value
                    np.testing.assert_array_equal(
                        enum_obj_load.value[key],
                        enum_obj_orig.value[key],
                    )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_save_enum_nested(self, filename, version):
        """Test writing nested enum data to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["enum_nested"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["enum_nested"])
            assert "enum_nested" in mload
            assert isinstance(mload["enum_nested"], MatlabOpaque)
            assert mload["enum_nested"].classname == mdict["enum_nested"].classname
            assert mload["enum_nested"].type_system == mdict["enum_nested"].type_system

            assert isinstance(
                mload["enum_nested"].properties["a"], MatlabEnumerationArray
            )
            assert (
                mload["enum_nested"].properties["a"].classname
                == mdict["enum_nested"].properties["a"].classname
            )
            assert (
                mload["enum_nested"].properties["a"].type_system
                == mdict["enum_nested"].properties["a"].type_system
            )
            assert (
                mload["enum_nested"].properties["a"].shape
                == mdict["enum_nested"].properties["a"].shape
            )
            assert (
                mload["enum_nested"].properties["a"].dtype
                == mdict["enum_nested"].properties["a"].dtype
            )
            assert (
                mload["enum_nested"].properties["a"][0, 0].name
                == mdict["enum_nested"].properties["a"][0, 0].name
            )

            assert isinstance(mload["enum_nested"].properties["b"], np.ndarray)
            assert isinstance(
                mload["enum_nested"].properties["b"][0, 0], MatlabEnumerationArray
            )
            assert (
                mload["enum_nested"].properties["b"][0, 0].classname
                == mdict["enum_nested"].properties["b"][0, 0].classname
            )
            assert (
                mload["enum_nested"].properties["b"][0, 0].type_system
                == mdict["enum_nested"].properties["b"][0, 0].type_system
            )
            assert (
                mload["enum_nested"].properties["b"][0, 0].shape
                == mdict["enum_nested"].properties["b"][0, 0].shape
            )
            assert (
                mload["enum_nested"].properties["b"][0, 0].dtype
                == mdict["enum_nested"].properties["b"][0, 0].dtype
            )
            assert (
                mload["enum_nested"].properties["b"][0, 0][0, 0].name
                == mdict["enum_nested"].properties["b"][0, 0][0, 0].name
            )
            assert isinstance(mload["enum_nested"].properties["c"], np.ndarray)
            assert isinstance(
                mload["enum_nested"].properties["c"][0, 0]["InnerProp"],
                MatlabEnumerationArray,
            )
            assert (
                mload["enum_nested"].properties["c"][0, 0]["InnerProp"].classname
                == mdict["enum_nested"].properties["c"][0, 0]["InnerProp"].classname
            )
            assert (
                mload["enum_nested"].properties["c"][0, 0]["InnerProp"].type_system
                == mdict["enum_nested"].properties["c"][0, 0]["InnerProp"].type_system
            )
            assert (
                mload["enum_nested"].properties["c"][0, 0]["InnerProp"].shape
                == mdict["enum_nested"].properties["c"][0, 0]["InnerProp"].shape
            )
            assert (
                mload["enum_nested"].properties["c"][0, 0]["InnerProp"].dtype
                == mdict["enum_nested"].properties["c"][0, 0]["InnerProp"].dtype
            )
            assert (
                mload["enum_nested"].properties["c"][0, 0]["InnerProp"][0, 0].name
                == mdict["enum_nested"].properties["c"][0, 0]["InnerProp"][0, 0].name
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
