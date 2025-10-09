import os
import tempfile

import numpy as np
import pytest

from matio import load_from_mat, save_to_mat

files = [("test_string_v7.mat", "v7"), ("test_string_v73.mat", "v7.3")]


@pytest.mark.parametrize("filename, version", files)
class TestLoadMatlabString:

    def test_string_scalar(self, filename, version):
        """Test reading string scalar data from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["string_scalar"])
        assert "string_scalar" in mdict

        str_scalar = np.array(
            [["Hello"]], dtype=np.dtypes.StringDType(na_object=np.nan)
        )
        np.testing.assert_array_equal(mdict["string_scalar"], str_scalar, strict=True)

    def test_string_array(self, filename, version):
        """Test reading string array data from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["string_array"])
        assert "string_array" in mdict

        str_array = np.array(
            ["Apple", "Banana", "Cherry", "Date", "Fig", "Grapes"],
            dtype=np.dtypes.StringDType(na_object=np.nan),
        ).reshape(2, 3)
        np.testing.assert_array_equal(mdict["string_array"], str_array, strict=True)

    def test_string_empty(self, filename, version):
        """Test reading empty string data from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["string_empty"])
        assert "string_empty" in mdict

        str_empty = np.array([[""]], dtype=np.dtypes.StringDType(na_object=np.nan))
        np.testing.assert_array_equal(mdict["string_empty"], str_empty, strict=True)


@pytest.mark.parametrize("filename, version", files)
class TestSaveMatlabString:

    def test_string_scalar(self, filename, version):
        """Test writing string scalar to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["string_scalar"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["string_scalar"])

            np.testing.assert_array_equal(
                mdict["string_scalar"], mload["string_scalar"], strict=True
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_string_array(self, filename, version):
        """Test writing string array to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["string_array"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["string_array"])

            np.testing.assert_array_equal(
                mdict["string_array"], mload["string_array"], strict=True
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_string_empty(self, filename, version):
        """Test writing empty string to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["string_empty"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["string_empty"])

            np.testing.assert_array_equal(
                mdict["string_empty"], mload["string_empty"], strict=True
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
