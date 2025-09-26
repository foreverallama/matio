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

        str_scalar = np.array([["Hello"]], dtype=np.dtypes.StringDType())
        np.testing.assert_array_equal(mdict["string_scalar"], str_scalar, strict=True)

    def test_string_array(self, filename, version):
        """Test reading string array data from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["string_array"])
        assert "string_array" in mdict

        str_array = np.array(
            ["Apple", "Banana", "Cherry", "Date", "Fig", "Grapes"],
            dtype=np.dtypes.StringDType(),
        ).reshape(2, 3)
        np.testing.assert_array_equal(mdict["string_array"], str_array, strict=True)

    def test_string_empty(self, filename, version):
        """Test reading empty string data from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["string_empty"])
        assert "string_empty" in mdict

        str_empty = np.array([[""]], dtype=np.dtypes.StringDType())
        np.testing.assert_array_equal(mdict["string_empty"], str_empty, strict=True)
