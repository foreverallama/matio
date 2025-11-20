import os
import tempfile

import numpy as np
import pytest

from matio import load_from_mat, save_to_mat
from matio.utils.matclass import MatConvertWarning

files = [("test_time_v7.mat", "v7"), ("test_time_v73.mat", "v7.3")]


@pytest.mark.parametrize("filename, version", files)
class TestLoadMatlabDatetime:

    def test_datetime_scalar(self, filename, version):
        """Test reading datetime scalar data from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["dt_basic"])
        assert "dt_basic" in mdict

        dt_scalar = np.array([["2025-04-01T12:00:00"]], dtype="datetime64[ns]").reshape(
            1, 1
        )

        np.testing.assert_array_equal(mdict["dt_basic"], dt_scalar, strict=True)

    def test_datetime_array(self, filename, version):
        """Test reading datetime array data from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["dt_array"])
        assert "dt_array" in mdict

        dt_array = np.array(
            [
                [
                    "2025-04-01",
                    "2025-04-03",
                    "2025-04-05",
                    "2025-04-02",
                    "2025-04-04",
                    "2025-04-06",
                ]
            ],
            dtype="datetime64[ns]",
        ).reshape(2, 3)

        np.testing.assert_array_equal(mdict["dt_array"], dt_array, strict=True)

    def test_datetime_empty(self, filename, version):
        """Test reading empty datetime data from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["dt_empty"])
        assert "dt_empty" in mdict

        dt_empty = np.empty((0, 0), dtype="datetime64[ns]")
        np.testing.assert_array_equal(mdict["dt_empty"], dt_empty, strict=True)

    def test_datetime_fmt(self, filename, version):
        """Test reading datetime with format data from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        with pytest.warns(MatConvertWarning, match="Ignoring 'fmt' property"):
            mdict = load_from_mat(file_path, variable_names=["dt_fmt"])
            assert "dt_fmt" in mdict

            dt_fmt = np.array(
                [["2025-04-01T12:00:00"]], dtype="datetime64[ns]"
            ).reshape(1, 1)

            np.testing.assert_array_equal(mdict["dt_fmt"], dt_fmt, strict=True)

    def test_datetime_tz(self, filename, version):
        """Test reading datetime with timezone data from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        with pytest.warns(MatConvertWarning, match="does not support time zones"):
            mdict = load_from_mat(file_path, variable_names=["dt_tz"])
            assert "dt_tz" in mdict

            dt_tz = np.array([["2025-04-01T16:00:00"]], dtype="datetime64[ns]").reshape(
                1, 1
            )

            np.testing.assert_array_equal(mdict["dt_tz"], dt_tz, strict=True)


@pytest.mark.parametrize("filename, version", files)
class TestSaveMatlabDatetime:

    def test_datetime_scalar(self, filename, version):
        """Test writing datetime scalar to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["dt_basic"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["dt_basic"])

            np.testing.assert_array_equal(
                mdict["dt_basic"], mload["dt_basic"], strict=True
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_datetime_array(self, filename, version):
        """Test writing datetime array to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["dt_array"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["dt_array"])

            np.testing.assert_array_equal(
                mdict["dt_array"], mload["dt_array"], strict=True
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_datetime_empty(self, filename, version):
        """Test writing empty datetime to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["dt_empty"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["dt_empty"])

            np.testing.assert_array_equal(
                mdict["dt_empty"], mload["dt_empty"], strict=True
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
