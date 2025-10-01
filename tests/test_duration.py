import os
import tempfile

import numpy as np
import pytest

from matio import load_from_mat, save_to_mat
from matio.utils.matclass import MatConvertWarning

files = [("test_time_v7.mat", "v7"), ("test_time_v73.mat", "v7.3")]

# Notes
# During load, fallback subdtype is 'ms'
# However, during save, fallback is 's' as required by MATLAB


@pytest.mark.parametrize("filename, version", files)
class TestLoadMatlabDuration:

    def test_duration_seconds(self, filename, version):
        """Test reading duration seconds from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["dur_s"])
        assert "dur_s" in mdict

        dur_seconds = np.array([5], dtype="timedelta64[s]").reshape(1, 1)

        np.testing.assert_array_equal(mdict["dur_s"], dur_seconds, strict=True)

    def test_duration_minutes(self, filename, version):
        """Test reading duration minutes data from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["dur_m"])
        assert "dur_m" in mdict

        dur_minutes = np.array([5], dtype="timedelta64[m]").reshape(1, 1)

        np.testing.assert_array_equal(mdict["dur_m"], dur_minutes, strict=True)

    def test_duration_hours(self, filename, version):
        """Test reading duration hours from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["dur_h"])
        assert "dur_h" in mdict

        dur_hours = np.array([5], dtype="timedelta64[h]").reshape(1, 1)

        np.testing.assert_array_equal(mdict["dur_h"], dur_hours, strict=True)

    def test_duration_days(self, filename, version):
        """Test reading duration days from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["dur_days"])
        assert "dur_days" in mdict

        dur_days = np.array([5], dtype="timedelta64[D]").reshape(1, 1)

        np.testing.assert_array_equal(mdict["dur_days"], dur_days, strict=True)

    def test_duration_years(self, filename, version):
        """Test reading duration years from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["dur_years"])
        assert "dur_years" in mdict

        dur_years = np.array([1, 2, 3], dtype="timedelta64[Y]").reshape(1, 3)

        np.testing.assert_array_equal(mdict["dur_years"], dur_years, strict=True)

    def test_duration_hms(self, filename, version):
        """Test reading duration hh:mm:ss from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        with pytest.warns(MatConvertWarning, match="mat_to_duration: Unknown format"):
            mdict = load_from_mat(file_path, variable_names=["dur_hms"])
            assert "dur_hms" in mdict

            dur_vec = (
                (
                    np.timedelta64(1, "h")
                    + np.timedelta64(2, "m")
                    + np.timedelta64(3, "s")
                )
                .astype("timedelta64[ms]")
                .reshape(1, 1)
            )
            np.testing.assert_array_equal(mdict["dur_hms"], dur_vec, strict=True)

    def test_duration_array(self, filename, version):
        """Test reading duration array from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["dur_array"])
        assert "dur_array" in mdict

        dur_array = np.array([10, 20, 30, 40, 50, 60], dtype="timedelta64[s]").reshape(
            2, 3
        )
        np.testing.assert_array_equal(mdict["dur_array"], dur_array, strict=True)

    def test_duration_empty(self, filename, version):
        """Test reading duration empty from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)

        with pytest.warns(MatConvertWarning, match="mat_to_duration: Unknown format"):
            mdict = load_from_mat(file_path, variable_names=["dur_empty"])

        assert "dur_empty" in mdict

        dur_empty = np.empty((0, 0), dtype="timedelta64[ms]")
        np.testing.assert_array_equal(mdict["dur_empty"], dur_empty, strict=True)


@pytest.mark.parametrize("filename, version", files)
class TestSaveMatlabDuration:

    def test_duration_seconds(self, filename, version):
        """Test writing duration seconds to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["dur_s"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["dur_s"])

            np.testing.assert_array_equal(mdict["dur_s"], mload["dur_s"], strict=True)

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_duration_minutes(self, filename, version):
        """Test writing duration minutes to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["dur_m"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)

            mload = load_from_mat(temp_file_path, variable_names=["dur_m"])

            np.testing.assert_array_equal(mdict["dur_m"], mload["dur_m"], strict=True)

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_duration_hours(self, filename, version):
        """Test writing duration hours to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["dur_h"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["dur_h"])

            np.testing.assert_array_equal(mdict["dur_h"], mload["dur_h"], strict=True)

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_duration_days(self, filename, version):
        """Test writing duration days to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["dur_days"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["dur_days"])

            np.testing.assert_array_equal(
                mdict["dur_days"], mload["dur_days"], strict=True
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_duration_years(self, filename, version):
        """Test writing duration years to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["dur_years"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["dur_years"])

            np.testing.assert_array_equal(
                mdict["dur_years"], mload["dur_years"], strict=True
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_duration_array(self, filename, version):
        """Test writing duration array to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["dur_array"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["dur_array"])

            np.testing.assert_array_equal(
                mdict["dur_array"], mload["dur_array"], strict=True
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_duration_empty(self, filename, version):
        """Test writing empty duration to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)

        with pytest.warns(MatConvertWarning, match="mat_to_duration: Unknown format"):
            mdict = load_from_mat(file_path, variable_names=["dur_empty"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            with pytest.warns(MatConvertWarning, match="duration_to_mat"):
                save_to_mat(temp_file_path, mdict, version=version)

            mload = load_from_mat(temp_file_path, variable_names=["dur_empty"])

            np.testing.assert_array_equal(
                mdict["dur_empty"].astype("timedelta64[s]"),
                mload["dur_empty"],
                strict=True,
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_duration_hms(self, filename, version):
        """Test writing duration h:m:s to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        with pytest.warns(MatConvertWarning, match="mat_to_duration: Unknown format"):
            mdict = load_from_mat(file_path, variable_names=["dur_hms"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:

            with pytest.warns(MatConvertWarning, match="duration_to_mat"):
                save_to_mat(temp_file_path, mdict, version=version)

            mload = load_from_mat(temp_file_path, variable_names=["dur_hms"])

            np.testing.assert_array_equal(
                mdict["dur_hms"].astype("timedelta64[s]"), mload["dur_hms"], strict=True
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
