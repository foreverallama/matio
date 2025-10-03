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
        with pytest.warns(MatConvertWarning, match="converted to UTC"):
            mdict = load_from_mat(file_path, variable_names=["dt_tz"])
            assert "dt_tz" in mdict

            dt_tz = np.array([["2025-04-01T12:00:00"]], dtype="datetime64[ns]").reshape(
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


@pytest.mark.parametrize("filename, version", files)
class TestWriteNonSupportedDatetime:

    def test_non_supported_datetime_dates(self, filename, version):
        """Test writing numpy datetime64[date_units] data to MAT-file"""

        dt_years = np.array(["2025", "2026", "2027"], dtype="datetime64[Y]").reshape(
            1, 3
        )
        dt_months = np.array(
            ["2025-01", "2025-02", "2025-03"], dtype="datetime64[M]"
        ).reshape(1, 3)
        dt_weeks = np.array(
            ["2025-01", "2025-02", "2025-03"], dtype="datetime64[W]"
        ).reshape(1, 3)
        dt_days = np.array(
            ["2025-04-01", "2025-04-02", "2025-04-03"], dtype="datetime64[D]"
        ).reshape(1, 3)

        mdict = {
            "dt_years": dt_years,
            "dt_months": dt_months,
            "dt_weeks": dt_weeks,
            "dt_days": dt_days,
        }

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            # with pytest.warns(MatConvertWarning, match="not supported"):
            save_to_mat(temp_file_path, mdict, version=version)

            mload = load_from_mat(temp_file_path, variable_names=None)

            np.testing.assert_array_equal(
                mload["dt_years"],
                mdict["dt_years"].astype("datetime64[ns]"),
                strict=True,
            )
            np.testing.assert_array_equal(
                mload["dt_months"],
                mdict["dt_months"].astype("datetime64[ns]"),
                strict=True,
            )
            np.testing.assert_array_equal(
                mload["dt_weeks"],
                mdict["dt_weeks"].astype("datetime64[ns]"),
                strict=True,
            )
            np.testing.assert_array_equal(
                mload["dt_days"], mdict["dt_days"].astype("datetime64[ns]"), strict=True
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_non_supported_datetime_times(self, filename, version):
        """Test writing numpy datetime64[time_units] data to MAT-file"""

        dt_hours = np.array(
            ["2025-04-01T12", "2025-04-01T13", "2025-04-01T14"], dtype="datetime64[h]"
        ).reshape(1, 3)
        dt_minutes = np.array(
            ["2025-04-01T12:00", "2025-04-01T12:30", "2025-04-01T12:45"],
            dtype="datetime64[m]",
        ).reshape(1, 3)
        dt_seconds = np.array(
            ["2025-04-01T12:00:00", "2025-04-01T12:00:30", "2025-04-01T12:00:45"],
            dtype="datetime64[s]",
        ).reshape(1, 3)
        dt_millis = np.array(
            [
                "2025-04-01T12:00:00.123",
                "2025-04-01T12:00:00.456",
                "2025-04-01T12:00:00.789",
            ],
            dtype="datetime64[ms]",
        ).reshape(1, 3)

        mdict = {
            "dt_hours": dt_hours,
            "dt_minutes": dt_minutes,
            "dt_seconds": dt_seconds,
            "dt_millis": dt_millis,
        }

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            # with pytest.warns(MatConvertWarning, match="not supported"):
            save_to_mat(temp_file_path, mdict, version=version)

            mload = load_from_mat(temp_file_path, variable_names=None)

            np.testing.assert_array_equal(
                mload["dt_hours"],
                mdict["dt_hours"].astype("datetime64[ns]"),
                strict=True,
            )
            np.testing.assert_array_equal(
                mload["dt_minutes"],
                mdict["dt_minutes"].astype("datetime64[ns]"),
                strict=True,
            )
            np.testing.assert_array_equal(
                mload["dt_seconds"],
                mdict["dt_seconds"].astype("datetime64[ns]"),
                strict=True,
            )
            np.testing.assert_array_equal(
                mload["dt_millis"],
                mdict["dt_millis"].astype("datetime64[ns]"),
                strict=True,
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    #! FIXME: Check sub-ms precision in datetime
    # def test_non_supported_datetime_subms(self, filename, version):
    #     """Test writing numpy datetime64[sub-ms] data to MAT-file"""

    #     dt_micros = np.array(
    #         ["2025-04-01T12:00:00.000001", "2025-04-01T12:00:00.000002"], dtype="datetime64[us]"
    #     ).reshape(1, 2)
    #     dt_nanos = np.array(
    #         ["2025-04-01T12:00:00.000000001", "2025-04-01T12:00:00.000000002"],
    #         dtype="datetime64[ns]",
    #     ).reshape(1, 2)
    #     dt_picos = np.array(
    #         ["2025-04-01T12:00:00.000000000001", "2025-04-01T12:00:00.000000000001"], dtype="datetime64[ps]"
    #     ).reshape(1, 2)
    #     dt_femtos = np.array(
    #         ["2025-04-01T12:00:00.000000000000001", "2025-04-01T12:00:00.000000000000001"], dtype="datetime64[fs]"
    #     ).reshape(1, 2)
    #     dt_attos = np.array(
    #         ["2025-04-01T12:00:00.000000000000000001", "2025-04-01T12:00:00.000000000000000001"], dtype="datetime64[as]"
    #     ).reshape(1, 2)

    #     mdict = {"dt_micros": dt_micros, "dt_nanos": dt_nanos, "dt_picos": dt_picos, "dt_femtos": dt_femtos, "dt_attos": dt_attos}

    #     with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
    #         temp_file_path = tmpfile.name

    #     try:
    #         save_to_mat(temp_file_path, mdict, version=version)

    #         mload = load_from_mat(temp_file_path, variable_names=None)

    #         # MATLAB stores sub-ms precision as float ms
    #         # numpy does not support casting subms to ms directly due to precision loss

    #         np.testing.assert_array_equal(
    #             mload["dt_micros"].astype("float64")*1e3, mdict["dt_micros"].astype("float64"), strict=True
    #         )
    #         np.testing.assert_array_equal(
    #             mload["dt_nanos"], mdict["dt_nanos"].astype("datetime64[ms]"), strict=True
    #         )
    #         np.testing.assert_array_equal(
    #             mload["dt_picos"], mdict["dt_picos"].astype("datetime64[ms]"), strict=True
    #         )
    #         np.testing.assert_array_equal(
    #             mload["dt_femtos"], mdict["dt_femtos"].astype("datetime64[ms]"), strict=True
    #         )
    #         np.testing.assert_array_equal(
    #             mload["dt_attos"], mdict["dt_attos"].astype("datetime64[ms]"), strict=True
    #         )

    #     finally:
    #         if os.path.exists(temp_file_path):
    #             os.remove(temp_file_path)
