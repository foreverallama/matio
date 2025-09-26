import os
import tempfile

import numpy as np
import pytest

from matio import load_from_mat, save_to_mat
from matio.utils.matclass import MatConvertWarning

files = [("test_time_v7.mat", "v7"), ("test_time_v73.mat", "v7.3")]


@pytest.mark.parametrize("filename, version", files)
class TestLoadMatlabCalendarDuration:

    def test_calendarDuration_days(self, filename, version):
        """Test reading datetime scalar data from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["cdur_days"])
        assert "cdur_days" in mdict

        out_dtype = [
            ("months", "timedelta64[M]"),
            ("days", "timedelta64[D]"),
            ("millis", "timedelta64[ms]"),
        ]

        cdur_days = np.array(
            [
                (0, 1, 0),
                (0, 2, 0),
                (0, 3, 0),
            ],
            dtype=out_dtype,
        ).reshape(1, 3)

        np.testing.assert_array_equal(mdict["cdur_days"], cdur_days, strict=True)

    def test_calendarDuration_weeks(self, filename, version):
        """Test reading datetime scalar data from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["cdur_weeks"])
        assert "cdur_weeks" in mdict

        out_dtype = [
            ("months", "timedelta64[M]"),
            ("days", "timedelta64[D]"),
            ("millis", "timedelta64[ms]"),
        ]

        cdur_weeks = np.array(
            [
                (0, 7, 0),
                (0, 14, 0),
            ],
            dtype=out_dtype,
        ).reshape(1, 2)

        np.testing.assert_array_equal(mdict["cdur_weeks"], cdur_weeks, strict=True)

    def test_calendarDuration_days_and_months(self, filename, version):
        """Test reading datetime scalar data from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["cdur_days_and_months"])
        assert "cdur_days_and_months" in mdict

        out_dtype = [
            ("months", "timedelta64[M]"),
            ("days", "timedelta64[D]"),
            ("millis", "timedelta64[ms]"),
        ]

        cdur_days_and_weeks = np.array([(1, 1, 0), (0, 2, 0)], dtype=out_dtype).reshape(
            1, 2
        )

        np.testing.assert_array_equal(
            mdict["cdur_days_and_months"], cdur_days_and_weeks, strict=True
        )

    def test_calendarDuration_months_and_years(self, filename, version):
        """Test reading datetime scalar data from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["cdur_months_and_years"])
        assert "cdur_months_and_years" in mdict

        out_dtype = [
            ("months", "timedelta64[M]"),
            ("days", "timedelta64[D]"),
            ("millis", "timedelta64[ms]"),
        ]

        cdur_months_and_years = np.array(
            [(12, 0, 0), (18, 0, 0)], dtype=out_dtype
        ).reshape(1, 2)

        np.testing.assert_array_equal(
            mdict["cdur_months_and_years"], cdur_months_and_years, strict=True
        )

    def test_calendarDuration_days_and_quarters(self, filename, version):
        """Test reading datetime scalar data from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["cdur_days_and_qtrs"])
        assert "cdur_days_and_qtrs" in mdict

        out_dtype = [
            ("months", "timedelta64[M]"),
            ("days", "timedelta64[D]"),
            ("millis", "timedelta64[ms]"),
        ]

        cdur_days_and_quarters = np.array(
            [
                (3, 15, 0),
            ],
            dtype=out_dtype,
        ).reshape(1, 1)

        np.testing.assert_array_equal(
            mdict["cdur_days_and_qtrs"], cdur_days_and_quarters, strict=True
        )

    def test_calendarDuration_array(self, filename, version):
        """Test reading datetime array data from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["cdur_array"])
        assert "cdur_array" in mdict

        out_dtype = [
            ("months", "timedelta64[M]"),
            ("days", "timedelta64[D]"),
            ("millis", "timedelta64[ms]"),
        ]

        cdur_array = np.array(
            [
                (1, 0, 0),
                (0, 5, 0),
                (2, 0, 0),
                (0, 10, 0),
            ],
            dtype=out_dtype,
        ).reshape(2, 2)

        np.testing.assert_array_equal(mdict["cdur_array"], cdur_array, strict=True)

    def test_calendarDuration_millis(self, filename, version):
        """Test reading datetime scalar data from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["cdur_millis"])
        assert "cdur_millis" in mdict

        out_dtype = [
            ("months", "timedelta64[M]"),
            ("days", "timedelta64[D]"),
            ("millis", "timedelta64[ms]"),
        ]

        cdur_millis = np.array(
            [
                (0, 1, 3723000),
            ],
            dtype=out_dtype,
        ).reshape(1, 1)

        np.testing.assert_array_equal(mdict["cdur_millis"], cdur_millis, strict=True)

    def test_calendarDuration_empty(self, filename, version):
        """Test reading datetime scalar data from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["cdur_empty"])
        assert "cdur_empty" in mdict

        out_dtype = [
            ("months", "timedelta64[M]"),
            ("days", "timedelta64[D]"),
            ("millis", "timedelta64[ms]"),
        ]

        cdur_empty = np.empty((0, 0), dtype=out_dtype)

        np.testing.assert_array_equal(mdict["cdur_empty"], cdur_empty, strict=True)
