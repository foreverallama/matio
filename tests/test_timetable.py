import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from matio import load_from_mat, save_to_mat
from matio.utils.converters.mattables import get_row_times
from matio.utils.matclass import MatConvertWarning, MatlabOpaque

files = [("test_tables_v7.mat", "v7"), ("test_tables_v73.mat", "v7.3")]
namespace = "TestClasses"


@pytest.mark.parametrize("filename, version", files)
class TestLoadMatlabTimetable:

    def test_timetable_datetime(self, filename, version):
        """Test reading timetable with datetime index from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["timetable_datetime"])
        assert "timetable_datetime" in mdict

        df = pd.DataFrame(
            {"data1": [1.0, 2.0, 3.0]},
            index=pd.Index(
                np.array(
                    ["2023-01-01", "2023-01-02", "2023-01-03"],
                    dtype="datetime64[ns]",
                ),
                name="Time",
            ),
        )
        pd.testing.assert_frame_equal(mdict["timetable_datetime"], df, check_like=True)

    def test_timetable_duration(self, filename, version):
        """Test reading timetable with duration index from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["timetable_duration"])
        assert "timetable_duration" in mdict

        df = pd.DataFrame(
            {
                "data1": [1.0, 2.0, 3.0],
            },
            index=pd.Index(
                np.array([10, 20, 30], dtype="timedelta64[s]"),
                name="Time",
            ),
        )
        pd.testing.assert_frame_equal(mdict["timetable_duration"], df, check_like=True)

    def test_timetable_empty(self, filename, version):
        """Test reading empty timetable from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["timetable_empty"])
        assert "timetable_empty" in mdict

        df = pd.DataFrame(
            index=pd.DatetimeIndex([], dtype="datetime64[ns]", name="Time")
        )
        pd.testing.assert_frame_equal(mdict["timetable_empty"], df, check_like=True)

    def test_timetable_from_duration(self, filename, version):
        """Test reading timetable with duration starttime from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        with pytest.warns(MatConvertWarning):
            mdict = load_from_mat(file_path, variable_names=["timetable_from_duration"])
        assert "timetable_from_duration" in mdict

        df = pd.DataFrame(
            {
                "data1": [1.0, 2.0, 3.0],
            },
            index=pd.Index(
                np.array([0, 1, 2], dtype="timedelta64[s]"),
                name="Time",
            ),
        )
        pd.testing.assert_frame_equal(
            mdict["timetable_from_duration"], df, check_like=True
        )

    def test_timetable_from_sample_rate(self, filename, version):
        """Test reading timetable with sample rate and timestep from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        with pytest.warns(MatConvertWarning, match="get_row_times"):
            mdict = load_from_mat(
                file_path, variable_names=["timetable_from_sample_rate"]
            )
        assert "timetable_from_sample_rate" in mdict

        df = pd.DataFrame(
            {
                "data1": [1.0, 2.0, 3.0],
            },
            index=pd.Index(
                np.array([0, 100000, 200000], dtype="timedelta64[ns]"),
                name="Time",
            ),
        )
        pd.testing.assert_frame_equal(
            mdict["timetable_from_sample_rate"], df, check_like=True
        )

    def test_timetable_from_starttime_calendarDuration(self, filename, version):
        """Test reading timetable with calendarDuration starttime in MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(
            file_path, variable_names=["timetable_from_starttime_calendarDuration"]
        )
        assert "timetable_from_starttime_calendarDuration" in mdict

        df = pd.DataFrame(
            {
                "data1": [1.0, 2.0, 3.0],
            },
            index=pd.Index(
                np.array(["2020-01", "2020-04", "2020-07"], dtype="datetime64[ns]"),
                name="Time",
            ),
        )
        pd.testing.assert_frame_equal(
            mdict["timetable_from_starttime_calendarDuration"], df, check_like=True
        )

    def test_timetable_from_starttime_datetime(self, filename, version):
        """Test reading timetable with datetime starttime in MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(
            file_path, variable_names=["timetable_from_starttime_datetime"]
        )
        assert "timetable_from_starttime_datetime" in mdict

        df = pd.DataFrame(
            {
                "data1": [1.0, 2.0, 3.0],
            },
            index=pd.Index(
                np.array(
                    [
                        "2020-01-01T00:00:00",
                        "2020-01-01T00:00:01",
                        "2020-01-01T00:00:02",
                    ],
                    dtype="datetime64[ns]",
                ),
                name="Time",
            ),
        )
        pd.testing.assert_frame_equal(
            mdict["timetable_from_starttime_datetime"], df, check_like=True
        )

    def test_timetable_datetime_unit(self, filename, version):
        """Test reading timetable row times with datetime_unit"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(
            file_path,
            variable_names=["timetable_from_starttime_datetime"],
            datetime_unit="ms",
        )
        times = mdict["timetable_from_starttime_datetime"].index.to_numpy()

        expected = np.array(
            ["2020-01-01T00:00:00", "2020-01-01T00:00:01", "2020-01-01T00:00:02"],
            dtype="datetime64[ms]",
        )
        np.testing.assert_array_equal(times, expected, strict=True)

    def test_timetable_from_starttime_duration(self, filename, version):
        """Test reading timetable with duration starttime in MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(
            file_path, variable_names=["timetable_from_starttime_duration"]
        )
        assert "timetable_from_starttime_duration" in mdict

        df = pd.DataFrame(
            {
                "data1": [1.0, 2.0, 3.0],
            },
            index=pd.Index(
                np.array([10, 11, 12], dtype="timedelta64[s]"),
                name="Time",
            ),
        )
        pd.testing.assert_frame_equal(
            mdict["timetable_from_starttime_duration"], df, check_like=True
        )

    def test_timetable_var_names(self, filename, version):
        """Test reading timetable with variable names from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["timetable_var_names"])
        assert "timetable_var_names" in mdict

        df = pd.DataFrame(
            {"Pressure": [1.0, 2.0, 3.0]},
            index=pd.Index(
                np.array(
                    ["2023-01-01", "2023-01-02", "2023-01-03"],
                    dtype="datetime64[ns]",
                ),
                name="Time",
            ),
        )
        pd.testing.assert_frame_equal(mdict["timetable_var_names"], df, check_like=True)

    def test_timetable_multi_col(self, filename, version):
        """Test reading timetable with multi-column data from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["timetable_multi_col"])
        assert "timetable_multi_col" in mdict

        df = pd.DataFrame(
            {
                "data2_1": [1.0, 2.0, 3.0],
                "data2_2": [4.0, 5.0, 6.0],
            },
            index=pd.Index(
                np.array([10, 20, 30], dtype="timedelta64[s]"),
                name="Time",
            ),
        )
        pd.testing.assert_frame_equal(mdict["timetable_multi_col"], df, check_like=True)

    def test_timetable_with_attrs(self, filename, version):
        """Test reading timetable with attributes from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(
            file_path, variable_names=["timetable_with_attrs"], add_table_attrs=True
        )
        assert "timetable_with_attrs" in mdict

        df = pd.DataFrame(
            {"data1": [1.0, 2.0, 3.0]},
            index=pd.Index(
                np.array(
                    ["2023-01-01", "2023-01-02", "2023-01-03"],
                    dtype="datetime64[ns]",
                ),
                name="Date",
            ),
        )
        pd.testing.assert_frame_equal(
            mdict["timetable_with_attrs"], df, check_like=True
        )

        attrs = {
            "Description": "Random Description",
            "varUnits": ["m/s"],
            "varDescriptions": ["myVar"],
            "varContinuity": ["continuous"],
            "UserData": np.empty((0, 0), dtype=np.float64),
        }

        for key, value in mdict["timetable_with_attrs"].attrs.items():
            assert key in attrs
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, attrs[key])
            else:
                assert value == attrs[key]


@pytest.mark.parametrize("filename, version", files)
class TestSaveMatlabTimetable:

    def test_timetable_datetime(self, filename, version):
        """Test writing numeric timetable to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["timetable_datetime"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["timetable_datetime"])

            pd.testing.assert_frame_equal(
                mdict["timetable_datetime"],
                mload["timetable_datetime"],
                check_like=True,
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_timetable_duration(self, filename, version):
        """Test writing numeric timetable to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["timetable_duration"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["timetable_duration"])

            pd.testing.assert_frame_equal(
                mdict["timetable_duration"],
                mload["timetable_duration"],
                check_like=True,
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_timetable_empty(self, filename, version):
        """Test writing numeric timetable to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["timetable_empty"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["timetable_empty"])

            pd.testing.assert_frame_equal(
                mdict["timetable_empty"], mload["timetable_empty"], check_like=True
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_timetable_from_duration(self, filename, version):
        """Test writing numeric timetable to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        with pytest.warns(MatConvertWarning):
            mdict = load_from_mat(file_path, variable_names=["timetable_from_duration"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(
                temp_file_path, variable_names=["timetable_from_duration"]
            )

            pd.testing.assert_frame_equal(
                mdict["timetable_from_duration"],
                mload["timetable_from_duration"],
                check_like=True,
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_timetable_from_starttime_calendarDuration(self, filename, version):
        """Test writing numeric timetable to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(
            file_path, variable_names=["timetable_from_starttime_calendarDuration"]
        )

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(
                temp_file_path,
                variable_names=["timetable_from_starttime_calendarDuration"],
            )

            pd.testing.assert_frame_equal(
                mdict["timetable_from_starttime_calendarDuration"],
                mload["timetable_from_starttime_calendarDuration"],
                check_like=True,
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_timetable_from_starttime_datetime(self, filename, version):
        """Test writing numeric timetable to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(
            file_path, variable_names=["timetable_from_starttime_datetime"]
        )

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(
                temp_file_path, variable_names=["timetable_from_starttime_datetime"]
            )

            pd.testing.assert_frame_equal(
                mdict["timetable_from_starttime_datetime"],
                mload["timetable_from_starttime_datetime"],
                check_like=True,
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_timetable_from_starttime_duration(self, filename, version):
        """Test writing numeric timetable to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(
            file_path, variable_names=["timetable_from_starttime_duration"]
        )

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(
                temp_file_path, variable_names=["timetable_from_starttime_duration"]
            )

            pd.testing.assert_frame_equal(
                mdict["timetable_from_starttime_duration"],
                mload["timetable_from_starttime_duration"],
                check_like=True,
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_timetable_var_names(self, filename, version):
        """Test writing numeric timetable to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["timetable_var_names"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(
                temp_file_path, variable_names=["timetable_var_names"]
            )

            pd.testing.assert_frame_equal(
                mdict["timetable_var_names"],
                mload["timetable_var_names"],
                check_like=True,
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_timetable_multi_col(self, filename, version):
        """Test writing numeric timetable to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["timetable_multi_col"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(
                temp_file_path, variable_names=["timetable_multi_col"]
            )

            pd.testing.assert_frame_equal(
                mdict["timetable_multi_col"],
                mload["timetable_multi_col"],
                check_like=True,
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_timetable_datetime_tz(self, filename, version):
        """Test writing timetable with timezone-aware datetime index to MAT-file"""
        index = pd.DatetimeIndex(
            ["2024-01-01 01:00", "2024-01-01 01:01"], name="Time"
        ).tz_localize("Europe/Berlin")
        mdict = {"timetable_tz": pd.DataFrame({"x": [1.0, 2.0]}, index=index)}

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            with pytest.warns(MatConvertWarning, match="converted to UTC"):
                save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["timetable_tz"])

            expected = pd.DatetimeIndex(
                ["2024-01-01 00:00", "2024-01-01 00:01"],
                dtype="datetime64[ns]",
                name="Time",
            )
            pd.testing.assert_index_equal(mload["timetable_tz"].index, expected)
            np.testing.assert_array_equal(mload["timetable_tz"]["x"], [1.0, 2.0])

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    @pytest.mark.xfail(reason="Writing attributes not implemented")
    def test_timetable_with_attrs(self, filename, version):
        """Test writing numeric timetable to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(
            file_path, variable_names=["timetable_with_attrs"], add_table_attrs=True
        )

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(
                temp_file_path,
                variable_names=["timetable_with_attrs"],
                add_table_attrs=True,
            )

            pd.testing.assert_frame_equal(
                mdict["timetable_with_attrs"],
                mload["timetable_with_attrs"],
                check_like=True,
            )

            for key, value in mdict["timetable_with_attrs"].attrs.items():
                assert key in mload["timetable_with_attrs"].attrs
                if isinstance(value, np.ndarray):
                    np.testing.assert_array_equal(
                        value, mload["timetable_with_attrs"].attrs[key]
                    )
                else:
                    assert value == mload["timetable_with_attrs"].attrs[key]

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)


timetablev7_files = [("test_tablev5_v7.mat", "v7"), ("test_tablev5_v73.mat", "v7.3")]


@pytest.mark.parametrize("filename, version", timetablev7_files)
class TestLoadMatlabTimetableV7:

    def test_timetable_v7_long_varname(self, filename, version):
        """Test reading timetable with variable names longer than 63 characters"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(
            file_path,
            variable_names=["timetable_v7_long_varname"],
        )
        assert "timetable_v7_long_varname" in mdict

        long_name = "A" * 100
        df = pd.DataFrame(
            {long_name: [1.0, 2.0, 3.0]},
            index=pd.Index(
                np.array(
                    ["2023-01-01", "2023-01-02", "2023-01-03"],
                    dtype="datetime64[ns]",
                ),
                name="Time",
            ),
        )
        pd.testing.assert_frame_equal(
            mdict["timetable_v7_long_varname"], df, check_like=True
        )

    def test_timetable_v7_long_dimname(self, filename, version):
        """Test reading timetable with dimension names longer than 63 characters"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(
            file_path,
            variable_names=["timetable_v7_long_dimname"],
        )
        assert "timetable_v7_long_dimname" in mdict

        df = pd.DataFrame(
            {"Value": [1.0, 2.0, 3.0]},
            index=pd.Index(
                np.array(
                    ["2023-01-01", "2023-01-02", "2023-01-03"],
                    dtype="datetime64[ns]",
                ),
                name="R" * 100,
            ),
        )
        pd.testing.assert_frame_equal(
            mdict["timetable_v7_long_dimname"], df, check_like=True
        )


timetablev8_files = [
    ("test_timetablev8_v7.mat", "v7"),
    ("test_timetablev8_v73.mat", "v7.3"),
]


@pytest.mark.parametrize("filename, version", timetablev8_files)
class TestLoadMatlabTimetableV8:

    @pytest.mark.parametrize(
        "var_name, col_name",
        [("timetable_v8", "data1"), ("timetable_v8_long_varname", "A" * 100)],
    )
    def test_timetable_v8(self, filename, version, var_name, col_name):
        """Test reading timetable saved with MATLAB 2026b (version 8)"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=[var_name])
        assert var_name in mdict

        df = pd.DataFrame(
            {col_name: [1.0, 2.0, 3.0]},
            index=pd.Index(
                np.array(
                    ["2023-01-01", "2023-01-02", "2023-01-03"],
                    dtype="datetime64[ns]",
                ),
                name="Time",
            ),
        )
        pd.testing.assert_frame_equal(mdict[var_name], df, check_like=True)


timetable_steps_files = [
    ("test_timetable_steps_v7.mat", "v7"),
    ("test_timetable_steps_v73.mat", "v7.3"),
]


@pytest.mark.parametrize("filename, version", timetable_steps_files)
class TestLoadMatlabTimetableSteps:

    @pytest.mark.parametrize(
        "var_name, row_times",
        [
            (
                "timetable_step_millis",
                [
                    "2020-01-01T00:00:00.25",
                    "2020-01-01T00:00:00.26",
                    "2020-01-01T00:00:00.27",
                ],
            ),
            (
                "timetable_step_start_offset",
                [
                    "2020-01-01T00:00:00.25",
                    "2020-01-01T00:00:01.25",
                    "2020-01-01T00:00:02.25",
                ],
            ),
            (
                "timetable_step_years",
                ["2020-01-01T00:00", "2020-12-31T05:49:12", "2021-12-31T11:38:24"],
            ),
            (
                "timetable_step_caldays",
                ["2020-01-01T12:30", "2020-01-02T12:30", "2020-01-03T12:30"],
            ),
            (
                "timetable_step_calmonths",
                ["2020-01-15T12:30", "2020-02-15T12:30", "2020-03-15T12:30"],
            ),
        ],
    )
    def test_timetable_step_row_times(self, filename, version, var_name, row_times):
        """Test row times of timetables with TimeStep keep start time and step"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=[var_name])
        assert var_name in mdict

        df = pd.DataFrame(
            {"data1": [1.0, 2.0, 3.0]},
            index=pd.Index(np.array(row_times, dtype="datetime64[ns]"), name="Time"),
        )
        pd.testing.assert_frame_equal(mdict[var_name], df, check_like=True)

    def test_timetable_sample_rate_no_drift(self, filename, version):
        """Test row times of timetables with SampleRate are k / fs without drift"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        with pytest.warns(MatConvertWarning, match="get_row_times"):
            mdict = load_from_mat(file_path, variable_names=["timetable_sample_rate"])
        assert "timetable_sample_rate" in mdict

        df = pd.DataFrame(
            {"data1": [1.0, 2.0, 3.0, 4.0]},
            index=pd.Index(
                np.array(
                    [0, 333333333, 666666667, 1000000000], dtype="timedelta64[ns]"
                ),
                name="Time",
            ),
        )
        pd.testing.assert_frame_equal(
            mdict["timetable_sample_rate"], df, check_like=True
        )


def test_row_times_sample_rate_datetime_unit():
    """SampleRate row times keep the unit of a datetime start time"""
    fields = ("origin", "specifiedAsRate", "sampleRate")
    row_times = np.empty((1, 1), dtype=[(f, object) for f in fields])
    row_times[0, 0] = (
        np.array([["2500-01-01"]], dtype="datetime64[us]"),
        np.array([[True]]),
        np.array([[100.0]]),
    )
    with pytest.warns(MatConvertWarning, match="get_row_times"):
        times = get_row_times(row_times, 3)

    expected = np.array(
        ["2500-01-01T00:00:00", "2500-01-01T00:00:00.01", "2500-01-01T00:00:00.02"],
        dtype="datetime64[us]",
    )
    np.testing.assert_array_equal(times, expected, strict=True)


categorical_files = [
    ("test_table_categorical_v7.mat", "v7"),
    ("test_table_categorical_v73.mat", "v7.3"),
]


@pytest.mark.parametrize("filename, version", categorical_files)
class TestLoadMatlabTimetableCategorical:

    def test_timetable_categorical(self, filename, version):
        """Test reading timetable with categorical column from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["timetable_categorical"])
        assert "timetable_categorical" in mdict

        df = pd.DataFrame(
            {
                "Level": pd.Categorical(
                    ["low", "high", "low"], categories=["high", "low"]
                ),
            },
            index=pd.Index(
                np.array([1, 2, 3], dtype="timedelta64[s]"),
                name="Time",
            ),
        )
        pd.testing.assert_frame_equal(
            mdict["timetable_categorical"], df, check_like=True
        )


original_names_files = [
    ("test_table_original_names_v7.mat", "v7"),
    ("test_table_original_names_v73.mat", "v7.3"),
]


@pytest.mark.parametrize("filename, version", original_names_files)
class TestLoadMatlabTimetableOriginalNames:

    def test_timetable_original_names(self, filename, version):
        """Test reading timetable with names that are not valid identifiers"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["timetable_original_names"])
        assert "timetable_original_names" in mdict

        df = pd.DataFrame(
            {"p [bar]": [1.0, 2.0]},
            index=pd.Index(
                np.array([1, 2], dtype="timedelta64[s]"),
                name="Time [s]",
            ),
        )
        pd.testing.assert_frame_equal(
            mdict["timetable_original_names"], df, check_like=True
        )


struct_files = [
    ("test_table_struct_v7.mat", "v7"),
    ("test_table_struct_v73.mat", "v7.3"),
]


@pytest.mark.parametrize("filename, version", struct_files)
class TestLoadMatlabTimetableStruct:

    def test_timetable_struct(self, filename, version):
        """Test reading timetable with struct array column from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["timetable_struct"])
        assert "timetable_struct" in mdict
        assert isinstance(mdict["timetable_struct"], pd.DataFrame)

        np.testing.assert_array_equal(
            mdict["timetable_struct"].index,
            np.array([1, 2], dtype="timedelta64[s]"),
        )
        np.testing.assert_array_equal(
            mdict["timetable_struct"]["Value"], np.array([3.0, 4.0])
        )

        col = mdict["timetable_struct"]["S"]
        for i, (a, b) in enumerate([(1.0, "x"), (2.0, "y")]):
            assert isinstance(col.iloc[i], np.ndarray)
            assert col.iloc[i].shape == (1, 1)
            assert set(col.iloc[i].dtype.names) == {"a", "b"}
            np.testing.assert_array_equal(col.iloc[i]["a"][0, 0], np.array([[a]]))
            np.testing.assert_array_equal(col.iloc[i]["b"][0, 0], np.array([b]))
