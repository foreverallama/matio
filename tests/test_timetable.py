import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from matio import load_from_mat, save_to_mat
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
