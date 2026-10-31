import os
import tempfile
import warnings

import numpy as np
import pandas as pd
import pytest

from matio import load_from_mat, save_to_mat
from matio.utils.converters.mattimes import mat_to_datetime
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

    def test_datetime_unit(self, filename, version):
        """Test reading datetime data with datetime_unit"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(
            file_path, variable_names=["dt_basic"], datetime_unit="us"
        )

        dt_scalar = np.array([["2025-04-01T12:00:00"]], dtype="datetime64[us]")
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

    @pytest.mark.parametrize("unit", ["ms", "ns"])
    def test_datetime_nat(self, filename, version, unit):
        """Test writing datetime with NaT to MAT-file"""
        dt_nat = np.array([["2024-01-01T12:00:00", "NaT"]], dtype=f"datetime64[{unit}]")

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, {"dt_nat": dt_nat}, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["dt_nat"])

            np.testing.assert_array_equal(np.isnat(mload["dt_nat"]), [[False, True]])
            assert abs(mload["dt_nat"][0, 0] - dt_nat[0, 0]) < np.timedelta64(1, "us")

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_datetime_subms(self, filename, version):
        """Test writing datetime with sub-millisecond precision to MAT-file"""
        dt_subms = np.array(
            [["2024-01-01T12:00:00.000000123", "1969-12-31T23:59:59.999999999", "NaT"]],
            dtype="datetime64[ns]",
        )

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, {"dt_subms": dt_subms}, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["dt_subms"])

            np.testing.assert_array_equal(mload["dt_subms"], dt_subms, strict=True)

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)


precision_files = [
    ("test_datetime_precision_v7.mat", "v7"),
    ("test_datetime_precision_v73.mat", "v7.3"),
]


@pytest.mark.parametrize("version", ["v7", "v7.3"])
@pytest.mark.parametrize("kind", ["datetime", "timedelta", "timezone"])
def test_save_pandas_time_index(tmp_path, version, kind):
    if kind == "timedelta":
        index = pd.to_timedelta([0, 1, 2], unit="s").as_unit("s")
        expected = index.to_numpy().astype("timedelta64[s]").reshape(1, -1)
    else:
        index = pd.date_range("2020-01-01", periods=3, freq="s")
        if kind == "timezone":
            index = index.tz_localize("Europe/Berlin")
        expected = (
            index.tz_convert(None) if index.tz is not None else index
        ).to_numpy()
        expected = expected.astype("datetime64[ns]").reshape(1, -1)
    path = tmp_path / "time_index.mat"
    if kind == "timezone":
        with pytest.warns(MatConvertWarning, match="converted to UTC"):
            save_to_mat(path, {"t": index}, version=version)
    else:
        save_to_mat(path, {"t": index}, version=version)
    np.testing.assert_array_equal(load_from_mat(path)["t"], expected, strict=True)


@pytest.mark.parametrize("version", ["v7", "v7.3"])
def test_save_pandas_time_scalars(tmp_path, version):
    path = tmp_path / "time_scalars.mat"
    mdict = {
        "ts": pd.Timestamp("2020-01-01 00:00:01.5"),
        "td": pd.Timedelta(minutes=2).as_unit("s"),
    }
    save_to_mat(path, mdict, version=version)
    mload = load_from_mat(path)
    expected = np.array([["2020-01-01T00:00:01.5"]], dtype="datetime64[ns]")
    np.testing.assert_array_equal(mload["ts"], expected, strict=True)
    assert mload["td"].shape == (1, 1)
    assert mload["td"][0, 0] == np.timedelta64(120, "s")


@pytest.mark.parametrize("filename, version", precision_files)
class TestLoadMatlabDatetimePrecision:

    def test_datetime_subms(self, filename, version):
        """Test reading datetime with sub-millisecond precision from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["dt_subms"])
        assert "dt_subms" in mdict

        dt_subms = np.array([["2024-01-01T12:00:00.1234567"]], dtype="datetime64[ns]")

        np.testing.assert_array_equal(mdict["dt_subms"], dt_subms, strict=True)


def test_datetime_outside_ns_range():
    """MATLAB datetimes outside the datetime64[ns] range become NaT"""
    # datetime(9999, 12, 31), datetime(2024, 1, 1) and the last whole milliseconds
    # inside the range, as stored by MATLAB (ms since 1970)
    data = np.array(
        [[253402214400000.0, 1704067200000.0, 9223372036854.0, -9223372036854.0]]
    )
    with pytest.warns(MatConvertWarning, match="outside the datetime64"):
        dt = mat_to_datetime({"data": data + 0j})
    expected = np.array(
        [["NaT", "2024-01-01", "2262-04-11T23:47:16.854", "1677-09-21T00:12:43.146"]],
        dtype="datetime64[ns]",
    )
    np.testing.assert_array_equal(dt, expected, strict=True)


def test_datetime_unit_range():
    """datetime_unit="us" keeps MATLAB datetimes up to the year 9999"""
    # datetime(9999, 12, 31) and datetime(2024, 1, 1) as stored by MATLAB
    data = np.array([[253402214400000.0, 1704067200000.0]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dt = mat_to_datetime({"data": data + 0j}, datetime_unit="us")
    expected = np.array([["9999-12-31", "2024-01-01"]], dtype="datetime64[us]")
    np.testing.assert_array_equal(dt, expected, strict=True)
