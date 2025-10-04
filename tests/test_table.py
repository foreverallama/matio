import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from matio import load_from_mat, save_to_mat
from matio.utils.matclass import MatlabOpaque

files = [("test_tables_v7.mat", "v7"), ("test_tables_v73.mat", "v7.3")]
namespace = "TestClasses"


@pytest.mark.parametrize("filename, version", files)
class TestLoadMatlabTable:

    def test_table_numeric(self, filename, version):
        """Test reading datetime scalar data from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["table_numeric"])
        assert "table_numeric" in mdict

        df = pd.DataFrame({"Var1": [1.1, 2.2, 3.3], "Var2": [4.4, 5.5, 6.6]})
        pd.testing.assert_frame_equal(mdict["table_numeric"], df, check_like=True)

    def test_table_strings(self, filename, version):
        """Test reading table with string data from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["table_strings"])
        assert "table_strings" in mdict

        df = pd.DataFrame({"Var1": ["apple", "banana", "cherry"]})
        pd.testing.assert_frame_equal(mdict["table_strings"], df, check_like=True)

    def test_table_empty(self, filename, version):
        """Test reading empty table from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["table_empty"])
        assert "table_empty" in mdict

        df = pd.DataFrame()
        pd.testing.assert_frame_equal(mdict["table_empty"], df, check_like=True)

    def test_table_time(self, filename, version):
        """Test reading timetable from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["table_time"])
        assert "table_time" in mdict

        df = pd.DataFrame(
            {
                "Time": np.array(
                    [
                        "2020-01-01T00:00:00.000",
                        "2020-01-02T00:00:00.000",
                        "2020-01-03T00:00:00.000",
                    ],
                    dtype="datetime64[ns]",
                ),
                "Duration": np.array([30, 60, 90], dtype="timedelta64[s]"),
            }
        )
        pd.testing.assert_frame_equal(mdict["table_time"], df, check_like=True)

    def test_table_nan(self, filename, version):
        """Test reading table with NaN values from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["table_nan"])
        assert "table_nan" in mdict

        df = pd.DataFrame(
            {
                "Var1": [1.1, np.nan, 3.3],
                "Var2": np.array(["A", "", "C"]),
            }
        )
        pd.testing.assert_frame_equal(mdict["table_nan"], df, check_like=True)

    def test_table_from_cell(self, filename, version):
        """Test reading table created from cell array in MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["table_from_cell"])
        assert "table_from_cell" in mdict

        df = pd.DataFrame(
            {
                "Var1": [
                    np.array([[1.0]]),
                    np.array(["text"]),
                    np.array([["2023-01-01T00:00:00.000"]], dtype="datetime64[ns]"),
                ],
            }
        )
        pd.testing.assert_frame_equal(mdict["table_from_cell"], df, check_like=True)

    def test_table_var_names(self, filename, version):
        """Test reading table with variable names from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["table_var_names"])
        assert "table_var_names" in mdict

        df = pd.DataFrame(
            {
                "X": np.array([10.0, 20.0, 30.0]),
                "Y": np.array([100.0, 200.0, 300.0]),
            }
        )
        pd.testing.assert_frame_equal(mdict["table_var_names"], df, check_like=True)

    def test_table_row_names(self, filename, version):
        """Test reading table with row names from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["table_row_names"])
        assert "table_row_names" in mdict

        df = pd.DataFrame(
            {
                "Col1": np.array([1.0, 2.0, 3.0]),
                "Col2": np.array([4.0, 5.0, 6.0]),
            },
            index=["R1", "R2", "R3"],
        )
        pd.testing.assert_frame_equal(mdict["table_row_names"], df, check_like=True)

    def test_table_multi_col_data(self, filename, version):
        """Test reading table with multi-column data from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["table_multi_col_data"])
        assert "table_multi_col_data" in mdict

        df = pd.DataFrame(
            {
                "Var1": np.array(
                    [
                        "2023-01-01T00:00:00.000",
                        "2023-01-02T00:00:00.000",
                        "2023-01-03T00:00:00.000",
                    ],
                    dtype="datetime64[ns]",
                ),
                "multicoldata_1": np.array([1.0, 2.0, 3.0]),
                "multicoldata_2": np.array([4.0, 5.0, 6.0]),
            }
        )
        pd.testing.assert_frame_equal(
            mdict["table_multi_col_data"], df, check_like=True
        )

    def test_table_with_objects(self, filename, version):
        """Test reading table with object data from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["table_with_objects"])
        assert "table_with_objects" in mdict

        struct_field1 = np.array([[123]], dtype=np.float64)
        struct_field2 = np.array(["abc"], dtype=np.str_)

        col1 = mdict["table_with_objects"]["C"]
        for i in range(col1.size):
            assert isinstance(col1[i], np.ndarray)
            assert col1[i].dtype.hasobject
            assert set(col1[i].dtype.names) == {"field1", "field2"}
            np.testing.assert_array_equal(col1[i]["field1"][0, 0], struct_field1)
            np.testing.assert_array_equal(col1[i]["field2"][0, 0], struct_field2)

        col2 = mdict["table_with_objects"]["Var2"]
        for i in range(col2.size):
            assert isinstance(col2[i], MatlabOpaque)
            assert col2[i].classname == f"{namespace}.BasicClass"
            assert col2[i].type_system == "MCOS"
            np.testing.assert_array_equal(
                col2[i].properties["a"], np.array([[1]], dtype=np.float64)
            )

    def test_table_with_attrs(self, filename, version):
        """Test reading table with attributes from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(
            file_path, variable_names=["table_with_attrs"], add_table_attrs=True
        )
        assert "table_with_attrs" in mdict

        df = pd.DataFrame(
            {
                "ID": np.array([1.0, 2.0]),
                "Label": np.array(["one", "two"]),
            }
        )
        pd.testing.assert_frame_equal(mdict["table_with_attrs"], df, check_like=True)

        attrs = {
            "Description": "Test table with full metadata",
            "DimensionNames": ["RowId", "Features"],
            "VariableUnits": ["", "category"],
            "VariableDescriptions": ["ID number", "Category label"],
            "VariableContinuity": ["continuous", "step"],
            "UserData": np.array(
                [[(np.array(["UnitTest"]), np.array([[1.0]]))]],
                dtype=[("CreatedBy", "O"), ("Version", "O")],
            ),
        }

        for key, value in attrs.items():
            assert key in mdict["table_with_attrs"].attrs
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(
                    mdict["table_with_attrs"].attrs[key], value
                )
            else:
                assert mdict["table_with_attrs"].attrs[key] == value


@pytest.mark.parametrize("filename, version", files)
class TestSaveMatlabTable:

    def test_table_numeric(self, filename, version):
        """Test writing numeric table to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["table_numeric"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["table_numeric"])

            pd.testing.assert_frame_equal(
                mdict["table_numeric"], mload["table_numeric"], check_like=True
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_table_strings(self, filename, version):
        """Test writing string table to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["table_strings"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["table_strings"])

            pd.testing.assert_frame_equal(
                mdict["table_strings"], mload["table_strings"], check_like=True
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_table_empty(self, filename, version):
        """Test writing empty table to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["table_empty"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["table_empty"])

            pd.testing.assert_frame_equal(
                mdict["table_empty"], mload["table_empty"], check_like=True
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_table_time(self, filename, version):
        """Test writing timetable to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["table_time"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["table_time"])

            pd.testing.assert_frame_equal(
                mdict["table_time"], mload["table_time"], check_like=True
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_table_nan(self, filename, version):
        """Test writing table with NaN values to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["table_nan"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["table_nan"])

            pd.testing.assert_frame_equal(
                mdict["table_nan"], mload["table_nan"], check_like=True
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_table_from_cell(self, filename, version):
        """Test writing table created from cell array to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["table_from_cell"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["table_from_cell"])

            pd.testing.assert_frame_equal(
                mdict["table_from_cell"], mload["table_from_cell"], check_like=True
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_table_var_names(self, filename, version):
        """Test writing table with variable names to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["table_var_names"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["table_var_names"])

            pd.testing.assert_frame_equal(
                mdict["table_var_names"], mload["table_var_names"], check_like=True
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_table_row_names(self, filename, version):
        """Test writing table with row names to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["table_row_names"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["table_row_names"])

            pd.testing.assert_frame_equal(
                mdict["table_row_names"], mload["table_row_names"], check_like=True
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_table_multi_col_data(self, filename, version):
        """Test writing table with multi-column data to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["table_multi_col_data"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(
                temp_file_path, variable_names=["table_multi_col_data"]
            )

            pd.testing.assert_frame_equal(
                mdict["table_multi_col_data"],
                mload["table_multi_col_data"],
                check_like=True,
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_table_with_objects(self, filename, version):
        """Test writing table with object data to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["table_with_objects"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["table_with_objects"])

            col1_orig = mdict["table_with_objects"]["C"]
            col1_load = mload["table_with_objects"]["C"]
            for i in range(col1_orig.size):
                assert isinstance(col1_load[i], np.ndarray)
                assert col1_load[i].dtype.hasobject
                assert set(col1_load[i].dtype.names) == {"field1", "field2"}
                np.testing.assert_array_equal(
                    col1_load[i]["field1"][0, 0], col1_orig[i]["field1"][0, 0]
                )
                np.testing.assert_array_equal(
                    col1_load[i]["field2"][0, 0], col1_orig[i]["field2"][0, 0]
                )

            col2_orig = mdict["table_with_objects"]["Var2"]
            col2_load = mload["table_with_objects"]["Var2"]
            for i in range(col2_orig.size):
                assert isinstance(col2_load[i], MatlabOpaque)
                assert col2_load[i].classname == col2_orig[i].classname
                assert col2_load[i].type_system == col2_orig[i].type_system
                np.testing.assert_array_equal(
                    col2_load[i].properties["a"], col2_orig[i].properties["a"]
                )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    @pytest.mark.xfail(reason="Writing attributes not implemented")
    def test_table_with_attrs(self, filename, version):
        """Test writing table with attributes to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(
            file_path, variable_names=["table_with_attrs"], add_table_attrs=True
        )

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version, do_compression=False)
            mload = load_from_mat(
                temp_file_path,
                variable_names=["table_with_attrs"],
                add_table_attrs=True,
            )

            pd.testing.assert_frame_equal(
                mdict["table_with_attrs"],
                mload["table_with_attrs"],
                check_like=True,
            )

            for key in mdict["table_with_attrs"].attrs.keys():
                assert key in mload["table_with_attrs"].attrs
                val_orig = mdict["table_with_attrs"].attrs[key]
                val_load = mload["table_with_attrs"].attrs[key]
                if isinstance(val_orig, np.ndarray):
                    np.testing.assert_array_equal(val_orig, val_load)
                else:
                    assert val_orig == val_load

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
