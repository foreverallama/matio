import os
import tempfile

import numpy as np
import pytest

from matio import load_from_mat, save_to_mat
from matio.utils.matclass import MatConvertWarning, MatlabDictionary, MatlabOpaque

map_files = [("test_maps_v7.mat", "v7"), ("test_maps_v73.mat", "v7.3")]
dict_files = [("test_dictionary_v7.mat", "v7"), ("test_dictionary_v73.mat", "v7.3")]
string_dtype = np.dtypes.StringDType(na_object=np.nan)


def load_dict(filename, varname):
    file_path = os.path.join(os.path.dirname(__file__), "data", filename)
    mdict = load_from_mat(file_path, variable_names=[varname])
    assert isinstance(mdict[varname], MatlabDictionary)
    return mdict[varname]


@pytest.mark.parametrize("filename, version", map_files)
class TestLoadMatlabDictionary:

    def test_dict_string_keys(self, filename, version):
        """Test reading dictionary with string keys from MAT-file"""
        d = load_dict(filename, "dict_string_keys")
        np.testing.assert_array_equal(
            d.keys, np.array([["x"], ["y"], ["z"]], dtype=string_dtype), strict=True
        )
        np.testing.assert_array_equal(
            d.values, np.array([[10.0], [20.0], [30.0]]), strict=True
        )
        assert d.to_dict() == {"x": 10.0, "y": 20.0, "z": 30.0}

    def test_dict_numeric_keys(self, filename, version):
        """Test reading dictionary with numeric keys from MAT-file"""
        d = load_dict(filename, "dict_numeric_keys")
        np.testing.assert_array_equal(
            d.keys, np.array([[1.0], [2.0], [3.0]]), strict=True
        )
        np.testing.assert_array_equal(
            d.values,
            np.array([["apple"], ["banana"], ["cherry"]], dtype=string_dtype),
            strict=True,
        )

    def test_dict_cell_keys(self, filename, version):
        """Test reading dictionary with cell keys from MAT-file"""
        d = load_dict(filename, "dict_cell_keys")
        assert d.keys.dtype == object
        assert d.keys.shape == (3, 1)
        for i in range(3):
            np.testing.assert_array_equal(
                d.keys[i, 0], np.array([[i + 1.0]]), strict=True
            )
        with pytest.raises(TypeError):
            d.to_dict()

    def test_dict_empty(self, filename, version):
        """Test reading unconfigured dictionary from MAT-file"""
        d = load_dict(filename, "dict_empty")
        assert d.keys is None
        assert d.values is None
        assert d.to_dict() == {}


@pytest.mark.parametrize("filename, version", dict_files)
class TestLoadMatlabDictionaryElementwise:

    def test_dict_int_keys(self, filename, version):
        """Test reading dictionary with integer keys from MAT-file"""
        d = load_dict(filename, "dict_int_keys")
        np.testing.assert_array_equal(
            d.keys, np.array([[1], [2]], dtype=np.int8), strict=True
        )
        assert d.to_dict() == {1: "a", 2: "b"}

    def test_dict_datetime_vals(self, filename, version):
        """Test reading dictionary with datetime values from MAT-file"""
        d = load_dict(filename, "dict_datetime_vals")
        np.testing.assert_array_equal(
            d.values,
            np.array([["2024-01-01"], ["2024-01-02"]], dtype="datetime64[ns]"),
            strict=True,
        )

    def test_dict_struct_vals(self, filename, version):
        """Test reading dictionary with struct values from MAT-file"""
        d = load_dict(filename, "dict_struct_vals")
        assert d.values.shape == (2, 1)
        assert d.values.dtype.names == ("f",)
        for i in range(2):
            np.testing.assert_array_equal(
                d.values[i, 0]["f"], np.array([[i + 1.0]]), strict=True
            )

    def test_dict_cell_vals(self, filename, version):
        """Test reading dictionary with cell values from MAT-file"""
        d = load_dict(filename, "dict_cell_vals")
        assert d.values.dtype == object
        assert d.values.shape == (2, 1)
        np.testing.assert_array_equal(d.values[0, 0], np.array([[1.0]]), strict=True)
        np.testing.assert_array_equal(d.values[1, 0], np.array(["x"]), strict=True)

    def test_dict_configured_empty(self, filename, version):
        """Test reading configured empty dictionary from MAT-file"""
        d = load_dict(filename, "dict_configured_empty")
        np.testing.assert_array_equal(
            d.keys, np.empty((0, 1), dtype=string_dtype), strict=True
        )
        np.testing.assert_array_equal(d.values, np.empty((0, 1)), strict=True)

    def test_dict_empty_datetime_keys(self, filename, version):
        """Test reading empty dictionary with datetime keys from MAT-file"""
        d = load_dict(filename, "dict_empty_datetime_keys")
        assert d.keys.dtype.kind == "M"
        assert d.keys.shape == (0, 1)
        np.testing.assert_array_equal(d.values, np.empty((0, 1)), strict=True)

    def test_dict_unsupported_vals(self, filename, version):
        """Test reading dictionary with categorical values from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        with pytest.warns(MatConvertWarning):
            mdict = load_from_mat(file_path, variable_names=["dict_categorical_vals"])
        assert isinstance(mdict["dict_categorical_vals"], MatlabOpaque)


def assert_dict_equal(actual, expected):
    if expected.keys is None:
        assert actual.keys is None and actual.values is None
        return
    for a, e in ((actual.keys, expected.keys), (actual.values, expected.values)):
        if e.dtype.names:
            assert a.shape == e.shape and a.dtype.names == e.dtype.names
            for name in e.dtype.names:
                for x, y in zip(a[name].ravel(), e[name].ravel()):
                    np.testing.assert_array_equal(x, y, strict=True)
        elif e.dtype == object:
            assert a.shape == e.shape and a.dtype == object
            for x, y in zip(a.ravel(), e.ravel()):
                np.testing.assert_array_equal(x, y, strict=True)
        else:
            np.testing.assert_array_equal(a, e, strict=True)


save_files = [
    (
        "test_maps_v7.mat",
        [
            "dict_cell_keys",
            "dict_cell_vals",
            "dict_empty",
            "dict_numeric_keys",
            "dict_string_keys",
            "dict_val_scalar",
        ],
    ),
    (
        "test_dictionary_v7.mat",
        [
            "dict_cell_vals",
            "dict_configured_empty",
            "dict_datetime_vals",
            "dict_empty_datetime_keys",
            "dict_int_keys",
            "dict_struct_vals",
        ],
    ),
]


@pytest.mark.parametrize("version", ["v7", "v7.3"])
@pytest.mark.parametrize("filename, varnames", save_files)
def test_save_dictionary(filename, varnames, version):
    """Test writing dictionaries read from MAT-file"""
    file_path = os.path.join(os.path.dirname(__file__), "data", filename)
    mdict = load_from_mat(file_path, variable_names=varnames)

    with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
        temp_file_path = tmpfile.name

    try:
        save_to_mat(temp_file_path, mdict, version=version)
        mload = load_from_mat(temp_file_path)
        assert set(mload) == set(varnames)
        for name, d in mdict.items():
            assert isinstance(mload[name], MatlabDictionary)
            assert_dict_equal(mload[name], d)

    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@pytest.mark.parametrize("version", ["v7", "v7.3"])
def test_save_dictionary_char(version):
    """Test writing dictionary with char keys to MAT-file"""
    mdict = {"dict_char": MatlabDictionary(np.array(["a", "b"]), np.array([1.0, 2.0]))}

    with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
        temp_file_path = tmpfile.name

    try:
        save_to_mat(temp_file_path, mdict, version=version)
        mload = load_from_mat(temp_file_path)
        np.testing.assert_array_equal(
            mload["dict_char"].keys,
            np.array([["a"], ["b"]], dtype=string_dtype),
            strict=True,
        )

    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
