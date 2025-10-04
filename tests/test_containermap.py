import os
import tempfile

import numpy as np
import pytest

from matio import load_from_mat, save_to_mat
from matio.utils.matclass import MatlabContainerMap

files = [("test_maps_v7.mat", "v7"), ("test_maps_v73.mat", "v7.3")]
namespace = "TestClasses"


@pytest.mark.parametrize("filename, version", files)
class TestLoadMatlabContainerMap:

    def test_map_char_keys(self, filename, version):
        """Test reading container.Map with char keys from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["map_char_keys"])
        assert "map_char_keys" in mdict
        assert isinstance(mdict["map_char_keys"], MatlabContainerMap)

        d = {
            "a": np.array([[1]], dtype=np.float64),
            "b": np.array([[2]], dtype=np.float64),
        }

        for key, val in d.items():
            assert key in mdict["map_char_keys"]
            if isinstance(val, np.ndarray):
                np.testing.assert_array_equal(
                    mdict["map_char_keys"][key], val, strict=True
                )
            else:
                assert mdict["map_char_keys"][key] == val

    def test_map_numeric_keys(self, filename, version):
        """Test reading container.Map with numeric keys from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["map_numeric_keys"])
        assert "map_numeric_keys" in mdict
        assert isinstance(mdict["map_numeric_keys"], MatlabContainerMap)

        d = {
            1: np.array(["a"]),
            2: np.array(["b"]),
        }

        for key, val in d.items():
            assert key in mdict["map_numeric_keys"]
            if isinstance(val, np.ndarray):
                np.testing.assert_array_equal(
                    mdict["map_numeric_keys"][key], val, strict=True
                )
            else:
                assert mdict["map_numeric_keys"][key] == val

    def test_map_string_keys(self, filename, version):
        """Test reading container.Map with char keys (from string) from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["map_string_keys"])
        assert "map_string_keys" in mdict
        assert isinstance(mdict["map_string_keys"], MatlabContainerMap)

        d = {
            "a": np.array([[1]], dtype=np.float64),
            "b": np.array([[2]], dtype=np.float64),
        }

        for key, val in d.items():
            assert key in mdict["map_string_keys"]
            if isinstance(val, np.ndarray):
                np.testing.assert_array_equal(
                    mdict["map_string_keys"][key], val, strict=True
                )
            else:
                assert mdict["map_string_keys"][key] == val

    def test_map_empty(self, filename, version):
        """Test reading container.Map empty from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["map_empty"])
        assert "map_empty" in mdict
        assert isinstance(mdict["map_empty"], MatlabContainerMap)
        assert len(mdict["map_empty"]) == 0


@pytest.mark.parametrize("filename, version", files)
class TestSaveMatlabContainerMap:

    def test_map_char_keys(self, filename, version):
        """Test writing container.Map with char keys to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["map_char_keys"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["map_char_keys"])

            for key, val in mdict["map_char_keys"].items():
                assert key in mload["map_char_keys"]
                if isinstance(val, np.ndarray):
                    np.testing.assert_array_equal(
                        mload["map_char_keys"][key], val, strict=True
                    )
                else:
                    assert mload["map_char_keys"][key] == val

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_map_numeric_keys(self, filename, version):
        """Test writing container.Map with numeric keys to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["map_numeric_keys"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["map_numeric_keys"])

            for key, val in mdict["map_numeric_keys"].items():
                assert key in mload["map_numeric_keys"]
                if isinstance(val, np.ndarray):
                    np.testing.assert_array_equal(
                        mload["map_numeric_keys"][key], val, strict=True
                    )
                else:
                    assert mload["map_numeric_keys"][key] == val

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_map_string_keys(self, filename, version):
        """Test writing container.Map with char keys to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["map_string_keys"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["map_string_keys"])

            for key, val in mdict["map_string_keys"].items():
                assert key in mload["map_string_keys"]
                if isinstance(val, np.ndarray):
                    np.testing.assert_array_equal(
                        mload["map_string_keys"][key], val, strict=True
                    )
                else:
                    assert mload["map_string_keys"][key] == val

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_map_empty(self, filename, version):
        """Test writing container.Map empty to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["map_empty"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["map_empty"])
            assert len(mload["map_empty"]) == 0

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
