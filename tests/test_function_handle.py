import os
import tempfile

import numpy as np
import pytest

from matio import load_from_mat, save_to_mat
from matio.utils.matclass import MatlabFunction, MatlabOpaque

files = [
    ("test_function_handles_v7.mat", "v7"),
    ("test_function_handles_v73.mat", "v7.3"),
]


@pytest.mark.parametrize("filename, version", files)
class TestLoadMatlabFunctionHandles:

    def test_fh_simple(self, filename, version):
        """Test reading function handle simple from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["builtin_fh"])
        assert "builtin_fh" in mdict
        assert isinstance(mdict["builtin_fh"], MatlabFunction)

        fh_type = np.array(["simple"])
        np.testing.assert_array_equal(
            mdict["builtin_fh"][0, 0]["function_handle"][0, 0]["type"],
            fh_type,
            strict=True,
        )

    def test_fh_simple_custom(self, filename, version):
        """Test reading function handle simple custom from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["custom_fh"])
        assert "custom_fh" in mdict
        assert isinstance(mdict["custom_fh"], MatlabFunction)

        fh_type = np.array(["simple"])
        np.testing.assert_array_equal(
            mdict["custom_fh"][0, 0]["function_handle"][0, 0]["type"],
            fh_type,
            strict=True,
        )

    def test_fh_nested(self, filename, version):
        """Test reading function handle nested from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["nested_fh"])
        assert "nested_fh" in mdict
        assert isinstance(mdict["nested_fh"], MatlabFunction)

        fh_type = np.array(["nested"])
        np.testing.assert_array_equal(
            mdict["nested_fh"][0, 0]["function_handle"][0, 0]["type"],
            fh_type,
            strict=True,
        )
        assert isinstance(
            mdict["nested_fh"][0, 0]["function_handle"][0, 0]["workspace"], MatlabOpaque
        )

    def test_fh_classmethod(self, filename, version):
        """Test reading function handle class method from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["class_fh"])
        assert "class_fh" in mdict
        assert isinstance(mdict["class_fh"], MatlabFunction)

        fh_type = np.array(["anonymous"])
        np.testing.assert_array_equal(
            mdict["class_fh"][0, 0]["function_handle"][0, 0]["type"],
            fh_type,
            strict=True,
        )
        assert isinstance(
            mdict["class_fh"][0, 0]["function_handle"][0, 0]["workspace"], MatlabOpaque
        )

    def test_fh_anonymous(self, filename, version):
        """Test reading function handle anonymous from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["anonymous_fh"])
        assert "anonymous_fh" in mdict
        assert isinstance(mdict["anonymous_fh"], MatlabFunction)

        fh_type = np.array(["anonymous"])
        np.testing.assert_array_equal(
            mdict["anonymous_fh"][0, 0]["function_handle"][0, 0]["type"],
            fh_type,
            strict=True,
        )
        assert isinstance(
            mdict["anonymous_fh"][0, 0]["function_handle"][0, 0]["workspace"],
            MatlabOpaque,
        )


@pytest.mark.parametrize("filename, version", files)
class TestSaveMatlabFunctionHandles:

    def test_fh_simple(self, filename, version):
        """Test writing function handle simple to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["builtin_fh"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["builtin_fh"])

            assert set(mload["builtin_fh"].dtype.names) == set(
                mdict["builtin_fh"].dtype.names
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_fh_nested(self, filename, version):
        """Test writing function handle nested to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["nested_fh"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["nested_fh"])

            assert set(mload["nested_fh"].dtype.names) == set(
                mdict["nested_fh"].dtype.names
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_fh_classmethod(self, filename, version):
        """Test writing function handle class method to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["class_fh"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["class_fh"])

            assert set(mload["class_fh"].dtype.names) == set(
                mdict["class_fh"].dtype.names
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_fh_anonymous(self, filename, version):
        """Test writing function handle anonymous to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["anonymous_fh"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["anonymous_fh"])

            assert set(mload["anonymous_fh"].dtype.names) == set(
                mdict["anonymous_fh"].dtype.names
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_fh_simple_custom(self, filename, version):
        """Test writing function handle simple custom to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["custom_fh"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["custom_fh"])

            assert set(mload["custom_fh"].dtype.names) == set(
                mdict["custom_fh"].dtype.names
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
