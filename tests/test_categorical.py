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
class TestLoadMatlabCategorical:

    def test_categorical_scalar(self, filename, version):
        """Test reading categorical scalar data from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["cat_scalar"])
        assert "cat_scalar" in mdict

        cats = np.array(["blue", "green", "red"], dtype=object)
        codes = np.array([[2, 1, 0, 2]], dtype=np.int8)
        ordered = False
        np.testing.assert_array_equal(mdict["cat_scalar"].codes, codes, strict=True)
        np.testing.assert_array_equal(mdict["cat_scalar"].categories, cats, strict=True)
        np.testing.assert_array_equal(mdict["cat_scalar"].ordered, ordered, strict=True)

    def test_categorical_array(self, filename, version):
        """Test reading categorical array data from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["cat_array"])
        assert "cat_array" in mdict

        cats = np.array(["high", "low", "medium"], dtype=object)
        codes = np.array([[1, 2], [0, 1]], dtype=np.int8)
        ordered = False

        np.testing.assert_array_equal(mdict["cat_array"].codes, codes, strict=True)
        np.testing.assert_array_equal(mdict["cat_array"].categories, cats, strict=True)
        np.testing.assert_array_equal(mdict["cat_array"].ordered, ordered, strict=True)

    def test_categorical_empty(self, filename, version):
        """Test reading empty categorical data from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["cat_empty"])
        assert "cat_empty" in mdict

        cats = np.array([], dtype=object)
        codes = np.empty((0, 0), dtype=np.int8)
        ordered = False

        np.testing.assert_array_equal(mdict["cat_empty"].codes, codes, strict=True)
        np.testing.assert_array_equal(mdict["cat_empty"].categories, cats, strict=True)
        np.testing.assert_array_equal(mdict["cat_empty"].ordered, ordered, strict=True)

    def test_categorical_from_numeric(self, filename, version):
        """Test reading categorical data created from numeric array from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["cat_from_numeric"])
        assert "cat_from_numeric" in mdict

        cats = np.array(["low", "medium", "high"], dtype=object)
        codes = np.array([[0, 1, 2, 1, 0]], dtype=np.int8)
        ordered = False

        np.testing.assert_array_equal(
            mdict["cat_from_numeric"].codes, codes, strict=True
        )
        np.testing.assert_array_equal(
            mdict["cat_from_numeric"].categories, cats, strict=True
        )
        np.testing.assert_array_equal(
            mdict["cat_from_numeric"].ordered, ordered, strict=True
        )

    def test_categorical_unordered(self, filename, version):
        """Test reading unordered categorical data from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["cat_unordered"])
        assert "cat_unordered" in mdict

        cats = np.array(["cold", "warm", "hot"], dtype=object)
        codes = np.array([[0, 2, 1]], dtype=np.int8)
        ordered = False

        np.testing.assert_array_equal(mdict["cat_unordered"].codes, codes, strict=True)
        np.testing.assert_array_equal(
            mdict["cat_unordered"].categories, cats, strict=True
        )
        np.testing.assert_array_equal(
            mdict["cat_unordered"].ordered, ordered, strict=True
        )

    def test_categorical_ordered(self, filename, version):
        """Test reading ordered categorical data from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["cat_ordered"])
        assert "cat_ordered" in mdict

        cats = np.array(["small", "medium", "large"], dtype=object)
        codes = np.array([[0, 1, 2]], dtype=np.int8)
        ordered = True

        np.testing.assert_array_equal(mdict["cat_ordered"].codes, codes, strict=True)
        np.testing.assert_array_equal(
            mdict["cat_ordered"].categories, cats, strict=True
        )
        np.testing.assert_array_equal(
            mdict["cat_ordered"].ordered, ordered, strict=True
        )

    def test_categorical_missing(self, filename, version):
        """Test reading categorical data with missing values from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["cat_missing"])
        assert "cat_missing" in mdict

        cats = np.array(["cat", "dog", "mouse"], dtype=object)
        codes = np.array([[0, -1, 1, 2]], dtype=np.int8)
        ordered = False

        np.testing.assert_array_equal(mdict["cat_missing"].codes, codes, strict=True)
        np.testing.assert_array_equal(
            mdict["cat_missing"].categories, cats, strict=True
        )
        np.testing.assert_array_equal(
            mdict["cat_missing"].ordered, ordered, strict=True
        )

    def test_categorical_mixed_case(self, filename, version):
        """Test reading categorical data with mixed case categories from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["cat_mixed_case"])
        assert "cat_mixed_case" in mdict

        cats_list = sorted(["On", "off", "OFF", "ON", "on"])
        cats = np.array(cats_list, dtype=object)
        codes = np.array([[2, 3, 0, 1, 4]], dtype=np.int8)
        ordered = False

        np.testing.assert_array_equal(mdict["cat_mixed_case"].codes, codes, strict=True)
        np.testing.assert_array_equal(
            mdict["cat_mixed_case"].categories, cats, strict=True
        )
        np.testing.assert_array_equal(
            mdict["cat_mixed_case"].ordered, ordered, strict=True
        )

    def test_categorical_matlab_string(self, filename, version):
        """Test reading categorical data with categories as MATLAB strings from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["cat_string"])
        assert "cat_string" in mdict

        cats_list = sorted(["spring", "summer", "autumn", "winter"])
        cats = np.array(cats_list, dtype=object)
        codes = np.array([[1, 2, 0, 3]], dtype=np.int8)
        ordered = False

        np.testing.assert_array_equal(mdict["cat_string"].codes, codes, strict=True)
        np.testing.assert_array_equal(mdict["cat_string"].categories, cats, strict=True)
        np.testing.assert_array_equal(mdict["cat_string"].ordered, ordered, strict=True)

    def test_categorical_3D(self, filename, version):
        """Test reading 3D categorical array data from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["cat_3D"])
        assert "cat_3D" in mdict

        cats_list = sorted(["yes", "no", "maybe"])
        cats = np.array(cats_list, dtype=object)
        codes = np.tile(np.array([[2, 2], [1, 1], [0, 0]], dtype=np.int8), (2, 1, 1))
        ordered = False

        np.testing.assert_array_equal(mdict["cat_3D"].codes, codes, strict=True)
        np.testing.assert_array_equal(mdict["cat_3D"].categories, cats, strict=True)
        np.testing.assert_array_equal(mdict["cat_3D"].ordered, ordered, strict=True)


@pytest.mark.parametrize("filename, version", files)
class TestSaveMatlabCategorical:

    def test_categorical_scalar(self, filename, version):
        """Test writing categorical scalar to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["cat_scalar"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["cat_scalar"])

            np.testing.assert_array_equal(
                mload["cat_scalar"].codes, mdict["cat_scalar"].codes, strict=True
            )
            np.testing.assert_array_equal(
                mload["cat_scalar"].categories,
                mdict["cat_scalar"].categories,
                strict=True,
            )
            np.testing.assert_array_equal(
                mload["cat_scalar"].ordered, mdict["cat_scalar"].ordered, strict=True
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_categorical_array(self, filename, version):
        """Test writing categorical array to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["cat_array"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["cat_array"])

            np.testing.assert_array_equal(
                mload["cat_array"].codes, mdict["cat_array"].codes, strict=True
            )
            np.testing.assert_array_equal(
                mload["cat_array"].categories,
                mdict["cat_array"].categories,
                strict=True,
            )
            np.testing.assert_array_equal(
                mload["cat_array"].ordered, mdict["cat_array"].ordered, strict=True
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_categorical_empty(self, filename, version):
        """Test writing categorical empty to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["cat_empty"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["cat_empty"])

            np.testing.assert_array_equal(
                mload["cat_empty"].codes, mdict["cat_empty"].codes, strict=True
            )
            np.testing.assert_array_equal(
                mload["cat_empty"].categories,
                mdict["cat_empty"].categories,
                strict=True,
            )
            np.testing.assert_array_equal(
                mload["cat_empty"].ordered, mdict["cat_empty"].ordered, strict=True
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_categorical_from_numeric(self, filename, version):
        """Test writing categorical numeric to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["cat_from_numeric"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["cat_from_numeric"])

            np.testing.assert_array_equal(
                mload["cat_from_numeric"].codes,
                mdict["cat_from_numeric"].codes,
                strict=True,
            )
            np.testing.assert_array_equal(
                mload["cat_from_numeric"].categories,
                mdict["cat_from_numeric"].categories,
                strict=True,
            )
            np.testing.assert_array_equal(
                mload["cat_from_numeric"].ordered,
                mdict["cat_from_numeric"].ordered,
                strict=True,
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_categorical_unordered(self, filename, version):
        """Test writing categorical unordered to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["cat_unordered"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["cat_unordered"])

            np.testing.assert_array_equal(
                mload["cat_unordered"].codes, mdict["cat_unordered"].codes, strict=True
            )
            np.testing.assert_array_equal(
                mload["cat_unordered"].categories,
                mdict["cat_unordered"].categories,
                strict=True,
            )
            np.testing.assert_array_equal(
                mload["cat_unordered"].ordered,
                mdict["cat_unordered"].ordered,
                strict=True,
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_categorical_ordered(self, filename, version):
        """Test writing categorical ordered to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["cat_ordered"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["cat_ordered"])

            np.testing.assert_array_equal(
                mload["cat_ordered"].codes, mdict["cat_ordered"].codes, strict=True
            )
            np.testing.assert_array_equal(
                mload["cat_ordered"].categories,
                mdict["cat_ordered"].categories,
                strict=True,
            )
            np.testing.assert_array_equal(
                mload["cat_ordered"].ordered, mdict["cat_ordered"].ordered, strict=True
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_categorical_missing(self, filename, version):
        """Test writing categorical with missing values to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["cat_missing"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["cat_missing"])

            np.testing.assert_array_equal(
                mload["cat_missing"].codes, mdict["cat_missing"].codes, strict=True
            )
            np.testing.assert_array_equal(
                mload["cat_missing"].categories,
                mdict["cat_missing"].categories,
                strict=True,
            )
            np.testing.assert_array_equal(
                mload["cat_missing"].ordered, mdict["cat_missing"].ordered, strict=True
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_categorical_mixed_case(self, filename, version):
        """Test writing categorical mixed case to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["cat_mixed_case"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["cat_mixed_case"])

            np.testing.assert_array_equal(
                mload["cat_mixed_case"].codes,
                mdict["cat_mixed_case"].codes,
                strict=True,
            )
            np.testing.assert_array_equal(
                mload["cat_mixed_case"].categories,
                mdict["cat_mixed_case"].categories,
                strict=True,
            )
            np.testing.assert_array_equal(
                mload["cat_mixed_case"].ordered,
                mdict["cat_mixed_case"].ordered,
                strict=True,
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_categorical_matlab_string(self, filename, version):
        """Test writing categorical string to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["cat_string"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["cat_string"])

            np.testing.assert_array_equal(
                mload["cat_string"].codes, mdict["cat_string"].codes, strict=True
            )
            np.testing.assert_array_equal(
                mload["cat_string"].categories,
                mdict["cat_string"].categories,
                strict=True,
            )
            np.testing.assert_array_equal(
                mload["cat_string"].ordered, mdict["cat_string"].ordered, strict=True
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_categorical_3D(self, filename, version):
        """Test writing categorical 3D to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["cat_3D"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["cat_3D"])

            np.testing.assert_array_equal(
                mload["cat_3D"].codes, mdict["cat_3D"].codes, strict=True
            )
            np.testing.assert_array_equal(
                mload["cat_3D"].categories, mdict["cat_3D"].categories, strict=True
            )
            np.testing.assert_array_equal(
                mload["cat_3D"].ordered, mdict["cat_3D"].ordered, strict=True
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
