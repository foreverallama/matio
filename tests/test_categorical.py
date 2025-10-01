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
