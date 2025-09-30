import os
import tempfile

import numpy as np
import pytest

from matio import load_from_mat, save_to_mat

files = [("test_basic_v7.mat", "v7"), ("test_basic_v73.mat", "v7.3")]


@pytest.mark.parametrize("filename, version", files)
class TestLoadSparse:
    """Tests for reading sparse matrices from MAT-files"""

    def test_sparse_empty(self, filename, version):
        """Test reading Sparse Empty from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(
            file_path, variable_names=["sparse_empty"], spmatrix=False
        )
        assert "sparse_empty" in mdict

        sp = np.empty((0, 0), dtype=np.float64)
        np.testing.assert_array_equal(mdict["sparse_empty"].toarray(), sp, strict=True)

    def test_sparse_col(self, filename, version):
        """Test reading Sparse Column from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["sparse_col"], spmatrix=False)
        assert "sparse_col" in mdict

        sp = np.array([[0], [1], [0], [3]], dtype=np.float64)
        np.testing.assert_array_equal(mdict["sparse_col"].toarray(), sp, strict=True)

    def test_sparse_row(self, filename, version):
        """Test reading Sparse Row from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["sparse_row"], spmatrix=False)
        assert "sparse_row" in mdict

        sp = np.array([[0, 5, 0, 0]], dtype=np.float64)
        np.testing.assert_array_equal(mdict["sparse_row"].toarray(), sp, strict=True)

    def test_sparse_diag(self, filename, version):
        """Test reading Sparse Diagonal from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["sparse_diag"], spmatrix=False)
        assert "sparse_diag" in mdict

        sp = np.array(
            [
                [1, 0, 0, 0, 0],
                [0, 2, 0, 0, 0],
                [0, 0, 3, 0, 0],
                [0, 0, 0, 4, 0],
                [0, 0, 0, 0, 5],
            ],
            dtype=np.float64,
        )
        np.testing.assert_array_equal(mdict["sparse_diag"].toarray(), sp, strict=True)

    def test_sparse_rec_row(self, filename, version):
        """Test reading Sparse Rectangular Row from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(
            file_path, variable_names=["sparse_rec_row"], spmatrix=False
        )
        assert "sparse_rec_row" in mdict

        sp = np.array([[1, 0], [0, 2], [3, 0], [0, 4]], dtype=np.float64)
        np.testing.assert_array_equal(
            mdict["sparse_rec_row"].toarray(), sp, strict=True
        )

    def test_sparse_rec_col(self, filename, version):
        """Test reading Sparse Rectangular Column from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(
            file_path, variable_names=["sparse_rec_col"], spmatrix=False
        )
        assert "sparse_rec_col" in mdict

        sp = np.array([[1, 0, 0, 2], [0, 3, 0, 0]], dtype=np.float64)
        np.testing.assert_array_equal(
            mdict["sparse_rec_col"].toarray(), sp, strict=True
        )

    def test_sparse_symmetric(self, filename, version):
        """Test reading Sparse Symmetric from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(
            file_path, variable_names=["sparse_symmetric"], spmatrix=False
        )
        assert "sparse_symmetric" in mdict

        sp = np.array([[1, 2, 0], [2, 3, 4], [0, 4, 5]], dtype=np.float64)
        np.testing.assert_array_equal(
            mdict["sparse_symmetric"].toarray(), sp, strict=True
        )

    def test_sparse_neg(self, filename, version):
        """Test reading Sparse with Negative Values from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["sparse_neg"], spmatrix=False)
        assert "sparse_neg" in mdict

        sp = np.array([[0, -1, 0], [2, 0, 0], [0, 0, 3]], dtype=np.float64)
        np.testing.assert_array_equal(mdict["sparse_neg"].toarray(), sp, strict=True)

    def test_sparse_logical(self, filename, version):
        """Test reading Sparse Logical from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(
            file_path, variable_names=["sparse_logical"], spmatrix=False
        )
        assert "sparse_logical" in mdict

        sp = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.bool_)
        np.testing.assert_array_equal(
            mdict["sparse_logical"].toarray(), sp, strict=True
        )

    def test_sparse_complex(self, filename, version):
        """Test reading Sparse Complex from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(
            file_path, variable_names=["sparse_complex"], spmatrix=False
        )
        assert "sparse_complex" in mdict

        sp = np.array(
            [[1 + 1j, 0, 0], [0, 2 - 2j, 0], [0, 0, 3 + 3j]], dtype=np.complex128
        )
        np.testing.assert_array_equal(
            mdict["sparse_complex"].toarray(), sp, strict=True
        )

    def test_sparse_nnz(self, filename, version):
        """Test reading sparse with all non-zero elements from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["sparse_nnz"], spmatrix=False)
        assert "sparse_nnz" in mdict

        sp = np.array([[1, 2], [3, 4]], dtype=np.float64)
        np.testing.assert_array_equal(mdict["sparse_nnz"].toarray(), sp, strict=True)

    def test_sparse_all_zeros(self, filename, version):
        """Test reading sparse with all zero elements from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(
            file_path, variable_names=["sparse_all_zeros"], spmatrix=False
        )
        assert "sparse_all_zeros" in mdict

        sp = np.array([[0, 0], [0, 0]], dtype=np.float64)
        np.testing.assert_array_equal(
            mdict["sparse_all_zeros"].toarray(), sp, strict=True
        )


@pytest.mark.parametrize("filename, version", files)
class TestSaveSparse:

    def test_sparse_empty(self, filename, version):
        """Test writing empty sparse to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["sparse_empty"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["sparse_empty"])

            np.testing.assert_array_equal(
                mdict["sparse_empty"].toarray(),
                mload["sparse_empty"].toarray(),
                strict=True,
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_sparse_col(self, filename, version):
        """Test writing sparse column to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["sparse_col"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["sparse_col"])

            np.testing.assert_array_equal(
                mdict["sparse_col"].toarray(),
                mload["sparse_col"].toarray(),
                strict=True,
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_sparse_row(self, filename, version):
        """Test writing sparse row to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["sparse_row"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["sparse_row"])

            np.testing.assert_array_equal(
                mdict["sparse_row"].toarray(),
                mload["sparse_row"].toarray(),
                strict=True,
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_sparse_diag(self, filename, version):
        """Test writing sparse diagonal matrix to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["sparse_diag"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["sparse_diag"])

            np.testing.assert_array_equal(
                mdict["sparse_diag"].toarray(),
                mload["sparse_diag"].toarray(),
                strict=True,
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_sparse_rec_row(self, filename, version):
        """Test writing sparse rectangular row to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["sparse_rec_row"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["sparse_rec_row"])

            np.testing.assert_array_equal(
                mdict["sparse_rec_row"].toarray(),
                mload["sparse_rec_row"].toarray(),
                strict=True,
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_sparse_rec_col(self, filename, version):
        """Test writing sparse rectangular column to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["sparse_rec_col"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["sparse_rec_col"])

            np.testing.assert_array_equal(
                mdict["sparse_rec_col"].toarray(),
                mload["sparse_rec_col"].toarray(),
                strict=True,
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_sparse_symmetric(self, filename, version):
        """Test writing sparse symmetric to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["sparse_symmetric"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["sparse_symmetric"])

            np.testing.assert_array_equal(
                mdict["sparse_symmetric"].toarray(),
                mload["sparse_symmetric"].toarray(),
                strict=True,
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_sparse_neg(self, filename, version):
        """Test writing sparse matrix with negative values to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["sparse_neg"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["sparse_neg"])

            np.testing.assert_array_equal(
                mdict["sparse_neg"].toarray(),
                mload["sparse_neg"].toarray(),
                strict=True,
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_sparse_logical(self, filename, version):
        """Test writing sparse matrix with logical values to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["sparse_logical"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["sparse_logical"])

            np.testing.assert_array_equal(
                mdict["sparse_logical"].toarray(),
                mload["sparse_logical"].toarray(),
                strict=True,
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_sparse_complex(self, filename, version):
        """Test writing sparse matrix with complex values to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["sparse_complex"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["sparse_complex"])

            np.testing.assert_array_equal(
                mdict["sparse_complex"].toarray(),
                mload["sparse_complex"].toarray(),
                strict=True,
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_sparse_nnz(self, filename, version):
        """Test writing sparse with all non-zeros to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["sparse_nnz"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["sparse_nnz"])

            np.testing.assert_array_equal(
                mdict["sparse_nnz"].toarray(),
                mload["sparse_nnz"].toarray(),
                strict=True,
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
