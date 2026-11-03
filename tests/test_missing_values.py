import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from matio import load_from_mat, save_to_mat
from matio.utils.matclass import MatConvertWarning, MatWriteWarning


@pytest.mark.parametrize("version", ["v7", "v7.3"])
@pytest.mark.parametrize(
    "values", [[1.0, 2.0, 3.0], [1, 2, 3], [True, False, True]], ids=["f", "i", "b"]
)
def test_save_masked_array_as_nan(version, values):
    masked = np.ma.array(values, mask=[False, True, False])
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "masked.mat")
        with pytest.warns(MatWriteWarning, match="Masked values"):
            save_to_mat(file_path, {"m": masked}, version=version, oned_as="row")
        result = load_from_mat(file_path)["m"]
    np.testing.assert_array_equal(
        result, [[float(values[0]), np.nan, float(values[2])]], strict=True
    )


@pytest.mark.parametrize("version", ["v7", "v7.3"])
def test_save_masked_single_stays_single(version):
    masked = np.ma.array(np.array([1, 2], dtype=np.float32), mask=[False, True])
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "masked.mat")
        with pytest.warns(MatWriteWarning, match="Masked values"):
            save_to_mat(file_path, {"m": masked}, version=version, oned_as="row")
        result = load_from_mat(file_path)["m"]
    np.testing.assert_array_equal(
        result, np.array([[1, np.nan]], dtype=np.float32), strict=True
    )


@pytest.mark.parametrize("version", ["v7", "v7.3"])
def test_save_masked_array_without_masked_values_keeps_class(version):
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "masked.mat")
        save_to_mat(file_path, {"m": np.ma.array([1, 2])}, version=version)
        assert load_from_mat(file_path)["m"].dtype == np.array([1, 2]).dtype


@pytest.mark.parametrize("version", ["v7", "v7.3"])
def test_table_float_extension_column_with_na_keeps_class(version, tmp_path):
    df = pd.DataFrame({"x": pd.array([1.5, None], dtype="Float32")})
    path = tmp_path / "t.mat"
    save_to_mat(path, {"T": df}, version=version)
    column = load_from_mat(path)["T"]["x"].to_numpy()
    np.testing.assert_array_equal(
        column, np.array([1.5, np.nan], dtype=np.float32), strict=True
    )


@pytest.mark.parametrize("version", ["v7", "v7.3"])
@pytest.mark.parametrize("dtype", ["boolean", "Int64"])
def test_table_nullable_column_with_na_is_double(version, dtype, tmp_path):
    df = pd.DataFrame({"x": pd.array([1, None, 0], dtype=dtype)})
    path = tmp_path / "t.mat"
    with pytest.warns(MatConvertWarning, match="saved as double with NaN"):
        save_to_mat(path, {"T": df}, version=version)
    column = load_from_mat(path)["T"]["x"].to_numpy()
    np.testing.assert_array_equal(column, [1.0, np.nan, 0.0], strict=True)
