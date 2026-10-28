from pathlib import Path

import numpy as np
import pytest

from matio import load_from_mat


@pytest.mark.parametrize("spmatrix", [False, True])
def test_sparse_logical_without_data(spmatrix):
    # MATLAB: x = logical(sparse(3, 4)); save(..., 'x', '-v7.3')
    path = Path(__file__).parent / "data" / "sparse_logical_zero_v73.mat"
    x = load_from_mat(path, spmatrix=spmatrix)["x"]
    assert x.shape == (3, 4)
    assert x.nnz == 0
    np.testing.assert_array_equal(
        x.toarray(), np.zeros((3, 4), dtype=np.bool_), strict=True
    )
