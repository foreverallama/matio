import numpy as np
import pytest

from matio import load_from_mat, save_to_mat, whosmat


class ArrayLike:
    def __init__(self, values):
        self.values = values

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.values, dtype=dtype)


@pytest.mark.parametrize("version", ["v7", "v7.3"])
@pytest.mark.parametrize("oned_as", ["row", "col"])
@pytest.mark.parametrize(
    "values",
    [
        np.array(3.0),
        np.arange(3.0),
        np.array(["2024-02-03", "2024-02-04"], dtype="datetime64[D]"),
    ],
)
def test_array_protocol_normalizes_shape(tmp_path, version, oned_as, values):
    path = tmp_path / "array_like.mat"
    save_to_mat(
        path,
        {"wrapped": ArrayLike(values), "plain": values},
        version=version,
        oned_as=oned_as,
    )
    result = load_from_mat(path)
    np.testing.assert_array_equal(result["wrapped"], result["plain"], strict=True)
    info = whosmat(path)
    assert info["wrapped"] == info["plain"]
