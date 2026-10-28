from pathlib import Path

import pytest

from matio import whosmat


@pytest.mark.parametrize("name, shape", [("s", (1, 1)), ("a", (2, 3)), ("n", (1, 1))])
def test_whosmat_struct_with_cell_field(name, shape):
    # MATLAB: s = struct('c', {{1, 2}}); a = repmat(s, 2, 3);
    # n = struct('n', [1, 2, 3]); save(..., 's', 'a', 'n', '-v7.3')
    file = Path(__file__).parent / "data" / "struct_cells.mat"
    assert whosmat(file)[name] == (shape, "struct")
