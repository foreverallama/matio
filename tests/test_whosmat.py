from pathlib import Path

import numpy as np
import pytest

from matio import whosmat

DATA_DIR = Path(__file__).parent / "data"


def get_files():
    """Get *.mat files from data directory (only base name)"""
    files = list(DATA_DIR.glob("*.mat"))
    bases_v7 = {f.stem[:-3] for f in files if f.stem.endswith("_v7")}
    bases = sorted(bases_v7)
    return bases


file_pairs = [
    (str(DATA_DIR / f"{base}_v7.mat"), str(DATA_DIR / f"{base}_v73.mat"))
    for base in get_files()
]


@pytest.mark.parametrize("file_v7, file_v73", file_pairs)
def test_whosmat(file_v7, file_v73):
    """Test whosmat function for both v7 and v7.3 files."""
    print(f"Testing files: {file_v7} and {file_v73}")
    v7 = whosmat(file_v7)
    v73 = whosmat(file_v73)

    assert len(v7) == len(v73), f"Number of variables differ: {len(v7)} vs {len(v73)}"
    assert set(v7.keys()) == set(
        v73.keys()
    ), f"Variable names differ: {set(v7.keys())} vs {set(v73.keys())}"

    for var in v7.keys():
        shape_v7, classname_v7 = v7[var]
        shape_v73, classname_v73 = v73[var]

        assert (
            shape_v7 == shape_v73
        ), f"Shape mismatch for variable {var}: {shape_v7} vs {shape_v73}"
        assert (
            classname_v7 == classname_v73
        ), f"Class name mismatch for variable {var}: {classname_v7} vs {classname_v73}"


def test_whosmat_v4():
    """Test whosmat function for v4 file."""
    file_v4 = DATA_DIR / "test_basic_v4.mat"
    var_v4 = whosmat(file_v4)

    expected = {
        "char_array": ((3, 2), "char"),
        "char_empty": ((0, 0), "char"),
        "char_scalar": ((1, 5), "char"),
        "complex_array": ((3, 1), "complex double"),
        "complex_scalar": ((1, 1), "complex double"),
        "double_array": ((2, 3), "double"),
        "double_scalar": ((1, 1), "double"),
        "fp32": ((10002, 1), "double"),
        "fp64": ((10002, 1), "double"),
        "i16": ((10002, 1), "double"),
        "i32": ((10002, 1), "double"),
        "fp_small": ((9999, 1), "double"),
        "numeric_empty": ((0, 0), "double"),
        "u16": ((10002, 1), "double"),
        "u8": ((10002, 1), "double"),
        "sparse_all_zeros": ((2, 2), "sparse"),
        "sparse_col": ((4, 1), "sparse"),
        "sparse_complex": ((3, 3), "sparse"),
        "sparse_diag": ((5, 5), "sparse"),
        "sparse_empty": ((0, 0), "sparse"),
        "sparse_neg": ((3, 3), "sparse"),
        "sparse_nnz": ((2, 2), "sparse"),
        "sparse_rec_col": ((2, 4), "sparse"),
        "sparse_rec_row": ((4, 2), "sparse"),
        "sparse_row": ((1, 4), "sparse"),
        "sparse_symmetric": ((3, 3), "sparse"),
    }

    assert len(var_v4) == len(expected)
    assert set(var_v4.keys()) == set(expected.keys())

    for name, (shape, classname) in var_v4.items():
        expected_shape, expected_classname = expected[name]
        assert (
            shape == expected_shape
        ), f"Shape mismatch for variable {name}: expected {expected_shape}, got {shape}"
        assert (
            classname == expected_classname
        ), f"Class name mismatch for variable {name}: expected {expected_classname}, got {classname}"
