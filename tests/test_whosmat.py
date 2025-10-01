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

    var_v7 = whosmat(file_v7)
    var_v73 = whosmat(file_v73)

    var_v7 = sorted(var_v7, key=lambda x: x[0])
    var_v73 = sorted(var_v73, key=lambda x: x[0])

    for var_v7, var_v73 in zip(var_v7, var_v73):
        name_v7, shape_v7, classname_v7 = var_v7
        name_v73, shape_v73, classname_v73 = var_v73

        if "sparse" in classname_v7 or "sparse" in classname_v73:
            continue  # Skip sparse matrices for now

        assert name_v7 == name_v73, f"Variable names differ: {name_v7} vs {name_v73}"
        assert (
            shape_v7 == shape_v73
        ), f"Shapes differ for {name_v7}: {shape_v7} vs {shape_v73}"
        assert (
            classname_v7 == classname_v73
        ), f"Class names differ for {name_v7}: {classname_v7} vs {classname_v73}"
