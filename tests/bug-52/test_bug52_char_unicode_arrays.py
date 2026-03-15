import os
import tempfile

import numpy as np
import pytest

from matio import load_from_mat, save_to_mat
from matio.utils.matclass import MatWriteWarning

files = [("chars.mat", "v7"), ("chars_hdf.mat", "v7.3")]

char_a = np.array(["Hello, MATLAB! 12345 ~!@#$%^&*()_+-=[]{};:,.<>/?"])
char_b = np.array(["Café naïve résumé — π ≈ 3.14159"])
char_c = np.array(["Music symbol: 𝄞  | Gothic letter: 𐍈"])
char_d = np.array(["Mixed planes: A Ω Ж 中 😀 🚀 🧬"])
char_e = np.array(["AB", "😀"])
char_f = np.array(["😀𝄞𐍈🚀", "𝄞𐍈🚀😀", "𐍈🚀😀𝄞", "🚀😀𝄞𐍈", "😀𝄞𐍈🚀", "𝄞𐍈🚀😀"]).reshape(
    (3, 2), order="F"
)
char_g = np.array(["ABC", "DEF"])
char_h = np.array(["A\0B\0C\0D"])
char_i = np.array(["A\0B\0C\0D\0"])


@pytest.mark.parametrize("filename, version", files)
def test_load_char(filename, version):
    """Test reading char data from MAT-file"""
    file_path = os.path.join(os.path.dirname(__file__), filename)
    mdict = load_from_mat(file_path)
    assert set(mdict.keys()) == {"a", "b", "c", "d", "e", "f", "g", "h", "i"}

    np.testing.assert_array_equal(mdict["a"], char_a, strict=True)
    np.testing.assert_array_equal(mdict["b"], char_b, strict=True)
    np.testing.assert_array_equal(mdict["c"], char_c, strict=True)
    np.testing.assert_array_equal(mdict["d"], char_d, strict=True)
    np.testing.assert_array_equal(mdict["e"], char_e, strict=True)
    np.testing.assert_array_equal(mdict["f"], char_f, strict=True)
    np.testing.assert_array_equal(mdict["g"], char_g, strict=True)
    np.testing.assert_array_equal(mdict["h"], char_h, strict=True)
    try:
        np.testing.assert_array_equal(mdict["i"], char_i, strict=True)
    except AssertionError:
        pytest.xfail(
            "NumPy does not preserve trailing NULL characters in MATLAB char arrays"
        )


@pytest.mark.parametrize("filename, version", files)
def test_write_char(filename, version):
    """Test writing char data to MAT-file"""
    file_path = os.path.join(os.path.dirname(__file__), filename)
    mdict = load_from_mat(file_path)

    with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
        temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path)

            np.testing.assert_array_equal(mload["a"], char_a, strict=True)
            np.testing.assert_array_equal(mload["b"], char_b, strict=True)
            np.testing.assert_array_equal(mload["c"], char_c, strict=True)
            np.testing.assert_array_equal(mload["d"], char_d, strict=True)
            np.testing.assert_array_equal(mload["e"], char_e, strict=True)
            np.testing.assert_array_equal(mload["f"], char_f, strict=True)
            np.testing.assert_array_equal(mload["g"], char_g, strict=True)
            np.testing.assert_array_equal(mload["h"], char_h, strict=True)
            try:
                np.testing.assert_array_equal(mload["i"], char_i, strict=True)
            except AssertionError:
                pytest.xfail(
                    "NumPy does not preserve trailing NULL characters in MATLAB char arrays"
                )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)


# Using some old files from MAT.jl

simple_string = np.array(["the quick brown fox"])
empty_string = np.empty((0,), dtype="U1")
concatenated_strings = np.array(["this is a string      ", "this is another string"])
accented_string = np.array(["thé qüîck browñ fòx"])

files_old = ["chars2_v7.mat", "chars2_hdf.mat", "chars2_v6.mat"]


@pytest.mark.parametrize("filename", files_old)
def test_load_old_char(filename):
    """Test reading char data from older MAT-files"""
    file_path = os.path.join(os.path.dirname(__file__), filename)
    mdict = load_from_mat(file_path)
    assert set(mdict.keys()) == {
        "simple_string",
        "empty_string",
        "concatenated_strings",
        "accented_string",
        "cell_strings",
    }

    np.testing.assert_array_equal(mdict["simple_string"], simple_string, strict=True)
    np.testing.assert_array_equal(mdict["empty_string"], empty_string, strict=True)
    np.testing.assert_array_equal(
        mdict["concatenated_strings"], concatenated_strings, strict=True
    )
    np.testing.assert_array_equal(
        mdict["accented_string"], accented_string, strict=True
    )
