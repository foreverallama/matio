import os
import tempfile

import numpy as np
import pytest

from matio import load_from_mat, save_to_mat
from matio.utils.matclass import MatWriteWarning

files = [("test_basic_v7.mat", "v7"), ("test_basic_v73.mat", "v7.3")]


@pytest.mark.parametrize("filename, version", files)
class TestLoadBasicDataTypes:

    def test_numeric_int8(self, filename, version):
        """Test reading int8 data from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["int8_scalar", "int8_array"])
        assert "int8_scalar" in mdict
        assert "int8_array" in mdict

        int8_scalar = np.array([[42]], dtype=np.int8)
        int8_array = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.int8).reshape(2, 3)

        np.testing.assert_array_equal(mdict["int8_scalar"], int8_scalar, strict=True)
        np.testing.assert_array_equal(mdict["int8_array"], int8_array, strict=True)

    def test_numeric_uint8(self, filename, version):
        """Test reading uint8 data from MAT-file"""

        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["uint8_scalar", "uint8_array"])
        assert "uint8_scalar" in mdict
        assert "uint8_array" in mdict

        uint8_scalar = np.array([[42]], dtype=np.uint8)
        uint8_array = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.uint8).reshape(2, 3)

        np.testing.assert_array_equal(mdict["uint8_scalar"], uint8_scalar, strict=True)
        np.testing.assert_array_equal(mdict["uint8_array"], uint8_array, strict=True)

    def test_numeric_int16(self, filename, version):
        """Test reading int16 data from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["int16_scalar", "int16_array"])
        assert "int16_scalar" in mdict
        assert "int16_array" in mdict

        int16_scalar = np.array([[42]], dtype=np.int16)
        int16_array = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.int16).reshape(2, 3)

        np.testing.assert_array_equal(mdict["int16_scalar"], int16_scalar, strict=True)
        np.testing.assert_array_equal(mdict["int16_array"], int16_array, strict=True)

    def test_numeric_uint16(self, filename, version):
        """Test reading uint16 data from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(
            file_path, variable_names=["uint16_scalar", "uint16_array"]
        )
        assert "uint16_scalar" in mdict
        assert "uint16_array" in mdict

        uint16_scalar = np.array([[42]], dtype=np.uint16)
        uint16_array = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.uint16).reshape(2, 3)

        np.testing.assert_array_equal(
            mdict["uint16_scalar"], uint16_scalar, strict=True
        )
        np.testing.assert_array_equal(mdict["uint16_array"], uint16_array, strict=True)

    def test_numeric_int32(self, filename, version):
        """Test reading int32 data from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["int32_scalar", "int32_array"])
        assert "int32_scalar" in mdict
        assert "int32_array" in mdict

        int32_scalar = np.array([[42]], dtype=np.int32)
        int32_array = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.int32).reshape(2, 3)

        np.testing.assert_array_equal(mdict["int32_scalar"], int32_scalar, strict=True)
        np.testing.assert_array_equal(mdict["int32_array"], int32_array, strict=True)

    def test_numeric_uint32(self, filename, version):
        """Test reading uint32 data from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(
            file_path, variable_names=["uint32_scalar", "uint32_array"]
        )
        assert "uint32_scalar" in mdict
        assert "uint32_array" in mdict

        uint32_scalar = np.array([[42]], dtype=np.uint32)
        uint32_array = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.uint32).reshape(2, 3)

        np.testing.assert_array_equal(
            mdict["uint32_scalar"], uint32_scalar, strict=True
        )
        np.testing.assert_array_equal(mdict["uint32_array"], uint32_array, strict=True)

    def test_numeric_int64(self, filename, version):
        """Test reading int64 data from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["int64_scalar", "int64_array"])
        assert "int64_scalar" in mdict
        assert "int64_array" in mdict

        int64_scalar = np.array([[42]], dtype=np.int64)
        int64_array = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.int64).reshape(2, 3)

        np.testing.assert_array_equal(mdict["int64_scalar"], int64_scalar, strict=True)
        np.testing.assert_array_equal(mdict["int64_array"], int64_array, strict=True)

    def test_numeric_uint64(self, filename, version):
        """Test reading uint64 data from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(
            file_path, variable_names=["uint64_scalar", "uint64_array"]
        )
        assert "uint64_scalar" in mdict
        assert "uint64_array" in mdict

        uint64_scalar = np.array([[42]], dtype=np.uint64)
        uint64_array = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.uint64).reshape(2, 3)

        np.testing.assert_array_equal(
            mdict["uint64_scalar"], uint64_scalar, strict=True
        )
        np.testing.assert_array_equal(mdict["uint64_array"], uint64_array, strict=True)

    def test_numeric_single(self, filename, version):
        """Test reading single data from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(
            file_path, variable_names=["single_scalar", "single_array"]
        )
        assert "single_scalar" in mdict
        assert "single_array" in mdict

        single_scalar = np.array([[3.14]], dtype=np.float32)
        single_array = np.array(
            [[1.1, 2.2, 3.3, 4.4, 5.5, 6.6]], dtype=np.float32
        ).reshape(2, 3)

        np.testing.assert_array_equal(
            mdict["single_scalar"], single_scalar, strict=True
        )
        np.testing.assert_array_equal(mdict["single_array"], single_array, strict=True)

    def test_numeric_double(self, filename, version):
        """Test reading double data from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(
            file_path, variable_names=["double_scalar", "double_array"]
        )
        assert "double_scalar" in mdict
        assert "double_array" in mdict

        double_scalar = np.array([[3.14]], dtype=np.float64)
        double_array = np.array(
            [[1.1, 2.2, 3.3, 4.4, 5.5, 6.6]], dtype=np.float64
        ).reshape(2, 3)

        np.testing.assert_array_equal(
            mdict["double_scalar"], double_scalar, strict=True
        )
        np.testing.assert_array_equal(mdict["double_array"], double_array, strict=True)

    def test_logical(self, filename, version):
        """Test reading logical data from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(
            file_path, variable_names=["logical_scalar", "logical_array"]
        )
        assert "logical_scalar" in mdict
        assert "logical_array" in mdict

        logical_scalar = np.array([[1]], dtype=bool)
        logical_array = np.array([[1, 0, 1]], dtype=bool).reshape(1, 3)

        np.testing.assert_array_equal(
            mdict["logical_scalar"], logical_scalar, strict=True
        )
        np.testing.assert_array_equal(
            mdict["logical_array"], logical_array, strict=True
        )

    def test_complex(self, filename, version):
        """Test reading complex data from MAT-file"""

        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(
            file_path, variable_names=["complex_scalar", "complex_array"]
        )
        assert "complex_scalar" in mdict
        assert "complex_array" in mdict

        complex_scalar = np.array([[1.0 + 2.0j]], dtype=np.complex128).reshape(1, 1)
        complex_array = np.array(
            [[1.0 + 2.0j, 2.0 + 4.0j, 4.0 + 8.0j]], dtype=np.complex128
        ).reshape(3, 1)

        np.testing.assert_array_equal(
            mdict["complex_scalar"], complex_scalar, strict=True
        )
        np.testing.assert_array_equal(
            mdict["complex_array"], complex_array, strict=True
        )

    def test_char(self, filename, version):
        """Test reading char data from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["char_array", "char_scalar"])
        assert "char_array" in mdict
        assert "char_scalar" in mdict

        char_scalar = np.array(["Hello"], dtype=np.str_).reshape(
            1,
        )
        char_array = np.array(["ab", "cd", "ef"], dtype=np.str_).reshape(
            3,
        )

        np.testing.assert_array_equal(mdict["char_scalar"], char_scalar, strict=True)
        np.testing.assert_array_equal(mdict["char_array"], char_array, strict=True)

    def test_empty(self, filename, version):
        """Test reading empty arrays from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(
            file_path, variable_names=["numeric_empty", "char_empty", "logical_empty"]
        )
        assert "numeric_empty" in mdict
        assert "char_empty" in mdict
        assert "logical_empty" in mdict

        numeric_empty = np.empty((0, 0))
        char_empty = np.empty((0,), dtype=np.str_)
        logical_empty = np.empty((0, 0), dtype=np.bool_)

        np.testing.assert_array_equal(
            mdict["numeric_empty"], numeric_empty, strict=True
        )
        np.testing.assert_array_equal(mdict["char_empty"], char_empty, strict=True)
        np.testing.assert_array_equal(
            mdict["logical_empty"], logical_empty, strict=True
        )


@pytest.mark.parametrize("filename, version", files)
class TestSaveBasicDatatypes:

    def test_numeric_int8(self, filename, version):
        """Test writing int8 data to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["int8_scalar", "int8_array"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(
                temp_file_path, variable_names=["int8_scalar", "int8_array"]
            )

            np.testing.assert_array_equal(
                mdict["int8_scalar"], mload["int8_scalar"], strict=True
            )
            np.testing.assert_array_equal(
                mdict["int8_array"], mload["int8_array"], strict=True
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_numeric_uint8(self, filename, version):
        """Test writing uint8 data to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["uint8_scalar", "uint8_array"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(
                temp_file_path, variable_names=["uint8_scalar", "uint8_array"]
            )

            np.testing.assert_array_equal(
                mdict["uint8_scalar"], mload["uint8_scalar"], strict=True
            )
            np.testing.assert_array_equal(
                mdict["uint8_array"], mload["uint8_array"], strict=True
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_numeric_int16(self, filename, version):
        """Test writing int16 data to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["int16_scalar", "int16_array"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(
                temp_file_path, variable_names=["int16_scalar", "int16_array"]
            )

            np.testing.assert_array_equal(
                mdict["int16_scalar"], mload["int16_scalar"], strict=True
            )
            np.testing.assert_array_equal(
                mdict["int16_array"], mload["int16_array"], strict=True
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_numeric_uint16(self, filename, version):
        """Test writing uint16 data to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(
            file_path, variable_names=["uint16_scalar", "uint16_array"]
        )

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(
                temp_file_path, variable_names=["uint16_scalar", "uint16_array"]
            )

            np.testing.assert_array_equal(
                mdict["uint16_scalar"], mload["uint16_scalar"], strict=True
            )
            np.testing.assert_array_equal(
                mdict["uint16_array"], mload["uint16_array"], strict=True
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_numeric_int32(self, filename, version):
        """Test writing int32 data to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["int32_scalar", "int32_array"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(
                temp_file_path, variable_names=["int32_scalar", "int32_array"]
            )

            np.testing.assert_array_equal(
                mdict["int32_scalar"], mload["int32_scalar"], strict=True
            )
            np.testing.assert_array_equal(
                mdict["int32_array"], mload["int32_array"], strict=True
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_numeric_uint32(self, filename, version):
        """Test writing uint32 data to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(
            file_path, variable_names=["uint32_scalar", "uint32_array"]
        )

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(
                temp_file_path, variable_names=["uint32_scalar", "uint32_array"]
            )

            np.testing.assert_array_equal(
                mdict["uint32_scalar"], mload["uint32_scalar"], strict=True
            )
            np.testing.assert_array_equal(
                mdict["uint32_array"], mload["uint32_array"], strict=True
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_numeric_int64(self, filename, version):
        """Test writing int64 data to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["int64_scalar", "int64_array"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(
                temp_file_path, variable_names=["int64_scalar", "int64_array"]
            )

            np.testing.assert_array_equal(
                mdict["int64_scalar"], mload["int64_scalar"], strict=True
            )
            np.testing.assert_array_equal(
                mdict["int64_array"], mload["int64_array"], strict=True
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_numeric_uint64(self, filename, version):
        """Test writing uint64 data to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(
            file_path, variable_names=["uint64_scalar", "uint64_array"]
        )

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(
                temp_file_path, variable_names=["uint64_scalar", "uint64_array"]
            )

            np.testing.assert_array_equal(
                mdict["uint64_scalar"], mload["uint64_scalar"], strict=True
            )
            np.testing.assert_array_equal(
                mdict["uint64_array"], mload["uint64_array"], strict=True
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_numeric_single(self, filename, version):
        """Test writing single data to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(
            file_path, variable_names=["single_scalar", "single_array"]
        )

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(
                temp_file_path, variable_names=["single_scalar", "single_array"]
            )

            np.testing.assert_array_equal(
                mdict["single_scalar"], mload["single_scalar"], strict=True
            )
            np.testing.assert_array_equal(
                mdict["single_array"], mload["single_array"], strict=True
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_numeric_double(self, filename, version):
        """Test writing double data to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(
            file_path, variable_names=["double_scalar", "double_array"]
        )

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(
                temp_file_path, variable_names=["double_scalar", "double_array"]
            )

            np.testing.assert_array_equal(
                mdict["double_scalar"], mload["double_scalar"], strict=True
            )
            np.testing.assert_array_equal(
                mdict["double_array"], mload["double_array"], strict=True
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_logical(self, filename, version):
        """Test writing logical data to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(
            file_path, variable_names=["logical_scalar", "logical_array"]
        )

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(
                temp_file_path, variable_names=["logical_scalar", "logical_array"]
            )

            np.testing.assert_array_equal(
                mdict["logical_scalar"], mload["logical_scalar"], strict=True
            )
            np.testing.assert_array_equal(
                mdict["logical_array"], mload["logical_array"], strict=True
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_complex(self, filename, version):
        """Test writing complex data to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(
            file_path, variable_names=["complex_scalar", "complex_array"]
        )

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(
                temp_file_path, variable_names=["complex_scalar", "complex_array"]
            )

            np.testing.assert_array_equal(
                mdict["complex_scalar"], mload["complex_scalar"], strict=True
            )
            np.testing.assert_array_equal(
                mdict["complex_array"], mload["complex_array"], strict=True
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_char(self, filename, version):
        """Test writing char data to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["char_scalar", "char_array"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(
                temp_file_path, variable_names=["char_scalar", "char_array"]
            )

            np.testing.assert_array_equal(
                mdict["char_scalar"], mload["char_scalar"], strict=True
            )
            np.testing.assert_array_equal(
                mdict["char_array"], mload["char_array"], strict=True
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_empty(self, filename, version):
        """Test writing empty char data to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(
            file_path, variable_names=["char_empty", "numeric_empty", "logical_empty"]
        )

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(
                temp_file_path,
                variable_names=["char_empty", "numeric_empty", "logical_empty"],
            )

            np.testing.assert_array_equal(
                mdict["char_empty"], mload["char_empty"], strict=True
            )
            np.testing.assert_array_equal(
                mdict["numeric_empty"], mload["numeric_empty"], strict=True
            )
            np.testing.assert_array_equal(
                mdict["logical_empty"], mload["logical_empty"], strict=True
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)


@pytest.mark.parametrize("filename, version", files)
class TestWriteNonSupportedNumeric:

    def test_numpy_floats(self, filename, version):
        """Test writing numpy float16 data to MAT-file"""
        arr_16 = np.array([1, 2, 3], dtype=np.float16).reshape(1, 3)
        arr_128 = np.array([1, 2, 3], dtype=np.float128).reshape(1, 3)
        mdict = {"float16_array": arr_16, "float128_array": arr_128}

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            with pytest.warns(MatWriteWarning, match="not supported"):
                save_to_mat(temp_file_path, mdict, version=version)

            mload = load_from_mat(temp_file_path, variable_names=None)

            np.testing.assert_array_equal(
                mload["float16_array"],
                mdict["float16_array"].astype("float64"),
                strict=True,
            )
            np.testing.assert_array_equal(
                mload["float128_array"],
                mdict["float128_array"].astype("float64"),
                strict=True,
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_numpy_complex(self, filename, version):
        """Test writing numpy complex64 and complex256 data to MAT-file"""
        arr_c64 = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex64).reshape(1, 3)
        arr_c256 = np.array([1 + 2j, 3 + 4j, 5 + 6j], dtype=np.complex256).reshape(1, 3)
        mdict = {"complex64_array": arr_c64, "complex256_array": arr_c256}

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            with pytest.warns(MatWriteWarning, match="not supported"):
                save_to_mat(temp_file_path, mdict, version=version)

            mload = load_from_mat(temp_file_path, variable_names=None)

            np.testing.assert_array_equal(
                mload["complex64_array"],
                mdict["complex64_array"].astype("complex128"),
                strict=True,
            )
            np.testing.assert_array_equal(
                mload["complex256_array"],
                mdict["complex256_array"].astype("complex128"),
                strict=True,
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
