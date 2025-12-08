import os
import tempfile

import numpy as np
import pytest

from matio import load_from_mat, save_to_mat

files = [("test_basic_v7.mat", "v7"), ("test_basic_v73.mat", "v7.3")]


@pytest.mark.parametrize("filename, version", files)
class TestLoadStructCell:
    """Tests for reading Struct and Cell arrays from MAT-files"""

    def test_cell_scalar(self, filename, version):
        """Test reading Cell Scalar from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["cell_scalar"])

        cell_scalar = np.array(
            [[np.array(["text"], dtype=np.str_).reshape(1, 1)]], dtype=object
        ).reshape(1, 1)

        np.testing.assert_array_equal(mdict["cell_scalar"], cell_scalar, strict=True)

    def test_cell_array(self, filename, version):
        """Test reading Cell Array from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["cell_array"])

        cell1 = np.array(["A"], dtype=np.str_)
        cell2 = np.array([[1, 2], [3, 4]], dtype=np.float64).reshape(2, 2)
        cell3 = np.array(
            [[np.array([[True]])], [np.array([[False]])]], dtype=object
        ).reshape(1, 2)

        assert mdict["cell_array"].shape == (1, 3)
        assert mdict["cell_array"].dtype == object

        np.testing.assert_array_equal(mdict["cell_array"][0, 0], cell1, strict=True)
        np.testing.assert_array_equal(mdict["cell_array"][0, 1], cell2, strict=True)
        np.testing.assert_array_equal(mdict["cell_array"][0, 2], cell3, strict=True)

    def test_cell_nested(self, filename, version):
        """Test reading Nested Cell Array from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["cell_nested"])

        level1 = np.array(["level1"], dtype=np.str_)
        level2 = np.array(["level2"], dtype=np.str_)
        level3a = np.array(["level3"], dtype=np.str_)
        level3b = np.array([[123]], dtype=np.float64).reshape(1, 1)

        # Root
        assert mdict["cell_nested"].shape == (1, 1)
        assert mdict["cell_nested"].dtype == object

        # Level 1
        assert mdict["cell_nested"][0, 0].shape == (1, 2)
        assert mdict["cell_nested"][0, 0].dtype == object
        np.testing.assert_array_equal(
            mdict["cell_nested"][0, 0][0, 0], level1, strict=True
        )
        assert mdict["cell_nested"][0, 0][0, 1].shape == (1, 1)
        assert mdict["cell_nested"][0, 0][0, 1].dtype == object

        # Level 2
        assert mdict["cell_nested"][0, 0][0, 1][0, 0].shape == (1, 2)
        assert mdict["cell_nested"][0, 0][0, 1][0, 0].dtype == object
        np.testing.assert_array_equal(
            mdict["cell_nested"][0, 0][0, 1][0, 0][0, 0], level2, strict=True
        )
        assert mdict["cell_nested"][0, 0][0, 1][0, 0][0, 1].shape == (1, 1)
        assert mdict["cell_nested"][0, 0][0, 1][0, 0][0, 1].dtype == object

        # Level 3
        assert mdict["cell_nested"][0, 0][0, 1][0, 0][0, 1][0, 0].shape == (1, 2)
        assert mdict["cell_nested"][0, 0][0, 1][0, 0][0, 1][0, 0].dtype == object
        np.testing.assert_array_equal(
            mdict["cell_nested"][0, 0][0, 1][0, 0][0, 1][0, 0][0, 0],
            level3a,
            strict=True,
        )
        np.testing.assert_array_equal(
            mdict["cell_nested"][0, 0][0, 1][0, 0][0, 1][0, 0][0, 1],
            level3b,
            strict=True,
        )

    def test_cell_empty(self, filename, version):
        """Test reading Empty Cell Array from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["cell_empty"])

        assert mdict["cell_empty"].shape == (0, 0)
        assert mdict["cell_empty"].dtype == object

    def test_struct_scalar(self, filename, version):
        """Test reading Struct Scalar from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["struct_scalar"])

        name = np.array(["test"], dtype=np.str_)
        value = np.array([[123]], dtype=np.float64).reshape(1, 1)
        data = np.array([[1, 2], [3, 4]], dtype=np.float64).reshape(2, 2)

        assert mdict["struct_scalar"].shape == (1, 1)
        assert set(mdict["struct_scalar"].dtype.names) == {"name", "value", "data"}

        np.testing.assert_array_equal(
            mdict["struct_scalar"]["name"][0, 0], name, strict=True
        )
        np.testing.assert_array_equal(
            mdict["struct_scalar"]["value"][0, 0], value, strict=True
        )
        np.testing.assert_array_equal(
            mdict["struct_scalar"]["data"][0, 0], data, strict=True
        )

    def test_struct_array(self, filename, version):
        """Test reading Struct Array from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["struct_array"])

        id1 = np.array([[1]], dtype=np.float64).reshape(1, 1)
        id2 = np.array([[2]], dtype=np.float64).reshape(1, 1)
        info1 = np.array(["first"], dtype=np.str_)
        info2 = np.array(["second"], dtype=np.str_)

        assert mdict["struct_array"].shape == (1, 2)
        assert set(mdict["struct_array"].dtype.names) == {"id", "info"}

        np.testing.assert_array_equal(
            mdict["struct_array"]["id"][0, 0], id1, strict=True
        )
        np.testing.assert_array_equal(
            mdict["struct_array"]["info"][0, 0], info1, strict=True
        )
        np.testing.assert_array_equal(
            mdict["struct_array"]["id"][0, 1], id2, strict=True
        )
        np.testing.assert_array_equal(
            mdict["struct_array"]["info"][0, 1], info2, strict=True
        )

    def test_struct_empty(self, filename, version):
        """Test reading Empty Struct Array from MAT-file"""
        from matio.utils.matclass import EmptyMatStruct

        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["struct_empty"])
        assert "struct_empty" in mdict
        assert isinstance(mdict["struct_empty"], EmptyMatStruct)
        assert mdict["struct_empty"].shape == (0, 0)

    def test_struct_no_fields(self, filename, version):
        """Test reading Empty Struct Array from MAT-file"""
        from matio.utils.matclass import EmptyMatStruct

        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["struct_no_fields"])
        assert "struct_no_fields" in mdict
        assert isinstance(mdict["struct_no_fields"], EmptyMatStruct)
        assert mdict["struct_no_fields"].shape == (1, 1)

    def test_struct_nested(self, filename, version):
        """Test reading Nested Struct Array from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["struct_nested"])

        assert mdict["struct_nested"].shape == (1, 1)
        assert set(mdict["struct_nested"].dtype.names) == {"level1"}

        level1 = mdict["struct_nested"]["level1"][0, 0]
        assert level1.shape == (1, 1)
        assert set(level1.dtype.names) == {"level2"}

        level2 = level1["level2"][0, 0]
        assert level2.shape == (1, 1)
        assert set(level2.dtype.names) == {"level3", "cell"}

        level3 = level2["level3"][0, 0]
        level3_np = np.array([[42]], dtype=np.float64).reshape(1, 1)
        assert level3.shape == (1, 1)
        assert set(level3.dtype.names) == {"value"}
        np.testing.assert_array_equal(level3["value"][0, 0], level3_np, strict=True)

        cell = level2["cell"][0, 0]
        assert cell.shape == (1, 1)
        assert cell.dtype == object

        assert cell[0, 0].shape == (1, 2)
        assert cell[0, 0].dtype == object

        cell1 = np.array(["nested"], dtype=np.str_)
        np.testing.assert_array_equal(cell[0, 0][0, 0], cell1, strict=True)

        assert cell[0, 0][0, 1].shape == (1, 1)
        assert set(cell[0, 0][0, 1].dtype.names) == {"a", "b"}

        cell2a = np.array([[1]], dtype=np.float64).reshape(1, 1)
        cell2b = np.array([[2]], dtype=np.float64).reshape(1, 1)
        np.testing.assert_array_equal(cell[0, 0][0, 1]["a"][0, 0], cell2a, strict=True)
        np.testing.assert_array_equal(cell[0, 0][0, 1]["b"][0, 0], cell2b, strict=True)

    def test_struct_large(self, filename, version):
        """Test reading Large Struct from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(
            file_path, variable_names=["struct_large", "struct_even_larger"]
        )

        struct_large_fields = {f"field{i}" for i in range(1, 527)}
        struct_larger_fields = {f"s{i}" for i in range(1, 4094)}

        struct_large = mdict["struct_large"]
        assert set(struct_large.dtype.names) == struct_large_fields

        struct_even_larger = mdict["struct_even_larger"]
        assert set(struct_even_larger.dtype.names) == struct_larger_fields

        # Check random entry
        np.testing.assert_array_equal(
            struct_large["field200"][0, 0],
            np.array([[1]], dtype=np.float64).reshape(1, 1),
            strict=True,
        )
        np.testing.assert_array_equal(
            struct_even_larger["s2048"][0, 0],
            np.array([[2]], dtype=np.float64).reshape(1, 1),
            strict=True,
        )


@pytest.mark.parametrize("filename, version", files)
class TestSaveStructCell:

    def test_cell_scalar(self, filename, version):
        """Test writing cell scalar to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["cell_scalar"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["cell_scalar"])

            assert mload["cell_scalar"].shape == mdict["cell_scalar"].shape
            assert mload["cell_scalar"].dtype == mdict["cell_scalar"].dtype
            np.testing.assert_array_equal(
                mdict["cell_scalar"][0, 0], mload["cell_scalar"][0, 0], strict=True
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_cell_array(self, filename, version):
        """Test writing cell array to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["cell_array"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["cell_array"])

            assert mload["cell_array"].shape == mdict["cell_array"].shape
            assert mload["cell_array"].dtype == mdict["cell_array"].dtype
            for i in np.ndindex(mdict["cell_array"].shape):
                np.testing.assert_array_equal(
                    mdict["cell_array"][i], mload["cell_array"][i], strict=True
                )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_cell_nested(self, filename, version):
        """Test writing nested cell array to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["cell_nested"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["cell_nested"])

            # Root
            assert mload["cell_nested"].shape == mdict["cell_nested"].shape
            assert mload["cell_nested"].dtype == mdict["cell_nested"].dtype

            # Level 1
            assert mload["cell_nested"][0, 0].shape == mdict["cell_nested"][0, 0].shape
            assert mload["cell_nested"][0, 0].dtype == mdict["cell_nested"][0, 0].dtype
            np.testing.assert_array_equal(
                mdict["cell_nested"][0, 0][0, 0],
                mload["cell_nested"][0, 0][0, 0],
                strict=True,
            )
            assert (
                mload["cell_nested"][0, 0][0, 1].shape
                == mdict["cell_nested"][0, 0][0, 1].shape
            )
            assert (
                mload["cell_nested"][0, 0][0, 1].dtype
                == mdict["cell_nested"][0, 0][0, 1].dtype
            )

            # Level 2
            assert (
                mload["cell_nested"][0, 0][0, 1][0, 0].shape
                == mdict["cell_nested"][0, 0][0, 1][0, 0].shape
            )
            assert (
                mload["cell_nested"][0, 0][0, 1][0, 0].dtype
                == mdict["cell_nested"][0, 0][0, 1][0, 0].dtype
            )
            np.testing.assert_array_equal(
                mdict["cell_nested"][0, 0][0, 1][0, 0][0, 0],
                mload["cell_nested"][0, 0][0, 1][0, 0][0, 0],
                strict=True,
            )
            assert (
                mload["cell_nested"][0, 0][0, 1][0, 0][0, 1].shape
                == mdict["cell_nested"][0, 0][0, 1][0, 0][0, 1].shape
            )
            assert (
                mload["cell_nested"][0, 0][0, 1][0, 0][0, 1].dtype
                == mdict["cell_nested"][0, 0][0, 1][0, 0][0, 1].dtype
            )

            # Level 3
            assert (
                mload["cell_nested"][0, 0][0, 1][0, 0][0, 1][0, 0].shape
                == mdict["cell_nested"][0, 0][0, 1][0, 0][0, 1][0, 0].shape
            )
            assert (
                mload["cell_nested"][0, 0][0, 1][0, 0][0, 1][0, 0].dtype
                == mdict["cell_nested"][0, 0][0, 1][0, 0][0, 1][0, 0].dtype
            )
            np.testing.assert_array_equal(
                mdict["cell_nested"][0, 0][0, 1][0, 0][0, 1][0, 0][0, 0],
                mload["cell_nested"][0, 0][0, 1][0, 0][0, 1][0, 0][0, 0],
                strict=True,
            )
            np.testing.assert_array_equal(
                mdict["cell_nested"][0, 0][0, 1][0, 0][0, 1][0, 0][0, 1],
                mload["cell_nested"][0, 0][0, 1][0, 0][0, 1][0, 0][0, 1],
                strict=True,
            )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_cell_empty(self, filename, version):
        """Test writing cell empty to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["cell_empty"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["cell_empty"])

            assert mload["cell_empty"].shape == mdict["cell_empty"].shape
            assert mload["cell_empty"].dtype == mdict["cell_empty"].dtype

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_struct_scalar(self, filename, version):
        """Test writing struct scalar to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["struct_scalar"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["struct_scalar"])

            assert mload["struct_scalar"].shape == mdict["struct_scalar"].shape
            assert set(mload["struct_scalar"].dtype.names) == set(
                mdict["struct_scalar"].dtype.names
            )
            for name in mload["struct_scalar"].dtype.names:
                np.testing.assert_array_equal(
                    mdict["struct_scalar"][name][0, 0],
                    mload["struct_scalar"][name][0, 0],
                    strict=True,
                )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_struct_array(self, filename, version):
        """Test writing struct array to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["struct_array"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["struct_array"])

            assert mload["struct_array"].shape == mdict["struct_array"].shape
            assert set(mload["struct_array"].dtype.names) == set(
                mdict["struct_array"].dtype.names
            )
            for name in mload["struct_array"].dtype.names:
                for i in np.ndindex(mdict["struct_array"].shape):
                    np.testing.assert_array_equal(
                        mdict["struct_array"][name][i],
                        mload["struct_array"][name][i],
                        strict=True,
                    )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_struct_empty(self, filename, version):
        """Test writing struct empty to MAT-file"""
        from matio.utils.matclass import EmptyMatStruct

        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["struct_empty"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["struct_empty"])

            assert "struct_empty" in mload
            assert isinstance(mload["struct_empty"], EmptyMatStruct)
            assert mload["struct_empty"].shape == mdict["struct_empty"].shape

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_struct_no_fields(self, filename, version):
        """Test writing struct no fields to MAT-file"""
        from matio.utils.matclass import EmptyMatStruct

        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["struct_no_fields"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["struct_no_fields"])

            assert "struct_no_fields" in mload
            assert isinstance(mload["struct_no_fields"], EmptyMatStruct)
            assert mload["struct_no_fields"].shape == mdict["struct_no_fields"].shape

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_struct_nested(self, filename, version):
        """Test writing struct nested to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["struct_nested"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["struct_nested"])

            # Root
            assert mload["struct_nested"].shape == mdict["struct_nested"].shape
            assert set(mload["struct_nested"].dtype.names) == set(
                mdict["struct_nested"].dtype.names
            )

            level1_md = mdict["struct_nested"]["level1"][0, 0]
            level1_ml = mload["struct_nested"]["level1"][0, 0]
            assert level1_ml.shape == level1_md.shape
            assert set(level1_ml.dtype.names) == set(level1_md.dtype.names)

            level2_md = level1_md["level2"][0, 0]
            level2_ml = level1_ml["level2"][0, 0]
            assert level2_ml.shape == level2_md.shape
            assert set(level2_ml.dtype.names) == set(level2_md.dtype.names)

            level3_md = level2_md["level3"][0, 0]
            level3_ml = level2_ml["level3"][0, 0]
            assert level3_ml.shape == level3_md.shape
            assert set(level3_ml.dtype.names) == set(level3_md.dtype.names)
            np.testing.assert_array_equal(
                level3_md["value"][0, 0], level3_ml["value"][0, 0], strict=True
            )

            cell_md = level2_md["cell"][0, 0]
            cell_ml = level2_ml["cell"][0, 0]
            assert cell_ml.shape == cell_md.shape
            assert cell_ml.dtype == cell_md.dtype

            assert cell_ml[0, 0].shape == cell_md[0, 0].shape
            assert cell_ml[0, 0].dtype == cell_md[0, 0].dtype
            np.testing.assert_array_equal(
                cell_md[0, 0][0, 0], cell_ml[0, 0][0, 0], strict=True
            )
            assert cell_ml[0, 0][0, 1].shape == cell_md[0, 0][0, 1].shape
            assert cell_ml[0, 0][0, 1].dtype == cell_md[0, 0][0, 1].dtype
            assert set(cell_ml[0, 0][0, 1].dtype.names) == set(
                cell_md[0, 0][0, 1].dtype.names
            )
            for name in cell_ml[0, 0][0, 1].dtype.names:
                np.testing.assert_array_equal(
                    cell_md[0, 0][0, 1][name][0, 0],
                    cell_ml[0, 0][0, 1][name][0, 0],
                    strict=True,
                )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def test_struct_large(self, filename, version):
        """Test writing struct large to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["struct_even_larger"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["struct_even_larger"])

            assert set(mload["struct_even_larger"].dtype.names) == set(
                mdict["struct_even_larger"].dtype.names
            )
            # for name in mload["struct_even_larger"].dtype.names:
            #     np.testing.assert_array_equal(
            #         mdict["struct_even_larger"][name][0, 0],
            #         mload["struct_even_larger"][name][0, 0],
            #         strict=True,
            #     )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
