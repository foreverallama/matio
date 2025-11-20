import os
import tempfile

import numpy as np
import pytest

from matio import load_from_mat, save_to_mat
from matio.utils.matclass import MatlabOpaque

files = ["test_class_alias.mat"]


@pytest.mark.parametrize("filename", files)
class TestLoadClassAlias:

    def test_class_alias(self, filename):
        """Test reading class alias from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["obj1"])
        assert "obj1" in mdict
        assert isinstance(mdict["obj1"], MatlabOpaque)
        assert mdict["obj1"].class_alias == "SecondName"
        assert mdict["obj1"].classname == "FirstName"


@pytest.mark.parametrize("filename", files)
class TestSaveClassAlias:

    def test_class_alias(self, filename):
        """Test writing function handle simple to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["obj1"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict)
            mload = load_from_mat(temp_file_path, variable_names=["obj1"])

            assert "obj1" in mload
            assert isinstance(mload["obj1"], MatlabOpaque)
            assert mload["obj1"].class_alias == mdict["obj1"].class_alias
            assert mload["obj1"].class_alias == "SecondName"
            assert mload["obj1"].classname == mdict["obj1"].classname

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
