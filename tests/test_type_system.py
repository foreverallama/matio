import os

import pytest

from matio import load_from_mat
from matio.utils.matclass import MatlabOpaque, MatReadWarning

files = [("test_type_systems_v7.mat", "v7"), ("test_type_systems_v73.mat", "v7.3")]


@pytest.mark.parametrize("filename, version", files)
class TestLoadTypeSystems:

    def test_type_system(self, filename, version):
        """Test reading different type systems from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        with pytest.warns(MatReadWarning):
            mdict = load_from_mat(file_path, variable_names=["javatype", "handletype"])

        assert "javatype" in mdict
        assert "handletype" in mdict

        assert isinstance(mdict["javatype"], MatlabOpaque)
        assert isinstance(mdict["handletype"], MatlabOpaque)

        assert mdict["javatype"].classname == "java.lang.String"
        assert mdict["handletype"].classname == "COM.Excel_Application"
        assert mdict["javatype"].type_system == "java"
        assert mdict["handletype"].type_system == "handle"
