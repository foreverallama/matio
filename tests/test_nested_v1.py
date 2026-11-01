import os
import warnings

import pytest

from matio import load_from_mat
from matio.utils.matclass import MatlabOpaque

files = [("test_nested_v1_v7.mat", "v7"), ("test_nested_v1_v73.mat", "v7.3")]


@pytest.mark.parametrize("filename, version", files)
class TestLoadNestedV1:

    def test_nested_v1(self, filename, version):
        """Test reading objects saved as nested FileWrapper__ v1 stream"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            mdict = load_from_mat(file_path, variable_names=["param"])

        assert isinstance(mdict["param"], MatlabOpaque)
        assert mdict["param"].classname == "Simulink.Parameter"

        coder_info = mdict["param"].properties["CoderInfo"]
        assert isinstance(coder_info, MatlabOpaque)
        assert coder_info.classname == "Simulink.CoderInfo"
        assert coder_info.properties["StorageClass"] == "ExportedGlobal"
