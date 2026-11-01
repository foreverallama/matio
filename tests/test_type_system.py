import os

import numpy as np
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

        handle_properties = mdict["handletype"].properties
        assert handle_properties.shape == (139, 1)
        assert isinstance(handle_properties[0, 0], np.ndarray)
        assert handle_properties[0, 0].shape == (4520, 1)
        assert handle_properties[0, 0].dtype == np.uint8


def test_java_property():
    """Test reading a Java object stored in a property of a classdef object"""
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    mdict_v7 = load_from_mat(os.path.join(data_dir, "test_java_property_v7.mat"))
    mdict_v73 = load_from_mat(os.path.join(data_dir, "test_java_property_v73.mat"))

    # The Java object stays raw uint8 data, in v7.3 as in v7
    java_v7 = mdict_v7["obj"].properties["a"]
    assert java_v7.dtype == np.uint8
    np.testing.assert_array_equal(
        mdict_v73["obj"].properties["a"], java_v7, strict=True
    )
    np.testing.assert_array_equal(mdict_v73["after"], np.array([[42.0]]), strict=True)
