import os
import tempfile

import numpy as np
import pytest

from matio import load_from_mat, save_to_mat
from matio.utils.matclass import MatlabOpaque, MatlabOpaqueProperty

files = [("test_dynamic_v7.mat", "v7"), ("test_dynamic_v73.mat", "v7.3")]
namespace = "TestClasses"


@pytest.mark.parametrize("filename, version", files)
class TestLoadMatlabDynamic:

    def test_dynamicprop(self, filename, version):
        """Test reading dynamic property from MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["obj"])

        assert "obj" in mdict
        assert isinstance(mdict["obj"], MatlabOpaque)
        assert mdict["obj"].classname == f"{namespace}.BasicDynamic"
        assert mdict["obj"].type_system == "MCOS"

        assert "__dynamic_property__1" in mdict["obj"].properties
        assert isinstance(
            mdict["obj"].properties["__dynamic_property__1"], MatlabOpaque
        )
        assert (
            mdict["obj"].properties["__dynamic_property__1"].classname
            == "meta.DynamicProperty"
        )
        assert mdict["obj"].properties["__dynamic_property__1"].type_system == "MCOS"


@pytest.mark.parametrize("filename, version", files)
class TestSaveMatlabDynamic:

    def test_dynamicprops(self, filename, version):
        """Test writing dynamic properties to MAT-file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=["obj"])

        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
            temp_file_path = tmpfile.name

        try:
            save_to_mat(temp_file_path, mdict, version=version)
            mload = load_from_mat(temp_file_path, variable_names=["obj"])

            assert "obj" in mload
            assert isinstance(mload["obj"], MatlabOpaque)
            assert mload["obj"].classname == mdict["obj"].classname
            assert mload["obj"].type_system == mdict["obj"].type_system

            assert "__dynamic_property__1" in mload["obj"].properties
            assert isinstance(
                mload["obj"].properties["__dynamic_property__1"], MatlabOpaque
            )
            assert (
                mload["obj"].properties["__dynamic_property__1"].classname
                == mdict["obj"].properties["__dynamic_property__1"].classname
            )
            assert (
                mload["obj"].properties["__dynamic_property__1"].type_system
                == mdict["obj"].properties["__dynamic_property__1"].type_system
            )

            for key, val in mdict["obj"].properties.items():
                if isinstance(val, MatlabOpaque):
                    assert key in mload["obj"].properties
                    assert isinstance(mload["obj"].properties[key], MatlabOpaque)
                    assert mload["obj"].properties[key].classname == val.classname
                    assert mload["obj"].properties[key].type_system == val.type_system
                elif isinstance(val, MatlabOpaqueProperty):
                    assert key in mload["obj"].properties
                    assert mload["obj"].properties[key].ptype == val.ptype
                    np.testing.assert_array_equal(
                        mload["obj"].properties[key], val, strict=True
                    )
                elif isinstance(val, np.ndarray):
                    assert key in mload["obj"].properties
                    np.testing.assert_array_equal(
                        mload["obj"].properties[key], val, strict=True
                    )

        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
