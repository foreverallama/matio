import os

import pytest

from matio import load_from_mat
from matio.utils.matclass import MatlabOpaque

files = ["test_figure.fig"]


@pytest.mark.parametrize("filename", files)
class TestLoadFig:

    def test_load_fig(self, filename):
        """Test reading fig file"""
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)
        mdict = load_from_mat(file_path, variable_names=None)

        assert "hgS_070000" in mdict
        assert "hgM_070000" in mdict
        assert "meta_data" in mdict
        assert isinstance(mdict["hgM_070000"][0, 0]["GraphicsObjects"], MatlabOpaque)
        assert (
            mdict["hgM_070000"][0, 0]["GraphicsObjects"].classname
            == "matlab.graphics.internal.figfile.GraphicsObjects"
        )
