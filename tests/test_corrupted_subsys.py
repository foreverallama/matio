import os

import numpy as np
import pytest

from matio import load_from_mat
from matio.utils.matclass import MatlabOpaque, MatReadWarning
from matio.utils.matheaders import MCOS_MAGIC_NUMBER


class TestLoadCorrupted:

    def test_load_corrupted_subsystem(self):
        """Test reading fig file"""

        filename = "test_corrupted_subsystem.mat"
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)

        with pytest.warns(MatReadWarning, match="Opaque objects will be skipped"):
            mdict = load_from_mat(file_path, variable_names=None)

        assert "var" in mdict
        assert isinstance(mdict["var"], np.ndarray)
        assert mdict["var"][0, 0] == MCOS_MAGIC_NUMBER

    def test_load_corrupted_mcos_object_metadata(self):
        """Test reading fig file"""

        filename = "test_corrupted_mcos_object_metadata.mat"
        file_path = os.path.join(os.path.dirname(__file__), "data", filename)

        with pytest.warns(MatReadWarning, match="Failed to load object instance"):
            mdict = load_from_mat(file_path, variable_names=None)

        assert "var" in mdict
        assert isinstance(mdict["var"], MatlabOpaque)
        assert isinstance(mdict["var"].properties, np.ndarray)
        assert mdict["var"].properties[0, 0] == MCOS_MAGIC_NUMBER
