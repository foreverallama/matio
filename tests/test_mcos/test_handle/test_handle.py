import os

from matio import load_from_mat
from matio.mat_opaque_tools import MatioOpaque


def test_handle_load_v7():
    file_path_v7 = os.path.join(os.path.dirname(__file__), "test_handle_v7.mat")
    prop_names = ["Name", "Next"]

    matdict = load_from_mat(file_path_v7, raw_data=False)

    assert isinstance(matdict["obj1"], MatioOpaque)
    assert isinstance(matdict["obj2"], MatioOpaque)

    assert matdict["obj1"].classname == "Node"
    assert matdict["obj2"].classname == "Node"

    assert set(matdict["obj1"].properties.keys()) == set(prop_names)
    assert set(matdict["obj2"].properties.keys()) == set(prop_names)

    assert matdict["obj1"].properties["Next"] is matdict["obj2"]
    assert matdict["obj2"].properties["Next"] is matdict["obj1"]


def test_handle_load_v73():
    file_path_v73 = os.path.join(os.path.dirname(__file__), "test_handle_v73.mat")
    prop_names = ["Name", "Next"]

    matdict = load_from_mat(file_path_v73, raw_data=False)

    assert isinstance(matdict["obj1"], MatioOpaque)
    assert isinstance(matdict["obj2"], MatioOpaque)

    assert matdict["obj1"].classname == "Node"
    assert matdict["obj2"].classname == "Node"

    assert set(matdict["obj1"].properties.keys()) == set(prop_names)
    assert set(matdict["obj2"].properties.keys()) == set(prop_names)

    assert matdict["obj1"].properties["Next"] is matdict["obj2"]
    assert matdict["obj2"].properties["Next"] is matdict["obj1"]
