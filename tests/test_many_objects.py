import copy
import os
import tempfile

import numpy as np
import pytest

from matio import load_from_mat, save_to_mat
from matio.subsystem.subsys import MatSubsystem

versions = ["v7", "v7.3"]
NUM_OBJECTS = 200


def roundtrip(mdict, version):
    """Saves mdict to a temporary MAT-file and loads it back"""

    with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmpfile:
        temp_file_path = tmpfile.name

    try:
        save_to_mat(temp_file_path, mdict, version=version)
        return load_from_mat(temp_file_path)
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


def test_record_offsets():
    """Test cached record offsets against walking from the start, in any order"""
    nfields = 3
    counts = [0, 1, 2, 3, 0, 4, 1]
    metadata = []
    for count in counts:
        metadata += [count] + [7] * (count * nfields)
        metadata += [0] * (len(metadata) % 2)  # Padding
    metadata = np.array(metadata, dtype="<i4")

    expected = [0]
    for count in counts[:-1]:
        offset = expected[-1] + 1 + nfields * count
        expected.append(offset + offset % 2)

    subsystem = MatSubsystem("<")
    for record_id in [3, 0, 6, 1, 5, 2, 4, 6]:
        offset = subsystem.get_record_offset("nobj", metadata, record_id, nfields, 0)
        assert offset == expected[record_id]
        assert metadata[offset] == counts[record_id]


@pytest.mark.parametrize("version", versions)
class TestManyObjects:
    """Test that each of many objects gets its own property record back"""

    def test_many_datetimes(self, version):
        """Test reading many datetime objects (object property records)"""
        start = np.datetime64("2025-04-01T00:00:00", "ns")
        values = np.empty((1, NUM_OBJECTS), dtype=object)
        for i in range(NUM_OBJECTS):
            values[0, i] = np.array([[start + np.timedelta64(i, "s")]])

        mload = roundtrip({"values": values}, version)

        assert mload["values"].shape == (1, NUM_OBJECTS)
        for i, value in enumerate(mload["values"].flat):
            np.testing.assert_array_equal(
                value, np.array([[start + np.timedelta64(i, "s")]]), strict=True
            )

    def test_many_strings(self, version):
        """Test reading many string objects (saveobj property records)"""
        values = np.empty((1, NUM_OBJECTS), dtype=object)
        for i in range(NUM_OBJECTS):
            values[0, i] = np.array([[f"string {i}"]], dtype=np.dtypes.StringDType())

        mload = roundtrip({"values": values}, version)

        assert mload["values"].shape == (1, NUM_OBJECTS)
        for i, value in enumerate(mload["values"].flat):
            assert value[0, 0] == f"string {i}"

    def test_many_dynamic_properties(self, version):
        """Test reading many objects with dynamic properties (dynamic property records)"""
        file_path = os.path.join(
            os.path.dirname(__file__), "data", "test_dynamic_v7.mat"
        )
        obj = load_from_mat(file_path, variable_names=["obj"])["obj"]

        objs = np.empty((1, NUM_OBJECTS), dtype=object)
        for i in range(NUM_OBJECTS):
            objs[0, i] = copy.deepcopy(obj)
            objs[0, i].properties["Name"] = np.array([f"obj {i}"])
            dynprop = objs[0, i].properties["__dynamic_property__1"]
            dynprop.properties["Name"] = np.array([f"dynprop {i}"])

        mload = roundtrip({"objs": objs}, version)

        assert mload["objs"].shape == (1, NUM_OBJECTS)
        for i, loaded in enumerate(mload["objs"].flat):
            assert loaded.properties["Name"] == f"obj {i}"
            dynprop = loaded.properties["__dynamic_property__1"]
            assert dynprop.classname == "meta.DynamicProperty"
            assert dynprop.properties["Name"] == f"dynprop {i}"
