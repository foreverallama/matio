from enum import Enum

import numpy as np

from matio.utils.matclass import MatlabEnumerationArray, MatlabOpaque, OpaqueType
from matio.utils.matheaders import MCOS_MAGIC_NUMBER

ENUM_INSTANCE_DTYPE = [
    ("EnumerationInstanceTag", object),
    ("ClassName", object),
    ("ValueNames", object),
    ("Values", object),
    ("ValueIndices", object),
    ("BuiltinClassName", object),
]


def mat_to_enum(values, value_names, classname, shapes):
    """Converts MATLAB enum to Python enum"""

    enum_class = Enum(
        classname,
        {name: val.properties for name, val in zip(value_names, values)},
    )

    enum_members = [enum_class(val.properties) for val in values]
    arr = np.array(enum_members, dtype=object).reshape(shapes, order="F")
    return MatlabEnumerationArray(arr, type_system=OpaqueType.MCOS, classname=classname)


def enum_to_opaque(classname, props):
    """Converts Python enum to MATLAB enum"""

    return MatlabOpaque(
        properties=props, classname=classname, type_system=OpaqueType.MCOS
    )
