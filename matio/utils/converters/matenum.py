import enum
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


class MatlabEnum(Enum):
    """Enum whose members return the properties of the MATLAB members as value"""

    def __new__(cls, name, properties):
        member = object.__new__(cls)
        # Members are told apart by name, MATLAB allows equal properties
        member._value_ = name
        member._properties = properties
        return member

    # enum.property keeps working when a member is called "value"
    @enum.property
    def value(self):
        return self._properties

    @classmethod
    def _missing_(cls, value):
        # Look members up by their properties, like an Enum with these values
        for member in cls:
            if member._properties == value:
                return member
        return None

    def __repr__(self):
        return f"<{type(self).__name__}.{self._name_}: {self._properties!r}>"


def mat_to_enum(values, value_names, classname, shapes):
    """Converts MATLAB enum to Python enum"""

    if not values:
        # Enumerations without properties store only the member names
        enum_class = Enum(classname, list(dict.fromkeys(value_names)))
    else:
        props = {name: val.properties for name, val in zip(value_names, values)}
        try:
            enum_class = Enum(classname, props)
            # Members with equal properties would become aliases
            distinct = len(enum_class) == len(props)
        except ValueError:  # == cannot compare properties such as vectors
            distinct = False
        if not distinct:
            enum_class = MatlabEnum(
                classname, {name: (name, p) for name, p in props.items()}
            )
    enum_members = [enum_class[name] for name in value_names]
    arr = np.array(enum_members, dtype=object).reshape(shapes, order="F")
    return MatlabEnumerationArray(arr, type_system=OpaqueType.MCOS, classname=classname)


def enum_to_opaque(classname, props):
    """Converts Python enum to MATLAB enum"""

    return MatlabOpaque(
        properties=props, classname=classname, type_system=OpaqueType.MCOS
    )
