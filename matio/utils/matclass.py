from collections import UserDict
from enum import IntEnum, StrEnum

import numpy as np

MCOS_SUBSYSTEM_CLASS = "FileWrapper__"
DYNAMIC_PROPERTY_PREFIX = "__dynamic_property__"
ENUMERATION_INSTANCE_TAG = "EnumerationInstanceTag"


class IntegerDecodingHint(IntEnum):
    """MATLAB integer arrays typedefs"""

    NO_INT_HINT = 0
    LOGICAL_HINT = 1
    UTF16_HINT = 2


class ObjectDecodingHint(IntEnum):
    """MATLAB object arrays typdefs"""

    NO_OBJ_HINT = 0
    FUNCTION_HINT = 1
    OBJECT_HINT = 2
    OPAQUE_HINT = 3


class PropertyType(IntEnum):
    """Enumeration for property types in FileWrapper__ data"""

    MATLAB_ENUMERATION = 0  # A string which names some internal enumeration probably
    PROPERTY_VALUE = 1
    INTEGER_VALUE = (
        2  # Mostly logical but I've seen integer values as well in fig files
    )


class OpaqueType(StrEnum):
    """Enumeration for different opaque object type systems"""

    MCOS = "MCOS"
    JAVA = "java"
    HANDLE = "handle"


class MatlabClasses(StrEnum):
    """Enumeration for different MATLAB class names"""

    CHAR = "char"
    STRUCT = "struct"
    CELL = "cell"
    FUNCTION = "function_handle"
    LOGICAL = "logical"
    DOUBLE = "double"
    SINGLE = "single"
    EMPTY = "canonical empty"


class MatReadError(Exception):
    """Exception indicating a read issue."""


class MatWriteError(Exception):
    """Exception indicating a write issue."""


class MatReadWarning(UserWarning):
    """Warning class for read issues."""


class MatWriteWarning(UserWarning):
    """Warning class for write issues."""


class MatConvertWarning(UserWarning):
    """Warning class for conversion issues."""


class MatConvertError(Exception):
    """Exception indicating a conversion issue."""


class MatlabObject(np.ndarray):
    """Subclass for a MATLAB object."""

    def __new__(cls, input_array, classname=None):
        """Create a new instance of MatlabObject"""
        obj = np.asarray(input_array).view(cls)
        obj.classname = classname
        return obj

    def __array_finalize__(self, obj):
        """Finalize the array, copying the classname."""
        self.classname = getattr(obj, "classname", None)


class MatlabFunction(np.ndarray):
    """Subclass for a MATLAB function."""

    def __new__(cls, input_array):
        """Create a new instance of MatlabFunction"""
        obj = np.asarray(input_array).view(cls)
        return obj


class MatlabOpaque:
    """Subclass for a MATLAB opaque object."""

    def __init__(
        self, properties, classname, type_system=OpaqueType.MCOS, class_alias=None
    ):
        """Create a new instance of MatlabOpaque"""
        self.properties = properties
        self.classname = classname
        self.type_system = type_system
        self.class_alias = class_alias

    def __repr__(self):
        """String representation of the object"""
        if isinstance(self.properties, tuple):
            shape = self.properties
        else:
            shape = (1, 1)
        return f"MatlabOpaque(classname={self.classname})"


class MatlabOpaqueArray(np.ndarray):
    """Subclass for a MATLAB Opaque array."""

    def __new__(cls, input_array, classname, type_system=OpaqueType.MCOS):
        """Create a new instance of MatlabMCOSArray"""
        obj = np.asarray(input_array).view(cls)
        obj.classname = classname
        obj.type_system = type_system
        return obj

    def __array_finalize__(self, obj):
        """Finalize the array, copying the classnames."""
        self.type_system = getattr(obj, "type_system", None)
        self.classname = getattr(obj, "classname", None)


class MatlabEnumerationArray(np.ndarray):
    """Subclass for a MATLAB Enumeration array."""

    def __new__(cls, input_array, classname, type_system=OpaqueType.MCOS):
        """Create a new instance of MatlabEnumerationArray"""
        obj = np.asarray(input_array).view(cls)
        obj.classname = classname
        obj.type_system = type_system
        return obj

    def __array_finalize__(self, obj):
        """Finalize the array, copying the classnames."""
        self.type_system = getattr(obj, "type_system", None)
        self.classname = getattr(obj, "classname", None)


class MatlabOpaqueProperty(np.ndarray):
    """Subclass for special Matlab Opaque properties."""

    def __new__(cls, input_array, ptype):
        """Create a new instance of MatlabObject"""
        obj = np.asarray(input_array).view(cls)
        obj.ptype = ptype
        return obj

    def __array_finalize__(self, obj):
        """Finalize the array, copying the classname."""
        self.ptype = getattr(obj, "ptype", None)


class MatlabContainerMap(UserDict):
    """Class to represent a MATLAB containers.Map object."""

    classname = "container.Map"


class EmptyMatStruct(np.ndarray):
    """Class to represent an empty MATLAB struct."""

    def __new__(cls, input_array):
        """Create a new instance of EmptyMatStruct"""
        obj = np.asarray(input_array).view(cls)
        return obj


class MatlabCanonicalEmpty:
    """Class to represent a canonical empty MATLAB array."""
