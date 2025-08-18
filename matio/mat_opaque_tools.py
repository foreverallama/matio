"""Convert MATLAB objects to Python compatible objects"""

import warnings
from enum import Enum

import numpy as np
from pandas import Categorical, DataFrame

from matio.utils import (
    calendarduration_to_mat,
    categorical_to_mat,
    containermap_to_mat,
    datetime_to_mat,
    dictionary_to_mat,
    duration_to_mat,
    mat_to_calendarduration,
    mat_to_categorical,
    mat_to_containermap,
    mat_to_datetime,
    mat_to_dictionary,
    mat_to_duration,
    mat_to_string,
    mat_to_table,
    mat_to_timetable,
    string_to_mat,
    table_to_mat,
    timetable_to_mat,
)

MAT_TO_PY = {
    "calendarDuration": mat_to_calendarduration,
    "categorical": mat_to_categorical,
    "containers.Map": mat_to_containermap,
    "datetime": mat_to_datetime,
    "dictionary": mat_to_dictionary,
    "duration": mat_to_duration,
    "string": mat_to_string,
    "table": mat_to_table,
    "timetable": mat_to_timetable,
}

PY_TO_MAT = {
    "calendarDuration": calendarduration_to_mat,
    "categorical": categorical_to_mat,
    "containers.Map": containermap_to_mat,
    "datetime": datetime_to_mat,
    "dictionary": dictionary_to_mat,
    "duration": duration_to_mat,
    "string": string_to_mat,
    "table": table_to_mat,
    "timetable": timetable_to_mat,
}


class MatioOpaque:
    """Represents a MATLAB opaque object"""

    def __init__(self, properties=None, classname=None, type_system="MCOS"):

        self.classname = classname
        self.type_system = type_system
        self.properties = properties
        self.is_array = False

    def __repr__(self):
        return f"MatioOpaque(classname={self.classname})"

    def __eq__(self, other):
        if isinstance(other, MatioOpaque):
            return self.properties == other.properties
        return self.properties == other


def convert_mat_to_py(obj, **kwargs):
    """Converts a MATLAB object to a Python object"""
    convert_func = MAT_TO_PY.get(obj.classname)
    if convert_func is not None:
        return convert_func(
            obj.properties,
            byte_order=kwargs.get("byte_order", None),
            add_table_attrs=kwargs.get("add_table_attrs", None),
        )
    return obj


def convert_py_to_mat(properties, classname):
    """Convert a Python object to a MATLAB object"""

    convert_func = PY_TO_MAT.get(classname)
    if convert_func is None:
        warnings.warn(
            f"convert_py_to_mat: Conversion of {type(properties)} into MATLAB type "
            f"{classname} is not yet implemented. This will be skipped"
        )
        return {}

    return convert_func(properties)


def mat_to_enum(values, value_names, class_name, shapes):
    """Converts MATLAB enum to Python enum"""

    enum_class = Enum(
        class_name,
        {name: val.properties for name, val in zip(value_names, values)},
    )

    enum_members = [enum_class(val.properties) for val in values]
    return np.array(enum_members, dtype=object).reshape(shapes, order="F")


def guess_class_name(properties):
    """Guess the class name based on properties"""

    if properties is None:
        return properties

    classname = None
    if isinstance(properties, DataFrame):
        if isinstance(properties.index.values, np.ndarray) and (
            np.issubdtype(properties.index.values.dtype, np.datetime64)
            or np.issubdtype(properties.index.values.dtype, np.timedelta64)
        ):
            classname = "timetable"
        else:
            classname = "table"
    elif isinstance(properties, Categorical):
        classname = "categorical"
    elif isinstance(properties, dict):
        classname = "containers.Map"
    elif isinstance(properties, tuple):
        classname = "dictionary"
    elif isinstance(properties, np.ndarray):
        if set(properties.dtype.names) == {"months", "days", "millis"}:
            classname = "calendarDuration"
        elif np.issubdtype(properties.dtype, np.datetime64):
            classname = "datetime"
        elif np.issubdtype(properties.dtype, np.timedelta64):
            classname = "duration"
        elif properties.dtype.kind in ("U", "S"):
            classname = "string"

    if classname is None:
        raise ValueError(
            f"Unable to determine MATLAB class equivalent for properties of "
            f"type {type(properties)}. Provide it explicitly."
        )

    return classname
