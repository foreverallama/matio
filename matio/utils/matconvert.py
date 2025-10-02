import warnings
from enum import Enum

import numpy as np
import pandas as pd

from matio.utils.converters.matstring import mat_to_string, string_to_mat
from matio.utils.converters.mattables import (
    categorical_to_mat,
    mat_to_categorical,
    mat_to_table,
    mat_to_timetable,
    table_to_mat,
    timetable_to_mat,
)
from matio.utils.converters.mattimes import (
    caldur_dtype,
    calendarduration_to_mat,
    datetime_to_mat,
    duration_to_mat,
    mat_to_calendarduration,
    mat_to_datetime,
    mat_to_duration,
)
from matio.utils.matclass import (
    MatlabContainerMap,
    MatlabEnumerationArray,
    MatlabOpaque,
    OpaqueType,
)

matlab_saveobj_ret_types = ["string", "timetable"]

matlab_classdef_types = [
    "calendarDuration",
    "categorical",
    # "containers.Map",
    "datetime",
    # "dictionary",
    "duration",
    "string",
    "table",
    "timetable",
]

MAT_TO_PY = {
    "calendarDuration": mat_to_calendarduration,
    "categorical": mat_to_categorical,
    # "containers.Map": mat_to_containermap,
    "datetime": mat_to_datetime,
    # "dictionary": mat_to_dictionary,
    "duration": mat_to_duration,
    "string": mat_to_string,
    "table": mat_to_table,
    "timetable": mat_to_timetable,
}

PY_TO_MAT = {
    "calendarDuration": calendarduration_to_mat,
    "categorical": categorical_to_mat,
    # "containers.Map": containermap_to_mat,
    "datetime": datetime_to_mat,
    # "dictionary": dictionary_to_mat,
    "duration": duration_to_mat,
    "string": string_to_mat,
    "table": table_to_mat,
    "timetable": timetable_to_mat,
}


def convert_mat_to_py(props, classname, **kwargs):
    """Converts a MATLAB object to a Python object"""
    convert_func = MAT_TO_PY.get(classname)

    return convert_func(
        props,
        byte_order=kwargs.get("byte_order", None),
        add_table_attrs=kwargs.get("add_table_attrs", None),
    )


def guess_type_system(classname):
    """Gets the type system for the given class name."""

    # Possibly, MATLAB decodes internally based on some object attribute
    # This function is just guess work as of now
    # Used for decoding HDF5 files
    if classname.startswith("java.") or classname.startswith("com."):
        type_system = "java"
    elif classname.startswith("COM."):
        type_system = "handle"
    else:
        type_system = "MCOS"

    return type_system


def guess_class_name(data):
    """Guess the MATLAB class name for a given Python object"""

    if isinstance(data, pd.DataFrame):
        if pd.api.types.is_datetime64_any_dtype(
            data.index
        ) or pd.api.types.is_timedelta64_dtype(data.index):
            return "timetable"
        else:
            return "table"
    elif isinstance(data, pd.Categorical):
        return "categorical"
    elif isinstance(data, pd.Series):
        raise NotImplementedError("pandas.Series to MATLAB object not yet supported")
    elif isinstance(data, MatlabContainerMap):
        raise NotImplementedError(
            "MatlabContainerMap to MATLAB containers.Map not yet supported"
        )
    elif isinstance(data, MatlabEnumerationArray):
        raise NotImplementedError("MatlabEnumerationArray is not yet supported")
    elif isinstance(data, (np.ndarray, np.generic)):
        if data.dtype.kind == "T":
            return "string"
        elif data.dtype.kind == "m":
            return "duration"
        elif data.dtype.kind == "M":
            return "datetime"
        elif data.dtype == caldur_dtype:
            return "calendarDuration"

    return None


def convert_py_to_mat(data, classname):
    """Convert a Python object to a MATLAB object"""

    convert_func = PY_TO_MAT.get(classname)
    type_system = guess_type_system(classname)
    prop_map = convert_func(data)

    obj = MatlabOpaque(
        prop_map,
        type_system=type_system,
        classname=classname,
    )

    return obj


def mat_to_enum(values, value_names, classname, shapes):
    """Converts MATLAB enum to Python enum"""

    enum_class = Enum(
        classname,
        {name: val.properties for name, val in zip(value_names, values)},
    )

    enum_members = [enum_class(val.properties) for val in values]
    arr = np.array(enum_members, dtype=object).reshape(shapes, order="F")
    return MatlabEnumerationArray(arr, type_system=OpaqueType.MCOS, classname=classname)
