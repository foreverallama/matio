from enum import Enum

import numpy as np

from .converters.matstring import mat_to_string
from .converters.mattables import mat_to_categorical, mat_to_table, mat_to_timetable
from .converters.mattimes import (
    mat_to_calendarduration,
    mat_to_datetime,
    mat_to_duration,
)
from .matclass import MatlabEnumerationArray, OpaqueType

matlab_classdef_types = [
    "calendarDuration",
    "categorical",
    "containers.Map",
    "datetime",
    "dictionary",
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

# PY_TO_MAT = {
#     "calendarDuration": calendarduration_to_mat,
#     "categorical": categorical_to_mat,
#     "containers.Map": containermap_to_mat,
#     "datetime": datetime_to_mat,
#     "dictionary": dictionary_to_mat,
#     "duration": duration_to_mat,
#     "string": string_to_mat,
#     "table": table_to_mat,
#     "timetable": timetable_to_mat,
# }


def convert_mat_to_py(props, classname, **kwargs):
    """Converts a MATLAB object to a Python object"""
    convert_func = MAT_TO_PY.get(classname)

    return convert_func(
        props,
        byte_order=kwargs.get("byte_order", None),
        add_table_attrs=kwargs.get("add_table_attrs", None),
    )


def mat_to_enum(values, value_names, classname, shapes):
    """Converts MATLAB enum to Python enum"""

    enum_class = Enum(
        classname,
        {name: val.properties for name, val in zip(value_names, values)},
    )

    enum_members = [enum_class(val.properties) for val in values]
    arr = np.array(enum_members, dtype=object).reshape(shapes, order="F")
    return MatlabEnumerationArray(arr, type_system=OpaqueType.MCOS, classname=classname)
