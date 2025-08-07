"""Convert MATLAB objects to Python compatible objects"""

from enum import Enum

import numpy as np

from matio.utils import (
    mat_to_calendarduration,
    mat_to_categorical,
    mat_to_containermap,
    mat_to_datetime,
    mat_to_dictionary,
    mat_to_duration,
    mat_to_string,
    mat_to_table,
    mat_to_timetable,
)

CLASS_TO_FUNCTION = {
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


class MatOpaque:
    """Represents a MATLAB opaque object"""

    def __init__(self, classname, type_system, properties=None):
        self.classname = classname
        self.type_system = type_system
        self.properties = properties

    def __repr__(self):
        return f"MatOpaque(classname={self.classname})"

    def __eq__(self, other):
        if isinstance(other, MatOpaque):
            return self.properties == other.properties
        return self.properties == other


def mat_to_enum(values, value_names, class_name, shapes):
    """Converts MATLAB enum to Python enum"""

    enum_class = Enum(
        class_name,
        {name: val.properties for name, val in zip(value_names, values)},
    )

    enum_members = [enum_class(val.properties) for val in values]
    return np.array(enum_members, dtype=object).reshape(shapes, order="F")
