"""Conversion utilities for matio"""

from .matmap import (
    containermap_to_mat,
    dictionary_to_mat,
    mat_to_containermap,
    mat_to_dictionary,
)
from .matstring import mat_to_string, string_to_mat
from .mattables import (
    categorical_to_mat,
    mat_to_categorical,
    mat_to_table,
    mat_to_timetable,
    table_to_mat,
    timetable_to_mat,
)
from .mattimes import (
    calendarduration_to_mat,
    datetime_to_mat,
    duration_to_mat,
    mat_to_calendarduration,
    mat_to_datetime,
    mat_to_duration,
)

__all__ = [
    "mat_to_containermap",
    "mat_to_dictionary",
    "mat_to_string",
    "mat_to_categorical",
    "mat_to_table",
    "mat_to_timetable",
    "mat_to_calendarduration",
    "mat_to_datetime",
    "mat_to_duration",
    "string_to_mat",
    "datetime_to_mat",
    "duration_to_mat",
    "calendarduration_to_mat",
    "dictionary_to_mat",
    "containermap_to_mat",
    "table_to_mat",
    "timetable_to_mat",
    "categorical_to_mat",
]
