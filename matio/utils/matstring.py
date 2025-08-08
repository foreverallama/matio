"""Utility functions for convertin MATLAB strings"""

import warnings

import numpy as np

MAT_STRING_VERSION = 1


def mat_to_string(props, byte_order, **_kwargs):
    """Converts MATLAB string to numpy string array"""

    data = props.get("any", np.empty((0, 0), dtype=np.str_))
    if data.size == 0:
        return np.array([[]], dtype=np.str_)

    if data[0, 0] != MAT_STRING_VERSION:
        warnings.warn(
            "String saved from a different MAT-file version. Returning raw data",
            UserWarning,
        )
        return props[0, 0].get("any")

    ndims = data[0, 1]
    shape = data[0, 2 : 2 + ndims]
    num_strings = np.prod(shape)
    char_counts = data[0, 2 + ndims : 2 + ndims + num_strings]
    byte_data = data[0, 2 + ndims + num_strings :].tobytes()

    strings = []
    pos = 0
    encoding = "utf-16-le" if byte_order[0] == "<" else "utf-16-be"
    for char_count in char_counts:
        byte_length = char_count * 2  # UTF-16 encoding
        extracted_string = byte_data[pos : pos + byte_length].decode(encoding)
        strings.append(np.str_(extracted_string))
        pos += byte_length

    return np.reshape(strings, shape, order="F")
