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
            "mat_to_string: String saved from a different MAT-file version. Returning raw data",
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


def string_to_mat(arr):
    """Converts numpy string array to MATLAB string format as uint64 array"""

    if not isinstance(arr, np.ndarray):
        try:
            arr = np.str_(arr)
        except Exception as e:
            raise TypeError(f"Expected numpy.ndarray, got {type(arr)}") from e

    # if arr.ndim == 1:
    #     if oned_as == "row":
    #         arr = arr.reshape(1, -1)
    #     elif oned_as == "col":
    #         arr = arr.reshape(-1, 1)

    ndims = arr.ndim
    shape = arr.shape
    encoding = "utf-16-le" if np.little_endian else "utf-16-be"

    utf16_data_list = []
    char_counts = []

    for s in arr.ravel():
        utf16_arr = np.frombuffer(s.encode(encoding), dtype=np.uint16)
        utf16_data_list.append(utf16_arr)
        char_counts.append(len(utf16_arr))
    all_utf16 = np.hstack(utf16_data_list)

    # Pad
    pad_len = (-len(all_utf16)) % 4
    if pad_len > 0:
        all_utf16 = np.hstack([all_utf16, np.zeros(pad_len, dtype=np.uint16)])
    utf16_data_uint64 = all_utf16.view(np.uint64)

    # Header
    header_list = [MAT_STRING_VERSION, ndims] + list(shape) + char_counts
    header = np.array(header_list, dtype=np.uint64)
    prop = np.hstack([header, utf16_data_uint64]).reshape(1, -1)

    prop_map = {
        "any": prop,
    }

    return prop_map
