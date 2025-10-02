"""Utility functions for converting MATLAB containerMap and Dictionary"""

import warnings

import numpy as np

from matio.utils.matclass import MatlabContainerMap

MAT_DICT_VERSION = 1

map_dtype_map = {
    np.dtype("float64"): "double",
    np.dtype("float32"): "single",
    np.dtype("bool"): "logical",
    np.dtype("int8"): "int8",
    np.dtype("uint8"): "uint8",
    np.dtype("int16"): "int16",
    np.dtype("uint16"): "uint16",
    np.dtype("int32"): "int32",
    np.dtype("uint32"): "uint32",
    np.dtype("int64"): "int64",
    np.dtype("uint64"): "uint64",
    np.dtype(np.str_): "char",
}


def mat_to_containermap(props, **_kwargs):
    """Converts MATLAB container.Map to Python dictionary"""
    comps = props.get("serialization", None)
    if comps is None:
        return props

    ks = comps[0, 0]["keys"]
    vals = comps[0, 0]["values"]

    result = {}
    for i in range(ks.shape[1]):
        key = ks[0, i].item()
        val = vals[0, i]
        result[key] = val

    return MatlabContainerMap(result)


# # def mat_to_dictionary(props, **_kwargs):
# #     """Converts MATLAB dictionary to Python list of tuples"""
# #     # List of tuples as Key-Value pairs can be any datatypes

# #     comps = props.get("data", None)
# #     if comps is None:
# #         return props

# #     ver = int(comps[0, 0]["Version"].item())
# #     if ver != MAT_DICT_VERSION:
# #         warnings.warn(
# #             f"mat_to_dictionary: Only v{MAT_DICT_VERSION} MATLAB dictionaries are supported. Got v{ver}",
# #             UserWarning,
# #         )
# #         return props

# #     ks = comps[0, 0]["Key"]
# #     vals = comps[0, 0]["Value"]

# #     return (ks, vals)


# def dictionary_to_mat(props):
#     """Converts a Python dictionary to MATLAB dictionary"""
#     if not (isinstance(props, tuple) and len(props) == 2):
#         raise TypeError("Expected tuple of (key, value)")

#     keys, values = props

#     def is_valid_array(obj):
#         return (
#             isinstance(obj, np.ndarray) or getattr(obj, "classname", None) == "string"
#         )

#     if not is_valid_array(keys) or not is_valid_array(values):
#         raise TypeError(
#             "Keys must be a numpy array or MatioOpaque with classname='string'"
#         )

#     dtype = [
#         ("Version", object),
#         ("IsKeyCombined", object),
#         ("IsValueCombined", object),
#         ("Key", object),
#         ("Value", object),
#     ]
#     data_arr = np.empty((1, 1), dtype=dtype)

#     data_arr["Key"][0, 0] = keys
#     data_arr["Value"][0, 0] = values
#     data_arr["Version"][0, 0] = np.uint64(MAT_DICT_VERSION)
#     data_arr["IsKeyCombined"][0, 0] = np.bool_(True)
#     data_arr["IsValueCombined"][0, 0] = np.bool_(True)

#     prop_map = {
#         "data": data_arr,
#     }

#     return prop_map


def detect_dtype_uniformity(vals):
    """Set uniformity and valueType for container.Map"""

    value_dtypes = set()
    val_type = None
    for v in vals:
        if not isinstance(v, np.ndarray):
            uniformity = np.bool_(False)
            val_type = "any"
            break
        if v.dtype.kind in ("U", "S"):
            value_dtypes.add(np.dtype(np.str_))
        else:
            value_dtypes.add(v.dtype)

    if val_type is None:
        if len(value_dtypes) == 1:
            uniformity = True
            val_type = map_dtype_map.get(next(iter(value_dtypes)), "any")
        else:
            uniformity = np.bool_(False)
            val_type = "any"

    return uniformity, val_type


def containermap_to_mat(props):
    """Converts a Python dictionary to MATLAB container.Map"""

    keys = list(props.keys())
    vals = list(props.values())

    val_arr = np.empty((1, len(vals)), dtype=object)
    keys_arr = np.empty((1, len(keys)), dtype=object)
    val_arr[0, :] = vals
    keys_arr[0, :] = keys

    if all(isinstance(k, str) for k in keys):
        key_type = "char"
    elif all(isinstance(k, int) for k in keys):
        key_type = "uint64"
        # Defaulting to highest precision as Python doesn't differentiate
    else:
        key_type = "double"

    uniformity, value_type = detect_dtype_uniformity(vals)

    ser_dtype = [
        ("keys", object),
        ("values", object),
        ("uniformity", object),
        ("keyType", object),
        ("valueType", object),
    ]
    serialization = np.empty((1, 1), dtype=ser_dtype)
    serialization[0, 0] = (keys_arr, val_arr, uniformity, key_type, value_type)
    prop_map = {
        "serialization": serialization,
    }
    return prop_map
