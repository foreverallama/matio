"""Utility functions for converting MATLAB containerMap and Dictionary"""

import warnings

import numpy as np

MAT_DICT_VERSION = 1


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

    return result


def mat_to_dictionary(props, **_kwargs):
    """Converts MATLAB dictionary to Python list of tuples"""
    # List of tuples as Key-Value pairs can be any datatypes

    comps = props.get("data", None)
    if comps is None:
        return props

    ver = int(comps[0, 0]["Version"].item())
    if ver != MAT_DICT_VERSION:
        warnings.warn(
            f"mat_to_dictionary: Only v{MAT_DICT_VERSION} MATLAB dictionaries are supported. Got v{ver}",
            UserWarning,
        )
        return props

    ks = comps[0, 0]["Key"]
    vals = comps[0, 0]["Value"]

    return (ks, vals)


def dictionary_to_mat(props):
    """Converts a Python dictionary to MATLAB dictionary"""
    if not (isinstance(props, tuple) and len(props) == 2):
        raise TypeError("Expected tuple of (key, value)")

    keys, values = props

    def is_valid_array(obj):
        return (
            isinstance(obj, np.ndarray) or getattr(obj, "classname", None) == "string"
        )

    if not is_valid_array(keys) or not is_valid_array(values):
        raise TypeError(
            "Keys must be a numpy array or MatioOpaque with classname='string'"
        )

    dtype = [
        ("Version", object),
        ("IsKeyCombined", object),
        ("IsValueCombined", object),
        ("Key", object),
        ("Value", object),
    ]
    data_arr = np.empty((1, 1), dtype=dtype)

    data_arr["Key"][0, 0] = keys
    data_arr["Value"][0, 0] = values
    data_arr["Version"][0, 0] = np.uint64(MAT_DICT_VERSION)
    data_arr["IsKeyCombined"][0, 0] = np.bool_(True)
    data_arr["IsValueCombined"][0, 0] = np.bool_(True)

    prop_map = {
        "data": data_arr,
    }

    return prop_map


def containermap_to_mat(props):
    """Converts a Python dictionary to MATLAB container.Map"""
    if not isinstance(props, dict):
        raise TypeError(f"Expected dict, got {type(props)}")
    if "serialization" in props:
        warnings.warn(
            "containermap_to_mat: Key 'Serialization' was found in the map. This clashes with an expected MATLAB "
            "keyword and will be treated as a raw property map."
        )
        return props

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

    value_dtypes = {v.dtype for v in vals if isinstance(v, np.ndarray)}
    if len(value_dtypes) == 1:
        uniformity = True
        dtype_map = {
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
        }
        value_type = dtype_map.get(next(iter(value_dtypes)), "any")
    else:
        uniformity = np.bool_(False)
        value_type = "any"

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
