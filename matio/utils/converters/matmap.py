"""Utility functions for converting MATLAB containerMap and Dictionary"""

import warnings

import numpy as np

from matio.utils.matclass import MatlabContainerMap, MatlabDictionary

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


def mat_to_dictionary(props, **_kwargs):
    """Converts MATLAB dictionary to MatlabDictionary"""

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

    if "Unconfigured" in comps.dtype.names:
        return MatlabDictionary()

    arrs = []
    for name in ("Key", "Value"):
        arr = comps[0, 0][name]
        # If not combined, each element is saved as a 1x1 array in a cell.
        # Only plain arrays concatenate without losing their MATLAB class
        if not comps[0, 0][f"Is{name}Combined"].item():
            if any(type(x) is not np.ndarray for x in arr.flat):
                raise TypeError(f"{name} class is not supported")
            arr = np.concatenate(arr.ravel())
        # MATLAB saves N x 1, empty datetimes are read as 0 x 0
        arrs.append(arr.reshape(-1, 1))

    return MatlabDictionary(*arrs)


def dictionary_to_mat(props):
    """Converts MatlabDictionary to MATLAB dictionary"""

    if props.keys is None:
        data_arr = np.empty(
            (1, 1), dtype=[("Version", object), ("Unconfigured", object)]
        )
        data_arr["Version"][0, 0] = np.uint64(MAT_DICT_VERSION)
        data_arr["Unconfigured"][0, 0] = np.bool_(True)
        return {"data": data_arr}

    dtype = [
        ("Version", object),
        ("IsKeyCombined", object),
        ("IsValueCombined", object),
        ("Key", object),
        ("Value", object),
    ]
    data_arr = np.empty((1, 1), dtype=dtype)

    # MATLAB dictionaries hold text as string. Char arrays load as unconfigured
    keys, values = (
        x.astype(np.dtypes.StringDType()) if x.dtype.kind == "U" else x
        for x in (props.keys, props.values)
    )

    data_arr["Key"][0, 0] = keys
    data_arr["Value"][0, 0] = values
    data_arr["Version"][0, 0] = np.uint64(MAT_DICT_VERSION)
    data_arr["IsKeyCombined"][0, 0] = np.bool_(True)
    data_arr["IsValueCombined"][0, 0] = np.bool_(True)

    prop_map = {
        "data": data_arr,
    }

    return prop_map


def detect_dtype_uniformity(vals):
    """Set uniformity and valueType for container.Map"""

    value_dtypes = set()
    val_type = None
    for v in vals:
        # Uniform values must be char vectors or numeric/logical scalars
        if not isinstance(v, np.ndarray) or (
            v.dtype.kind not in ("U", "S") and v.size != 1
        ):
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
        key_type = "int64" if min(keys) < 0 else "uint64"
        # Defaulting to highest precision as Python doesn't differentiate
        # MATLAB saturates negative keys to 0 for uint64
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
