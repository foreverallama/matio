"""Utility functions for converting MATLAB containerMap and Dictionary"""

import warnings

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
            f"Only v{MAT_DICT_VERSION} MATLAB dictionaries are supported. Got v{ver}",
            UserWarning,
        )
        return props

    ks = comps[0, 0]["Key"].ravel()
    vals = comps[0, 0]["Value"].ravel()

    return list(zip(ks, vals))
