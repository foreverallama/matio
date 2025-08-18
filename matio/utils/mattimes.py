"""Utility functions for converting MATLAB datetime, duration, and calendarDuration"""

import warnings
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np


def get_tz_offset(tz):
    """Get timezone offset in milliseconds (default UTC)"""
    try:
        tzinfo = ZoneInfo(tz)
        utc_offset = tzinfo.utcoffset(datetime.now())
        if utc_offset is not None:
            offset = int(utc_offset.total_seconds() * 1000)
        else:
            offset = 0
    except ZoneInfoNotFoundError as e:
        warnings.warn(
            f"mat_to_datetime: Could not get timezone offset for {tz}: {e}. Defaulting to UTC."
        )
        offset = 0
    return offset


def mat_to_datetime(props, **_kwargs):
    """Convert MATLAB datetime to Numpy datetime64 array"""

    data = props.get("data", np.array([]))
    if data.size == 0:
        return np.array([], dtype="datetime64[ms]")
    tz = props.get("tz", None)
    if tz is not None and tz.size > 0:
        offset = get_tz_offset(tz.item())
    else:
        offset = 0

    millis = data.real + data.imag * 1e3 + offset

    return millis.astype("datetime64[ms]")


def mat_to_duration(props, **_kwargs):
    """Convert MATLAB duration to Numpy timedelta64 array"""

    millis = props["millis"]
    if millis.size == 0:
        return np.array([], dtype="timedelta64[ms]")

    fmt = props.get("fmt", None)
    if fmt is None:
        return millis.astype("timedelta64[ms]")

    if fmt == "s":
        count = millis / 1000  # Seconds
        dur = count.astype("timedelta64[s]")
    elif fmt == "m":
        count = millis / (1000 * 60)  # Minutes
        dur = count.astype("timedelta64[m]")
    elif fmt == "h":
        count = millis / (1000 * 60 * 60)  # Hours
        dur = count.astype("timedelta64[h]")
    elif fmt == "d":
        count = millis / (1000 * 60 * 60 * 24)  # Days
        dur = count.astype("timedelta64[D]")
    elif fmt == "y":
        count = millis / (1000 * 60 * 60 * 24 * 365)  # Years
        dur = count.astype("timedelta64[Y]")
    else:
        count = millis
        dur = count.astype("timedelta64[ms]")
        # Default case

    return dur


def mat_to_calendarduration(props, **_kwargs):
    """Convert MATLAB calendarDuration to Dict of Python Timedeltas"""

    comps = props.get("components", None)
    if comps is None:
        return props

    comps[0, 0]["months"] = comps[0, 0]["months"].astype("timedelta64[M]")
    comps[0, 0]["days"] = comps[0, 0]["days"].astype("timedelta64[D]")
    comps[0, 0]["millis"] = comps[0, 0]["millis"].astype("timedelta64[ms]")

    return comps


def datetime_to_mat(arr):
    """Convert numpy.datetime64 array to MATLAB datetime format."""
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(arr)}")
    if not np.issubdtype(arr.dtype, np.datetime64):
        raise TypeError(f"Expected numpy.datetime64 array, got {arr.dtype}")

    millis = arr.astype("datetime64[ms]").astype(np.float64)

    if "us" not in str(arr.dtype) and "ns" not in str(arr.dtype):
        data = arr.astype("datetime64[ms]").astype(np.float64)
    else:
        # For sub-ms precision
        millis = arr.astype("datetime64[ms]").astype(np.float64)
        us = arr.astype("datetime64[us]").astype(np.float64)
        frac_ms = (us % 1000) / 1000.0
        data = millis + 1j * frac_ms

    tz = np.empty((0, 0), dtype=np.str_)
    fmt = np.empty((0, 0), dtype=np.str_)
    prop_map = {"data": data, "tz": tz, "fmt": fmt}

    return prop_map


def duration_to_mat(arr):
    """Convert numpy timedelta64 array to MATLAB duration format."""
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(arr)}")
    if not np.issubdtype(arr.dtype, np.timedelta64):
        raise TypeError(f"Expected numpy.timedelta64 array, got {arr.dtype}")

    unit, _ = np.datetime_data(arr.dtype)
    millis = arr.astype("timedelta64[ns]").astype(np.float64) / 1e6
    allowed_units = ("s", "m", "h", "D", "Y")
    if unit not in allowed_units:
        warnings.warn(
            f"duration_to_mat: MATLAB Duration arrays do not support timedelta64[{unit}]. Defaulting to 's'.",
            UserWarning,
        )
        unit = "s"

    unit = unit.lower()
    prop_map = {
        "millis": millis,
        "fmt": unit,
    }

    return prop_map


def calendarduration_to_mat(arr):
    """Convert numpy structured array with fields ['months', 'days', 'millis'] to MATLAB calendarDuration format."""
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(arr)}")
    if arr.dtype.names is None or set(arr.dtype.names) != {"months", "days", "millis"}:
        raise TypeError(
            "Expected structured array with fields ['months', 'days', 'millis']"
        )
    if arr.size > 0 and arr.ndim == 1:
        arr = arr.reshape(1, -1)  # Ensure 2D Shape

    if arr.size > 0:
        arr[0, 0]["months"] = (
            arr[0, 0]["months"].astype("timedelta64[M]").astype("float64")
        )
        arr[0, 0]["days"] = arr[0, 0]["days"].astype("timedelta64[D]").astype("float64")
        arr[0, 0]["millis"] = (
            arr[0, 0]["millis"].astype("timedelta64[ms]").astype("float64")
        )

    fmt = "ymdt"

    prop_map = {
        "components": arr,
        "fmt": fmt,
    }

    return prop_map
