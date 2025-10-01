"""Utility functions for converting MATLAB datetime, duration, and calendarDuration"""

import warnings
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np

from matio.utils.matclass import MatConvertWarning

caldur_dtype = [
    ("months", "timedelta64[M]"),
    ("days", "timedelta64[D]"),
    ("millis", "timedelta64[ms]"),
]


def get_tz_offset(tz):
    """Get timezone offset in milliseconds (default UTC)"""
    try:
        tzinfo = ZoneInfo(tz)
        utc_offset = tzinfo.utcoffset(datetime.now())
        if utc_offset is not None:
            offset = int(utc_offset.total_seconds() * 1000)
            warnings.warn(
                f"MATLAB datetime with timezone '{tz}' converted to UTC by applying an offset of {offset / 1000 / 60 / 60:.2f} hours. "
                "NumPy datetime64 does not support time zones, so the original timezone information is lost.",
                MatConvertWarning,
                stacklevel=2,
            )
        else:
            offset = 0
    except ZoneInfoNotFoundError as e:
        warnings.warn(
            f"mat_to_datetime: Could not get timezone offset for {tz}: {e}. Defaulting to UTC.",
            MatConvertWarning,
            stacklevel=2,
        )
        offset = 0
    return offset


def mat_to_datetime(props, **_kwargs):
    """Convert MATLAB datetime to Numpy datetime64 array"""

    data = props.get("data", None)
    if data is None or data.size == 0:
        return np.empty((0, 0), dtype="datetime64[ms]")

    tz = props.get("tz", None)
    if tz is not None and tz.size > 0:
        offset = get_tz_offset(tz.item())
    else:
        offset = 0

    millis = data.real + data.imag * 1e3 + offset

    fmt = props.get("fmt", None)
    if fmt is not None and fmt.size > 0:
        warnings.warn(
            "mat_to_datetime: Ignoring 'fmt' property. Numpy datetime64 does not support custom formats.",
            MatConvertWarning,
            stacklevel=2,
        )

    return millis.astype("datetime64[ms]")


def mat_to_duration(props, **_kwargs):
    """Convert MATLAB duration to Numpy timedelta64 array"""

    millis = props["millis"]
    # if millis.size == 0:
    # return np.empty((0, 0), dtype="timedelta64[ms]")

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
        warnings.warn(
            f"mat_to_duration: Unknown format '{fmt}'. Defaulting to 'ms'.",
            MatConvertWarning,
            stacklevel=2,
        )
        # Default case

    return dur


def mat_caldur_to_numpy(arr):
    """Converts MATLAB caledarDuration struct to numpy timedelta record"""

    def _broadcast_to_ref(x, dtype):
        x = np.asarray(x)
        if x.size == 1:
            return np.full(ref_shape, x.item(), dtype=dtype)
        return x.astype(dtype)

    months = arr[0, 0]["months"]
    days = arr[0, 0]["days"]
    millis = arr[0, 0]["millis"]

    ref_shape = max([x.shape for x in (months, days, millis)], key=lambda s: np.prod(s))
    res = np.empty(ref_shape, dtype=caldur_dtype)
    res["months"] = _broadcast_to_ref(months, "timedelta64[M]")
    res["days"] = _broadcast_to_ref(days, "timedelta64[D]")
    res["millis"] = _broadcast_to_ref(millis, "timedelta64[ms]")

    return res


def mat_to_calendarduration(props, **_kwargs):
    """Convert MATLAB calendarDuration to Dict of Python Timedeltas"""

    comps = props.get("components", None)
    if comps is None:
        return props

    arr = mat_caldur_to_numpy(comps)
    return arr


def datetime_to_mat(arr):
    """Convert numpy.datetime64 array to MATLAB datetime format."""

    # TODO: Allow python datetime input

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

    unit, _ = np.datetime_data(arr.dtype)
    millis = arr.astype("timedelta64[ns]").astype(np.float64) / 1e6
    allowed_units = ("s", "m", "h", "D", "Y")
    if unit not in allowed_units:
        warnings.warn(
            f"duration_to_mat: MATLAB Duration arrays do not support timedelta64[{unit}]. Defaulting to 's'.",
            MatConvertWarning,
        )
        unit = "s"

    unit = unit.lower()
    prop_map = {
        "millis": millis,
        "fmt": unit,
    }

    return prop_map


def numpy_to_matcaldur(arr):
    """Convert numpy structured array with fields ['months', 'days', 'millis'] to MATLAB calendarDuration format."""

    fields = ["months", "days", "millis"]
    comp_dtype = [(f, object) for f in fields]
    comps = np.empty((1, 1), dtype=comp_dtype)

    if arr.size == 0:
        for f in fields:
            comps[0, 0][f] = np.empty((0, 0), dtype=np.float64)
        return comps

    for f in fields:
        if np.all(arr[f] == arr[f].flat[0]):
            # MATLAB reduces to scalar if all values are the same
            reduced = np.array([[arr[f].flat[0]]], dtype=arr[f].dtype)
            comps[0, 0][f] = reduced.astype(np.float64)
        else:
            comps[0, 0][f] = arr[f].astype(np.float64)

    return comps


def calendarduration_to_mat(arr):
    """Convert numpy structured array with fields ['months', 'days', 'millis'] to MATLAB calendarDuration format."""

    comps = numpy_to_matcaldur(arr)
    fmt = "ymwdt"

    prop_map = {
        "components": comps,
        "fmt": fmt,
    }

    return prop_map
