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
            f"Could not get timezone offset for {tz}: {e}. Defaulting to UTC."
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

    months = comps[0, 0]["months"].astype("timedelta64[M]")
    days = comps[0, 0]["days"].astype("timedelta64[D]")
    millis = comps[0, 0]["millis"].astype("timedelta64[ms]")

    # Broadcast all components to the same shape
    # MATLAB optimizes by only broadcasting particular components
    months_bc, days_bc, millis_bc = np.broadcast_arrays(months, days, millis)

    dtype = [
        ("months", "timedelta64[M]"),
        ("days", "timedelta64[D]"),
        ("millis", "timedelta64[ms]"),
    ]
    result = np.empty(months_bc.shape, dtype=dtype)
    result["months"] = months_bc.astype("timedelta64[M]")
    result["days"] = days_bc.astype("timedelta64[D]")
    result["millis"] = millis_bc.astype("timedelta64[ms]")

    return result
