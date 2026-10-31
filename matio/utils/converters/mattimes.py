"""Utility functions for converting MATLAB datetime, duration, and calendarDuration"""

import warnings

import numpy as np

from matio.utils.matclass import MatConvertWarning

caldur_dtype = [
    ("months", "timedelta64[M]"),
    ("days", "timedelta64[D]"),
    ("millis", "timedelta64[ms]"),
]


def mat_to_datetime(props, datetime_unit="ns", **_kwargs):
    """Convert MATLAB datetime to Numpy datetime64 array"""

    data = props.get("data", None)
    if data is None or data.size == 0:
        shape = (0, 0) if data is None else data.shape
        return np.empty(shape, dtype=f"datetime64[{datetime_unit}]")

    tz = props.get("tz", None)
    if tz is not None and tz.size > 0:
        warnings.warn(
            f"MATLAB datetime was saved with timezone '{tz}'."
            "NumPy datetime64 does not support time zones, so the timezone information will be lost.",
            MatConvertWarning,
            stacklevel=2,
        )

    fmt = props.get("fmt", None)
    if fmt is not None and fmt.size > 0:
        warnings.warn(
            "mat_to_datetime: Ignoring 'fmt' property. Numpy datetime64 does not support custom formats.",
            MatConvertWarning,
            stacklevel=2,
        )

    # MATLAB saves NaT as NaN
    nat = np.isnan(data.real)
    data = np.where(nat, 0, data)

    ms_frac, ms_int_float = np.modf(data.real)
    if np.iscomplexobj(data):
        # MATLAB keeps sub-ms precision as a double-double:
        # the imaginary part is the rounding error of the real part
        ms_frac = ms_frac + data.imag
    per_ms = {"ms": 1, "us": 1_000, "ns": 1_000_000}[datetime_unit]
    # datetime64[ns] covers about the years 1678 to 2262, [us] and [ms] much more
    outside = np.abs(ms_int_float) > np.iinfo(np.int64).max // per_ms
    int_part = np.where(outside, 0, ms_int_float).astype(np.int64) * per_ms
    frac_part = np.round(ms_frac * per_ms).astype(np.int64)
    total = int_part + frac_part
    # the sign flips if the sub-ms part pushed the sum past the int64 limit
    outside |= (int_part > 0) & (total < 0) | (int_part < 0) & (total > 0)
    if outside.any():
        warnings.warn(
            f"MATLAB datetime outside the datetime64[{datetime_unit}] range was set to NaT.",
            MatConvertWarning,
            stacklevel=2,
        )
        nat |= outside

    dt = total.astype(f"datetime64[{datetime_unit}]")
    dt[nat] = np.datetime64("NaT", datetime_unit)
    return dt


def count_to_timedelta(count, unit, millis):
    """Cast to the coarsest of unit, ms, us, ns that holds the values exactly"""

    finite = np.isfinite(millis)
    if np.isinf(millis).any():
        warnings.warn(
            "MATLAB duration Inf was set to NaT.",
            MatConvertWarning,
            stacklevel=2,
        )
    # NumPy casts only NaN to NaT, the cast of Inf depends on the platform
    count = np.where(finite, count, np.nan)
    millis = np.where(finite, millis, np.nan)
    candidates = [
        (count, unit),
        (millis, "ms"),
        (millis * 1e3, "us"),
        (millis * 1e6, "ns"),
    ]
    result = None
    for values, values_unit in candidates:
        if np.any(np.abs(values[finite]) >= 2**63):
            break  # this and finer units overflow int64
        result = np.round(values).astype(f"timedelta64[{values_unit}]")
        if np.all(values[finite] == np.round(values[finite])):
            break
    if result is None:
        result = count.astype(f"timedelta64[{unit}]")
    return result


def mat_to_duration(props, **_kwargs):
    """Convert MATLAB duration to Numpy timedelta64 array"""

    millis = props["millis"]
    # if millis.size == 0:
    # return np.empty((0, 0), dtype="timedelta64[ms]")

    fmt = props.get("fmt", None)
    if fmt is None:
        return count_to_timedelta(millis, "ms", millis)

    if fmt == "s":
        count = millis / 1000  # Seconds
        dur = count_to_timedelta(count, "s", millis)
    elif fmt == "m":
        count = millis / (1000 * 60)  # Minutes
        dur = count_to_timedelta(count, "m", millis)
    elif fmt == "h":
        count = millis / (1000 * 60 * 60)  # Hours
        dur = count_to_timedelta(count, "h", millis)
    elif fmt == "d":
        count = millis / (1000 * 60 * 60 * 24)  # Days
        dur = count_to_timedelta(count, "D", millis)
    elif fmt == "y":
        count = millis / (1000 * 60 * 60 * 24 * 365.2425)  # Years like MATLAB
        dur = count_to_timedelta(count, "Y", millis)
    else:
        count = millis
        dur = count_to_timedelta(count, "ms", millis)
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
        x = np.asarray(x).astype(int)
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

    unit = np.datetime_data(arr.dtype)[0]

    # For sub-millisecond precision, write a double-double like MATLAB:
    # the imaginary part is the rounding error of the real part
    if unit in ["us", "ns", "ps", "fs", "as"]:
        per_ms = {"us": 10**3, "ns": 10**6, "ps": 10**9, "fs": 10**12, "as": 10**15}
        ms_int, rest = np.divmod(arr.astype(np.int64), per_ms[unit])
        ms_int = ms_int.astype(np.float64)
        ms_frac = rest / per_ms[unit]
        millis = ms_int + ms_frac
        error = (ms_int - millis) + ms_frac
        if np.any(error[~np.isnat(arr)] != 0):
            millis = millis + 1j * error
    else:
        millis = arr.astype("datetime64[ms]").astype(np.int64).astype(np.float64)
    millis = np.where(np.isnat(arr), np.nan, millis)  # MATLAB saves NaT as NaN

    tz = np.empty((0, 0), dtype=np.str_)
    fmt = np.empty((0, 0), dtype=np.str_)
    prop_map = {"data": millis, "tz": tz, "fmt": fmt}

    return prop_map


def duration_to_mat(arr):
    """Convert numpy timedelta64 array to MATLAB duration format."""

    unit, _ = np.datetime_data(arr.dtype)
    millis = arr.astype("timedelta64[ns]").astype(np.float64) / 1e6
    # MATLAB saves missing durations as NaN
    millis = np.where(np.isnat(arr), np.nan, millis)
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
            # MATLAB saves missing components as NaN
            comps[0, 0][f] = np.where(np.isnat(arr[f]), np.nan, arr[f].astype(float))

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
