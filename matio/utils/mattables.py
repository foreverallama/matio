"""Utility functions for converting MATLAB tables and timetables to pandas DataFrames"""

import warnings

import numpy as np
import pandas as pd
from scipy.io.matlab._mio5 import EmptyStructMarker

TABLE_VERSION = 4
MIN_TABLE_VERSION = 1

TIMETABLE_VERSION = 6
MIN_TIMETABLE_VERSION = 2


def add_table_props(df, tab_props):
    """Add MATLAB table properties to pandas DataFrame
    These properties are mostly cell arrays of character vectors
    """

    tab_props = tab_props[0, 0]

    df.attrs["Description"] = (
        tab_props["Description"].item() if tab_props["Description"].size > 0 else ""
    )
    df.attrs["VariableDescriptions"] = [
        s.item() if s.size > 0 else ""
        for s in tab_props["VariableDescriptions"].ravel()
    ]
    df.attrs["VariableUnits"] = [
        s.item() if s.size > 0 else "" for s in tab_props["VariableUnits"].ravel()
    ]
    df.attrs["VariableContinuity"] = [
        s.item() if s.size > 0 else "" for s in tab_props["VariableContinuity"].ravel()
    ]
    df.attrs["DimensionNames"] = [
        s.item() if s.size > 0 else "" for s in tab_props["DimensionNames"].ravel()
    ]
    df.attrs["UserData"] = tab_props["UserData"]

    return df


def add_timetable_props(df, tab_props):
    """Add MATLAB table properties to pandas DataFrame
    These properties are mostly cell arrays of character vectors
    """
    df.attrs["varDescriptions"] = [
        s.item() if s.size > 0 else "" for s in tab_props["varDescriptions"].ravel()
    ]
    df.attrs["varUnits"] = [
        s.item() if s.size > 0 else "" for s in tab_props["varUnits"].ravel()
    ]
    df.attrs["varContinuity"] = [
        s.item() if s.size > 0 else "" for s in tab_props["varContinuity"].ravel()
    ]
    df.attrs["UserData"] = tab_props["arrayProps"]["UserData"][0, 0]
    df.attrs["Description"] = (
        tab_props["arrayProps"]["Description"][0, 0].item()
        if tab_props["arrayProps"]["Description"][0, 0].size > 0
        else ""
    )

    return df


def to_dataframe(data, nvars, varnames):
    """Creates a dataframe from coldata and column names"""
    rows = {}
    for i in range(nvars):
        vname = varnames[0, i].item()
        coldata = data[0, i]

        # If variable is multicolumn data
        if isinstance(coldata, np.ndarray):
            if coldata.shape[1] == 1:
                rows[vname] = coldata[:, 0]
            else:
                for j in range(coldata.shape[1]):
                    colname = f"{vname}_{j + 1}"
                    rows[colname] = coldata[:, j]
        else:
            rows[vname] = coldata

    df = pd.DataFrame(rows)
    return df


def mat_to_table(props, add_table_attrs=False, **_kwargs):
    """Converts MATLAB table to pandas DataFrame"""

    table_attrs = props.get("props")
    ver = int(table_attrs[0, 0]["versionSavedFrom"].item())
    if ver > TABLE_VERSION:
        warnings.warn(
            f"mat_to_table: MATLAB table version {ver} is not supported.",
            UserWarning,
        )
        return props

    data = props.get("data")
    nvars = int(props.get("nvars").item())
    varnames = props.get("varnames")
    df = to_dataframe(data, nvars, varnames)

    # Add df.index
    nrows = int(props.get("nrows").item())
    rownames = props.get("rownames")
    if rownames.size > 0:
        rownames = [s.item() for s in rownames.ravel()]
        if len(rownames) == nrows:
            df.index = rownames

    if add_table_attrs:
        # Since pandas lists this as experimental, flag so we can switch off if it breaks
        df = add_table_props(df, table_attrs)

    return df


def get_row_times(row_times, num_rows):
    """Get row times from MATLAB timetable
    rowTimes is a duration or datetime array if explicitly specified
    If using "SampleRate" or "TimeStep", it is a struct array with the following fields:
    1. origin - the start time as a duration or datetime scalar
    2. specifiedAsRate - boolean indicating which to use - sampleRate or TimeStep
    3. stepSize - the time step as a duration scalar
    4. sampleRate - the sample rate as a float
    """
    if not row_times.dtype.names:
        return row_times.ravel()

    start = row_times[0, 0]["origin"]
    if row_times[0, 0]["specifiedAsRate"]:
        fs = row_times[0, 0]["sampleRate"].item()
        step = np.timedelta64(int(1e9 / fs), "ns")
    else:
        comps = row_times[0, 0]["stepSize"]
        if comps.dtype.names is not None:
            # calendarDuration
            # Only one of months, days, or millis is non-zero array
            for key in ["months", "days", "millis"]:
                arr = comps[0, 0][key]
                if np.any(arr != 0):
                    step = arr
                    break
            else:
                step = comps[0, 0]["millis"]  # fallback if all are zero
            step_unit = np.datetime_data(step.dtype)[0]
            start = start.astype(f"datetime64[{step_unit}]")
        else:
            step = comps.astype("timedelta64[ns]")

    return (start + step * np.arange(num_rows)).ravel()


def mat_to_timetable(props, add_table_attrs=False, **_kwargs):
    """Converts MATLAB timetable to pandas DataFrame"""

    timetable_data = props.get("any", None)
    if timetable_data is None:
        return props

    ver = int(timetable_data[0, 0]["versionSavedFrom"].item())
    if ver > TIMETABLE_VERSION or ver <= MIN_TIMETABLE_VERSION:
        warnings.warn(
            f"mat_to_timetable: MATLAB timetable version {ver} is not supported.",
            UserWarning,
        )
        return props

    num_vars = int(timetable_data[0, 0]["numVars"].item())
    var_names = timetable_data[0, 0]["varNames"]
    data = timetable_data[0, 0]["data"]
    df = to_dataframe(data, num_vars, var_names)

    row_times = timetable_data[0, 0]["rowTimes"]
    num_rows = int(timetable_data[0, 0]["numRows"].item())

    row_times = get_row_times(row_times, num_rows)
    dim_names = timetable_data[0, 0]["dimNames"]
    df.index = pd.Index(row_times, name=dim_names[0, 0].item())

    if add_table_attrs:
        # Since pandas lists this as experimental, flag so we can switch off if it breaks
        df = add_timetable_props(df, timetable_data[0, 0])

    return df


def mat_to_categorical(props, **_kwargs):
    """Converts MATLAB categorical to pandas Categorical
    MATLAB categorical objects are stored with the following properties:
    1. categoryNames - all unique categories
    2. codes
    3. isOrdinal - boolean indicating if the categorical is ordered
    4. isProtected - boolean indicating if the categorical is protected
    """

    raw_names = props.get("categoryNames")
    category_names = [name.item() for name in raw_names.ravel()]

    # MATLAB codes are 1-indexed as uint integers
    codes = props.get("codes").astype(int) - 1
    ordered = bool(props.get("isOrdinal").item())
    return pd.Categorical.from_codes(codes, categories=category_names, ordered=ordered)


def make_table_props():
    """Creates default properties for a MATLAB table"""
    dtype = [
        ("useVariableNamesOriginal", object),
        ("useDimensionNamesOriginal", object),
        ("CustomProps", object),
        ("VariableCustomProps", object),
        ("versionSavedFrom", object),
        ("minCompatibleVersion", object),
        ("incompatibilityMsg", object),
        ("VersionSavedFrom", object),
        ("Description", object),
        ("VariableNamesOriginal", object),
        ("DimensionNames", object),
        ("DimensionNamesOriginal", object),
        ("UserData", object),
        ("VariableDescriptions", object),
        ("VariableUnits", object),
        ("VariableContinuity", object),
    ]

    props = np.empty((1, 1), dtype=dtype)

    props["useVariableNamesOriginal"][0, 0] = np.bool_(False)
    props["useDimensionNamesOriginal"][0, 0] = np.bool_(False)
    props["CustomProps"][0, 0] = EmptyStructMarker()
    props["VariableCustomProps"][0, 0] = EmptyStructMarker()
    props["versionSavedFrom"][0, 0] = np.float64(TABLE_VERSION)
    props["minCompatibleVersion"][0, 0] = np.float64(MIN_TABLE_VERSION)
    props["incompatibilityMsg"][0, 0] = ""
    props["VersionSavedFrom"][0, 0] = np.float64(TABLE_VERSION)
    props["Description"][0, 0] = ""
    props["VariableNamesOriginal"][0, 0] = np.empty((0, 0), dtype=object)
    props["DimensionNames"][0, 0] = np.array(
        [np.array(["Row"]), np.array(["Variables"])], dtype=object
    ).reshape((1, 2))
    props["DimensionNamesOriginal"][0, 0] = np.empty((0, 0), dtype=object)
    props["UserData"][0, 0] = np.empty((0, 0), dtype=np.float64)
    props["VariableDescriptions"][0, 0] = np.empty((0, 0), dtype=object)
    props["VariableUnits"][0, 0] = np.empty((0, 0), dtype=object)
    props["VariableContinuity"][0, 0] = np.empty((0, 0), dtype=object)

    return props


def table_to_mat(df):
    """Converts a pandas DataFrame to a MATLAB table"""

    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")

    data = np.empty((1, len(df.columns)), dtype=object)
    for i, col in enumerate(df.columns):
        if pd.api.types.is_string_dtype(df[col]):
            coldata = df[col].to_numpy(dtype="U")
        else:
            coldata = df[col].to_numpy().reshape(-1, 1)
        data[0, i] = coldata

    nrows = np.float64(df.shape[0])
    nvars = np.float64(df.shape[1])

    varnames = np.array([str(col) for col in df.columns], dtype=object)

    if df.index.name is not None or not isinstance(df.index, pd.RangeIndex):
        rownames = np.array([str(idx) for idx in df.index], dtype=object)
    else:
        rownames = np.array([], dtype=object)

    # FIXME: Add table attributes
    extras = make_table_props()
    prop_map = {
        "data": data,
        "varnames": varnames,
        "nrows": nrows,
        "nvars": nvars,
        "rownames": rownames,
        "ndims": np.float64(2),
        "props": extras,
    }

    return prop_map


def make_timetable_props():
    """Creates default properties for a MATLAB timetable"""

    arrayprops_dtype = [
        ("Description", object),
        ("UserData", object),
        ("TableCustomProperties", object),
    ]
    arrayprops = np.empty((1, 1), dtype=arrayprops_dtype)
    arrayprops["Description"][0, 0] = ""
    arrayprops["UserData"][0, 0] = np.empty((0, 0), dtype=np.float64)
    arrayprops["TableCustomProperties"][0, 0] = EmptyStructMarker()

    return {
        "CustomProps": EmptyStructMarker(),
        "VariableCustomProps": EmptyStructMarker(),
        "versionSavedFrom": np.float64(TIMETABLE_VERSION),
        "minCompatibleVersion": np.float64(MIN_TIMETABLE_VERSION),
        "incompatibilityMsg": "",
        "arrayProps": arrayprops,
        "numDims": np.float64(2),
        "useVarNamesOrig": np.bool_(False),
        "useDimNamesOrig": np.bool_(False),
        "dimNamesOrig": np.empty((0, 0), dtype=object),
        "varNamesOrig": np.empty((0, 0), dtype=object),
        "varDescriptions": np.empty((0, 0), dtype=object),
        "varUnits": np.empty((0, 0), dtype=object),
        "timeEvents": np.empty((0, 0), dtype=np.float64),
        "varContinuity": np.empty((0, 0), dtype=object),
    }


def timetable_to_mat(df):
    """Converts a pandas DataFrame to a MATLAB timetable"""

    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")

    data = np.empty((1, len(df.columns)), dtype=object)
    for i, col in enumerate(df.columns):
        if pd.api.types.is_string_dtype(df[col]):
            coldata = df[col].to_numpy(dtype="U")
        else:
            coldata = df[col].to_numpy().reshape(-1, 1)
        data[0, i] = coldata

    nrows = np.float64(df.shape[0])
    nvars = np.float64(df.shape[1])

    varnames = np.array([str(col) for col in df.columns], dtype=object)
    dimnames = np.array(
        [np.array(["Time"]), np.array(["Variables"])], dtype=object
    ).reshape((1, 2))

    if isinstance(df.index, (pd.DatetimeIndex, pd.TimedeltaIndex)):
        rowtimes = df.index.to_numpy()
        if np.issubdtype(rowtimes.dtype, np.timedelta64):
            unit, _ = np.datetime_data(rowtimes.dtype)
            if unit not in ("s", "m", "h", "D", "Y"):
                warnings.warn(
                    f"timetable_to_mat: MATLAB Duration arrays do not support timedelta64[{unit}]. Defaulting to 'ns'.",
                    UserWarning,
                )
                rowtimes = rowtimes.astype("timedelta64[ns]")
    else:
        raise ValueError(
            "cannot convert DataFrame to MATLAB Timetable: "
            "Requires datetime or timedelta row index"
        )

    # Define timetable struct dtype
    timetable_dtype = [
        ("data", object),
        ("dimNames", object),
        ("varNames", object),
        ("numRows", object),
        ("numVars", object),
        ("rowTimes", object),
    ]

    # FIXME: Add timetable attributes
    extras = make_timetable_props()
    timetable_dtype.extend((key, object) for key in extras)

    # Create 1x1 structured array
    timetable = np.empty((1, 1), dtype=timetable_dtype)
    timetable[0, 0]["data"] = data
    timetable[0, 0]["dimNames"] = dimnames
    timetable[0, 0]["varNames"] = varnames.reshape((1, -1))
    timetable[0, 0]["numRows"] = nrows
    timetable[0, 0]["numVars"] = nvars
    timetable[0, 0]["rowTimes"] = rowtimes.reshape((-1, 1))

    for key, value in extras.items():
        timetable[0, 0][key] = value

    return {"any": timetable}


def categorical_to_mat(cat):
    """Converts a pandas Categorical to a MATLAB categorical"""

    category_names = cat.categories.to_numpy(dtype=object).reshape(-1, 1)
    codes = cat.codes.astype("int8") + 1  # 1-based indexing
    is_ordinal = np.bool_(cat.ordered)
    is_protected = np.bool_(False)  # not supported in pandas

    return {
        "categoryNames": category_names,
        "codes": codes,
        "isOrdinal": is_ordinal,
        "isProtected": is_protected,
    }
