"""Utility functions for converting MATLAB tables and timetables to pandas DataFrames"""

import warnings

import numpy as np
import pandas as pd

from matio.utils.converters.mattimes import caldur_dtype
from matio.utils.matclass import EmptyMatStruct, MatConvertError, MatConvertWarning

TABLE_VERSION = 4
MIN_TABLE_VERSION = 1

TIMETABLE_VERSION = 6
MIN_TIMETABLE_VERSION = 2

# Pandas marks this as experimental
# Using it here as its the closest match to MATLAB tables
# FIXME: Update with pandas 3.0 release
# https://pandas.pydata.org/docs/user_guide/migration-3-strings.html
pd.options.future.infer_string = True


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

    def make_series(col, dtype_kind):
        if dtype_kind == "T":
            return pd.Series(col, dtype="str")
        else:
            return pd.Series(col)

    rows = {}
    for i in range(nvars):
        vname = varnames[0, i].item()
        coldata = data[0, i]
        if isinstance(coldata, np.ndarray):
            if coldata.shape[1] == 1:
                rows[vname] = make_series(coldata[:, 0], coldata.dtype.kind)
            else:
                for j in range(coldata.shape[1]):
                    colname = f"{vname}_{j + 1}"
                    rows[colname] = make_series(coldata[:, j], coldata.dtype.kind)
        else:
            rows[vname] = pd.Series(coldata)

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
        warnings.warn(
            "get_row_times: MATLAB sampleRate is converted to TimeStep in nanoseconds for NumPy compatibility.",
            MatConvertWarning,
        )
    else:
        step = None
        comps = row_times[0, 0]["stepSize"]

        if comps.dtype == caldur_dtype:
            # calendarDuration
            # Only one of months, days, or millis is non-zero array
            for field in comps.dtype.names:
                if np.any(comps[field] != 0):
                    step = comps[field]
                    break
            if step is None:
                step = comps[0, 0]["millis"]  # fallback if all are zero

            step_unit = np.datetime_data(step.dtype)[0]
            start = start.astype(f"datetime64[{step_unit}]")
        else:
            step = comps
            step_dtype_unit = np.datetime_data(step.dtype)[0]

            if start.dtype.kind == "m":
                start_dtype_new = f"timedelta64[{step_dtype_unit}]"
            else:
                start_dtype_new = f"datetime64[{step_dtype_unit}]"
            start = start.astype(start_dtype_new)

    times = (start + step * np.arange(num_rows)).ravel()
    if times.dtype.kind == "M":
        # MATLAB saves datetime as [ms] data
        times = times.astype("datetime64[ns]")
    return times


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
    props["CustomProps"][0, 0] = EmptyMatStruct(np.empty((1, 1), dtype=object))
    props["VariableCustomProps"][0, 0] = EmptyMatStruct(np.empty((1, 1), dtype=object))
    props["versionSavedFrom"][0, 0] = np.float64(TABLE_VERSION)
    props["minCompatibleVersion"][0, 0] = np.float64(MIN_TABLE_VERSION)
    props["incompatibilityMsg"][0, 0] = np.empty((0, 0), dtype=np.str_)
    props["VersionSavedFrom"][0, 0] = np.float64(TABLE_VERSION)
    props["Description"][0, 0] = np.empty((0, 0), dtype=np.str_)
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

    data = np.empty((1, len(df.columns)), dtype=object)
    for i, col in enumerate(df.columns):
        if df[col].dtype == "str":
            coldata = (
                df[col]
                .to_numpy(dtype=np.dtypes.StringDType(na_object=np.nan))
                .reshape(-1, 1)
            )
        else:
            coldata = df[col].to_numpy().reshape(-1, 1)
        data[0, i] = coldata

    nrows = np.float64(df.shape[0])
    nvars = np.float64(df.shape[1])

    varnames = np.array([str(col) for col in df.columns], dtype=object)

    if df.index.name is not None or not isinstance(df.index, pd.RangeIndex):
        rownames = np.array([str(idx) for idx in df.index], dtype=object)
    else:
        rownames = np.empty((0, 0), dtype=object)

    # All serialized data must be the exact same types (including shape) as expected by MATLAB
    # If not, it might still load correctly, but some class methods may fail
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
    arrayprops["Description"][0, 0] = np.empty((0, 0), dtype=np.str_)
    arrayprops["UserData"][0, 0] = np.empty((0, 0), dtype=np.float64)
    arrayprops["TableCustomProperties"][0, 0] = EmptyMatStruct(
        np.empty((1, 1), dtype=object)
    )

    return {
        "CustomProps": EmptyMatStruct(np.empty((1, 1), dtype=object)),
        "VariableCustomProps": EmptyMatStruct(np.empty((1, 1), dtype=object)),
        "versionSavedFrom": np.float64(TIMETABLE_VERSION),
        "minCompatibleVersion": np.float64(MIN_TIMETABLE_VERSION),
        "incompatibilityMsg": np.empty((0, 0), dtype=np.str_),
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

    data = np.empty((1, len(df.columns)), dtype=object)
    for i, col in enumerate(df.columns):
        if df[col].dtype == "str":
            coldata = (
                df[col]
                .to_numpy(dtype=np.dtypes.StringDType(na_object=np.nan))
                .reshape(-1, 1)
            )
        else:
            coldata = df[col].to_numpy().reshape(-1, 1)
        data[0, i] = coldata

    nrows = np.float64(df.shape[0])
    nvars = np.float64(df.shape[1])

    varnames = np.array([str(col) for col in df.columns], dtype=object)

    dim1 = df.index.name if df.index.name is not None else "Time"
    dimnames = np.array(
        [np.array([dim1]), np.array(["Variables"])], dtype=object
    ).reshape((1, 2))

    if isinstance(df.index, (pd.DatetimeIndex, pd.TimedeltaIndex)):
        rowtimes = df.index.to_numpy()
    else:
        raise MatConvertError(
            "timetable_to_mat: DataFrame index must be DatetimeIndex or TimedeltaIndex."
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
