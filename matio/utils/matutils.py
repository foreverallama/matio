"""Utility functions for matio"""

import warnings
from collections.abc import Mapping

import numpy as np
from scipy.sparse import issparse

from matio.utils.matclass import (
    EmptyMatStruct,
    IntegerDecodingHint,
    MatlabCanonicalEmpty,
    MatlabClasses,
    MatlabOpaque,
    MatWriteWarning,
)
from matio.utils.matconvert import convert_py_to_mat, guess_class_name
from matio.utils.matheaders import MAT_5_VERSION, MAT_HDF_VERSION


def chars_to_strings(in_arr):
    """Convert a numpy array of characters to an array of strings.
    Notes:
    Scipy basically just collapses the last axis into the string length.
    """
    arr = in_arr
    ndim = arr.ndim
    last_dim = arr.shape[-1]

    # Handle empty last axis
    if last_dim == 0:
        new_dt_str = arr.dtype.str
        if ndim == 2:
            out_shape = (0,)
        else:
            out_shape = arr.shape[:-2] + (0,)
    else:
        # Construct new dtype with last_dim as string length
        new_dt_str = arr.dtype.str[:-1] + str(last_dim)
        out_shape = arr.shape[:-1]

    arr = np.ascontiguousarray(arr)
    arr = arr.view(new_dt_str)
    return arr.reshape(out_shape)


def _get_string_arr_dtype(arr, num=1):
    """Return dtype for given number of items per element"""
    if arr.dtype.kind == "U":
        return np.dtype(f"U{num}")
    elif arr.dtype.kind == "S":
        return np.dtype(f"S{num}")


def strings_to_chars(arr):
    """Convert numpy string array to char array.
    Notes:
    Expands 1D numpy string arrays to char arrays.
    Basically inverse of chars_to_strings in scipy.

    2D numpy string arrays will be treated as MATLAB strings instead
    """
    dims = list(arr.shape)
    if not dims:
        dims = [1]
    dims.append(int(arr.dtype.str[2:]))
    arr = np.ndarray(shape=dims, dtype=_get_string_arr_dtype(arr), buffer=arr)
    empties = [arr == np.array("", dtype=arr.dtype)]
    if not np.any(empties):
        return arr
    arr = arr.copy()
    arr[tuple(empties)] = " "
    return arr


def matlab_class_to_dtype(matlab_class, current_dtype):
    """Map MATLAB class names to numpy dtypes.
    Used for v7.3 format
    """
    matlab_class = matlab_class.decode("ascii")
    try:
        dtype = np.dtype(matlab_class)
    except TypeError:
        if matlab_class == MatlabClasses.LOGICAL:
            dtype = np.dtype(np.bool_)
        else:
            # Fallback
            dtype = current_dtype
    return dtype


def matdims(arr, oned_as="col"):
    """Determine equivalent MATLAB dimensions for given array"""
    shape = arr.shape
    if shape == ():  # scalar
        return (1, 1)
    if len(shape) == 1:  # 1D
        if shape[0] == 0:
            return (0, 0)
        elif oned_as == "col":
            return shape + (1,)
        elif oned_as == "row":
            return (1,) + shape
        else:
            raise ValueError(f'Expected oned_as to be "col" or "row", got {oned_as!r}')
    return shape


def mat_numeric(arr, version, classname=None, int_decode=None):
    """Convert numpy numeric type to MATLAB compatible type"""

    dt = arr.dtype
    if dt.kind in ("i", "u"):
        if dt.itemsize <= 8:
            target_dtype = dt
            classname = target_dtype.name
        else:
            target_dtype = np.int64 if dt.kind == "i" else np.uint64
            target_dtype = np.dtype(target_dtype)
            classname = target_dtype.name
            warnings.warn(
                f"Integer type {dt} not supported in MATLAB. Converting to {target_dtype}.",
                MatWriteWarning,
            )
    elif dt.kind == "f":
        if dt.itemsize == 4:
            target_dtype = np.dtype(np.float32)
            classname = MatlabClasses.SINGLE
        elif dt.itemsize == 8:
            target_dtype = np.dtype(np.float64)
            classname = MatlabClasses.DOUBLE
        else:
            target_dtype = np.dtype(np.float64)
            classname = MatlabClasses.DOUBLE
            warnings.warn(
                f"Float type {dt} not supported in MATLAB. Converting to {target_dtype}.",
                MatWriteWarning,
            )
    elif dt.kind == "c":
        if version == MAT_5_VERSION:
            target_dtype = np.dtype(np.complex128)
            if dt.itemsize != 16:
                warnings.warn(
                    f"Complex type {dt} not supported in MATLAB. Converting to {target_dtype}.",
                    MatWriteWarning,
                )
        elif version == MAT_HDF_VERSION:
            target_dtype = np.dtype([("real", np.float64), ("imag", np.float64)])
            classname = "double"
            if dt.itemsize != 16:
                warnings.warn(
                    f"Complex type {dt} not supported in MATLAB. Converting to {target_dtype}.",
                    MatWriteWarning,
                )

            data_new = np.empty(arr.shape, dtype=target_dtype)
            data_new["real"] = arr.real
            data_new["imag"] = arr.imag
            arr = data_new
        else:
            raise ValueError(f"Unknown MAT-file version '{version}' specified")

    elif dt.kind == "b":
        target_dtype = np.dtype(np.uint8)
        int_decode = IntegerDecodingHint.LOGICAL_HINT
        classname = MatlabClasses.LOGICAL

    # arr = arr.astype(target_dtype, copy=True)
    arr = arr.astype(target_dtype)
    return arr, classname, int_decode


def to_writeable(source, oned_as="col"):
    """Convert input object ``source`` to something we can write

    Parameters
    ----------
    source : object

    Returns
    -------
    arr : None or ndarray or EmptyStructMarker
        If `source` cannot be converted to something we can write to a matfile,
        return None.  If `source` is equivalent to an empty dictionary, return
        ``EmptyStructMarker``.  Otherwise return `source` converted to an
        ndarray with contents for writing to matfile.
    """
    if source is None:
        return np.empty((0, 0), dtype=np.float64)

    if isinstance(source, (MatlabOpaque, MatlabCanonicalEmpty)):
        return source

    if issparse(source):
        return source

    classname = guess_class_name(source)
    if classname is not None:
        if isinstance(source, (np.ndarray, np.generic)):
            source = np.asanyarray(source)
            source = np.atleast_2d(source)
            if source.ndim == 1 and oned_as == "col":
                source = source.T
        return convert_py_to_mat(source, classname)

    if isinstance(source, (np.ndarray, np.generic)):
        narr = np.asanyarray(source)

        if narr.dtype.kind in ("U", "S"):
            # Backwards compatibility with scipy "chars_to_strings"
            return narr

        narr = np.atleast_2d(narr)
        if narr.ndim == 1 and oned_as == "col":
            narr = narr.T

        return narr

    if hasattr(source, "__array__"):
        return np.asarray(source)

    if isinstance(source, Mapping):
        is_mapping = True
    elif hasattr(source, "__dict__"):
        source = {
            key: value
            for key, value in source.__dict__.items()
            if not key.startswith("_")
        }
        is_mapping = True
    else:
        is_mapping = False

    if is_mapping:
        dtype = []
        values = []
        for field, value in source.items():
            if isinstance(field, str):
                if field[0] not in "_0123456789":
                    dtype.append((str(field), object))
                    values.append(value)
                else:
                    msg = (
                        f"Starting field name with a underscore "
                        f"or a digit ({field}) is ignored"
                    )
                    warnings.warn(msg, MatWriteWarning, stacklevel=2)
        if dtype:
            return np.array([tuple(values)], dtype).reshape((1, 1))
        else:
            return EmptyMatStruct(np.array([]))

    # Try and convert to numpy array
    try:
        narr = np.asanyarray(source)
    except ValueError:
        narr = np.asanyarray(source, dtype=object)

    if narr.dtype.type in (object, np.object_) and narr.shape == () and narr == source:
        raise TypeError(f"Could not convert {type(source)} to a writeable datatype")

    if narr.dtype.kind in ("U", "S"):
        # Backwards compatibility with scipy "chars_to_strings"
        return np.atleast_1d(narr)

    narr = np.atleast_2d(narr)
    if oned_as == "col":
        narr = narr.T
    return narr


def shape_from_metadata(metadata):
    """Extract shape from MATLAB object metadata"""

    if not isinstance(metadata, np.ndarray):
        return ()

    if metadata.dtype == np.uint32:
        ndims = metadata.flat[1]
        dims = [int(x) for x in metadata.flat[2 : 2 + ndims]]

    elif metadata.dtype.hasobject and metadata.dtype.names:
        # Enumeration Array
        dims = metadata[0, 0]["ValueIndices"].shape

    else:
        dims = metadata.shape

    return tuple(dims)


def sanitize_input_lists(var_list, arg_name):
    """Sanitize input list of variable names"""
    if var_list is None:
        if arg_name == "variable_names":
            return None
        else:
            return []

    if isinstance(var_list, str):
        return [var_list]

    if not isinstance(var_list, (list, tuple)):
        raise ValueError(f"{arg_name} must be a list of strings")

    return list(var_list)
