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
from matio.utils.matheaders import MAT_FILE_VERSIONS


def decode_char_arrays(arr, codec="utf-8", mdtype=0, char_axis=1):
    """
    Decode char arrays to numpy unicode strings.
    Notes:
    Char arrays are stored as uint8/uint16 integer arrays.
    Read rows as strings and decode with specified codec.
    Reshape back to original shape with char (row) axis removed.
    """
    char_dims = arr.shape
    out_dims = char_dims[:char_axis] + char_dims[char_axis + 1 :]

    if arr.size == 0:
        return np.empty(out_dims, dtype="<U0")

    char_len = char_dims[char_axis]

    # Move char axis to end and flatten
    arr2 = arr if char_axis == arr.ndim - 1 else np.moveaxis(arr, char_axis, -1)
    flat = arr2.reshape(-1, char_len)

    if mdtype == 4:
        strings = []
        for char_data in flat:
            row = bytearray()
            for msb, lsb in zip(char_data[::2], char_data[1::2]):
                if msb == 0:
                    row.append(lsb)
                else:
                    row.extend((msb, lsb))
            strings.append(row.decode(codec, errors="surrogatepass"))
    else:
        byte_data = flat.tobytes()
        row_bytes = char_len * flat.dtype.itemsize
        strings = [
            byte_data[pos : pos + row_bytes].decode(codec, errors="surrogatepass")
            for pos in range(0, len(byte_data), row_bytes)
        ]

    out = np.array(strings, dtype=np.str_)
    return out.reshape(out_dims)


def encode_char_arrays(arr):
    """Convert numpy string array into matlab integer array"""

    arr = np.atleast_1d(arr)
    flat = arr.ravel(order="F")
    encoded = np.array([s.encode("utf-16-le") for s in flat], dtype="S")
    if encoded.dtype.itemsize == 1:
        # NumPy uses S1 when all encoded strings are empty.
        out = np.empty((len(flat), 0), dtype=np.uint16)
    else:
        out = encoded.view(np.uint16).reshape(len(flat), -1)
    if not out.size or out.max() <= 127:
        out = out.astype(np.uint8)
    reshaped = out.reshape(arr.shape + (out.shape[1],), order="F")
    return reshaped if reshaped.ndim == 2 else np.moveaxis(reshaped, -1, 1)


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
        # MATLAB stores complex single and complex double
        target_dtype = np.dtype(np.complex64 if dt.itemsize == 8 else np.complex128)
        if dt.itemsize not in (8, 16):
            warnings.warn(
                f"Complex type {dt} not supported in MATLAB. Converting to {target_dtype}.",
                MatWriteWarning,
            )
        if version == MAT_FILE_VERSIONS.HDF:
            part = np.float32 if target_dtype == np.complex64 else np.float64
            classname = (
                MatlabClasses.SINGLE if part == np.float32 else MatlabClasses.DOUBLE
            )
            data_new = np.empty(arr.shape, dtype=[("real", part), ("imag", part)])
            data_new["real"] = arr.real
            data_new["imag"] = arr.imag
            arr = data_new
            target_dtype = data_new.dtype
        elif version != MAT_FILE_VERSIONS.V5:
            raise ValueError(f"Unknown MAT-file version '{version}' specified")

    elif dt.kind == "b":
        target_dtype = np.dtype(np.uint8)
        int_decode = IntegerDecodingHint.LOGICAL_HINT
        classname = MatlabClasses.LOGICAL

    arr = arr.astype(target_dtype, copy=False)
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
        name = type(source).__name__
        source = {
            key: value
            for key, value in source.__dict__.items()
            if not key.startswith("_")
        }
        if not source:
            msg = (
                f"{name} object has no public instance attributes "
                f"and is saved as an empty struct"
            )
            warnings.warn(msg, MatWriteWarning, stacklevel=2)
        is_mapping = True
    else:
        is_mapping = False

    if is_mapping:
        dtype = []
        values = []
        for field, value in source.items():
            if isinstance(field, str) and is_valid_struct_fieldname(field):
                dtype.append((field, object))
                values.append(value)
            else:
                msg = f"{field!r} is not a valid MATLAB field name. Skipping it"
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


def is_valid_matlab_varname(name):
    """Checks if a name is a valid MATLAB variable name

    MATLAB names start with a letter and use only ASCII letters, digits and
    underscores. S = load(f) fails for the whole file with any other name.
    """
    return name.isascii() and name.isidentifier() and not name.startswith("_")


def is_valid_struct_fieldname(name):
    """Checks if a name can be written as a MATLAB struct field name

    Unlike variable names, MATLAB loads field names with spaces or other ASCII
    characters. Non-ASCII names fail to encode when writing.
    """
    return name.isascii() and name != "" and name[0] not in "_0123456789"


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
