"""Reads MAT-files v7 to v7.2 (MAT-file 5) and extracts variables including MATLAB objects"""

from io import BytesIO

import numpy as np
from scipy.io import loadmat
from scipy.io.matlab._mio5 import MatFile5Reader
from scipy.io.matlab._mio5_params import MatlabOpaque

from matio.subsystem import get_matio_context, load_opaque_object, set_file_wrapper


def new_opaque_object(arr):
    """Creates a new MatOpaque object in place of a MatlabOpaque array"""

    metadata = arr["_ObjectMetadata"].item()
    classname = arr["_Class"]
    type_system = arr["_TypeSystem"]

    obj = load_opaque_object(metadata, classname, type_system)
    return obj


def find_matlab_opaque(arr):
    """Iterate through scipy.loadmat return value to find and replace MatlabOpaque objects"""

    if not isinstance(arr, np.ndarray):
        return arr

    if isinstance(arr, MatlabOpaque):
        arr = new_opaque_object(arr)

    elif arr.dtype == object:
        # Iterate through cell arrays
        for idx in np.ndindex(arr.shape):
            cell_item = arr[idx]
            if isinstance(cell_item, MatlabOpaque):
                arr[idx] = new_opaque_object(cell_item)
            else:
                find_matlab_opaque(cell_item)

    elif arr.dtype.names:
        # Iterate though struct array
        for idx in np.ndindex(arr.shape):
            for name in arr.dtype.names:
                field_val = arr[idx][name]
                if isinstance(field_val, MatlabOpaque):
                    arr[idx][name] = new_opaque_object(field_val)
                else:
                    find_matlab_opaque(field_val)

    return arr


def read_subsystem(
    ssdata,
    byte_order,
    mat_dtype,
    verify_compressed_data_integrity,
):
    """Reads subsystem data as a MAT-file stream"""
    ss_stream = BytesIO(ssdata)

    ss_stream.seek(8)  # Skip subsystem header
    subsystem_reader = MatFile5Reader(
        ss_stream,
        byte_order=byte_order,
        mat_dtype=mat_dtype,
        verify_compressed_data_integrity=verify_compressed_data_integrity,
    )
    subsystem_reader.initialize_read()
    try:
        hdr, _ = subsystem_reader.read_var_header()
        res = subsystem_reader.read_var_array(hdr, process=False)
    except Exception as err:
        raise ValueError(f"Error reading subsystem data: {err}") from err

    return res


def read_matfile5(
    file_path,
    raw_data=False,
    add_table_attrs=False,
    spmatrix=True,
    byte_order=None,
    mat_dtype=False,
    chars_as_strings=True,
    verify_compressed_data_integrity=True,
    variable_names=None,
):
    """Loads variables from MAT-file < v7.3
    Inputs
        1. raw_data (bool): Whether to return raw data for objects
        2. add_table_attrs (bool): Add attributes to pandas DataFrame
        3. spmatrix (bool): Additional arguments for scipy.io.loadmat
        4. byte_order (str): Endianness
        5. mat_dtype (bool): Whether to load MATLAB data types
        6. chars_as_strings (bool): Whether to load character arrays as strings
        8. verify_compressed_data_integrity (bool): Whether to verify compressed data integrity
        9. variable_names (list): List of variable names to load
    Returns:
        1. matfile_dict (dict): Dictionary of loaded variables
    """
    if variable_names is not None:
        if isinstance(variable_names, str):
            variable_names = [variable_names, "__function_workspace__"]
        elif not isinstance(variable_names, list):
            raise TypeError("variable_names must be a string or a list of strings")
        else:
            variable_names.append("__function_workspace__")

    matfile_dict = loadmat(
        file_path,
        spmatrix=spmatrix,
        byte_order=byte_order,
        mat_dtype=mat_dtype,
        chars_as_strings=chars_as_strings,
        verify_compressed_data_integrity=verify_compressed_data_integrity,
        variable_names=variable_names,
    )
    ssdata = matfile_dict.pop("__function_workspace__", None)
    if ssdata is None:
        # No subsystem data in file
        return matfile_dict

    byte_order = "<" if ssdata[0, 2] == b"I"[0] else ">"

    ss_array = read_subsystem(
        ssdata,
        byte_order,
        mat_dtype,
        verify_compressed_data_integrity,
    )

    if "MCOS" in ss_array.dtype.names:
        if ss_array[0, 0]["MCOS"]["_Class"].item() != "FileWrapper__":
            raise ValueError("Missing-FileWrapper__: Cannot load MATLAB MCOS object")

    fwrap_data = ss_array[0, 0]["MCOS"][0]["_ObjectMetadata"]

    with get_matio_context():

        set_file_wrapper(fwrap_data, byte_order, raw_data, add_table_attrs)

        for var, data in matfile_dict.items():
            matfile_dict[var] = find_matlab_opaque(data)

    return matfile_dict
