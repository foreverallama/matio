"""Reads MAT-files v7 to v7.2 (MAT-file 5) and extracts variables including MATLAB objects"""

import zlib
from io import BytesIO

import numpy as np
from scipy.io import loadmat
from scipy.io.matlab._mio5 import MatFile5Reader, MatFile5Writer
from scipy.io.matlab._mio5_params import MatlabOpaque

from matio.subsystem import (
    get_matio_context,
    load_opaque_object,
    parse_input_dict,
    set_file_wrapper,
)


def new_opaque_object(arr):
    """Creates a new MatioOpaque object in place of a MatlabOpaque array"""

    metadata = arr["_ObjectMetadata"].item()
    classname = arr["_Class"].item()
    type_system = arr["_TypeSystem"].item()

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
            arr[idx] = find_matlab_opaque(cell_item)

    elif arr.dtype.names:
        # Iterate though struct array
        for idx in np.ndindex(arr.shape):
            for name in arr.dtype.names:
                field_val = arr[idx][name]
                arr[idx][name] = find_matlab_opaque(field_val)

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

        file_wrapper = set_file_wrapper()
        file_wrapper.init_load(fwrap_data, byte_order, raw_data, add_table_attrs)

        for var, data in matfile_dict.items():
            matfile_dict[var] = find_matlab_opaque(data)

    return matfile_dict


def save_matfile5(
    file_stream,
    mdict,
    do_compression=True,
    global_vars=None,
    oned_as="row",
    use_strings=True,
):
    """Saves variables to MAT-file 5 format (< 7.3)"""

    with get_matio_context():
        file_wrapper = set_file_wrapper()
        file_wrapper.init_save(use_strings)

        mdict = parse_input_dict(mdict)
    subsys_data = mdict.pop("__subsystem__", None)

    scipy_writer = MatFile5Writer(
        file_stream,
        do_compression=do_compression,
        global_vars=global_vars,
        oned_as=oned_as,
    )
    scipy_writer.put_variables(mdict)

    if subsys_data is None:
        return 0

    subsys_data = {"__subsystem__": subsys_data}
    subsys_offset = file_stream.tell()

    subsys_stream = BytesIO()
    pos_mimatrix, pos_ndims, pos_nbytes = write_subsystem_headers(subsys_stream)
    pos_subsys_start = subsys_stream.tell()

    if np.little_endian:
        subsys_stream.write(
            np.array([0, 1, ord("I"), ord("M"), 0, 0, 0, 0], dtype=np.uint8).tobytes()
        )
    else:
        subsys_stream.write(
            np.array([1, 0, ord("M"), ord("I"), 0, 0, 0, 0], dtype=np.uint8).tobytes()
        )

    scipy_writer.do_compression = False
    scipy_writer.file_stream = subsys_stream
    scipy_writer.put_variables(subsys_data)

    len_subsys_data = subsys_stream.tell() - pos_subsys_start
    update_subsystem_headers(
        subsys_stream, pos_mimatrix, pos_ndims, pos_nbytes, len_subsys_data
    )

    if do_compression:
        out_str = zlib.compress(subsys_stream.getvalue())
        tag = np.array([15, len(out_str)], dtype=np.uint32).view(np.uint8)
        file_stream.write(tag.tobytes())
        file_stream.write(out_str)
    else:
        file_stream.write(subsys_stream.getvalue())

    return subsys_offset


def write_subsystem_headers(file_stream):
    """Write subsystem headers to the file stream"""

    file_stream.write(np.array([14, 0], dtype=np.uint32).tobytes())  # miMATRIX Header
    pos_mimatrix = file_stream.tell() - 4

    file_stream.write(np.array([6, 8, 9, 0], dtype=np.uint32).tobytes())  # Array Flag
    file_stream.write(
        np.array([5, 8, 1, 0], dtype=np.uint32).tobytes()
    )  # Dimensions Flag
    pos_ndims = file_stream.tell() - 4

    file_stream.write(np.array([1, 0], dtype=np.uint32).tobytes())  # Array Name
    file_stream.write(np.array([2, 0], dtype=np.uint32).tobytes())  # Array Value Header
    pos_nbytes = file_stream.tell() - 4

    return pos_mimatrix, pos_ndims, pos_nbytes


def update_subsystem_headers(
    file_stream, pos_mimatrix, pos_ndims, pos_nbytes, len_subsys_data
):
    """Update subsystem headers in the file stream"""

    mi_matrix_nbytes = len_subsys_data + 48  # Include Data element headers

    file_stream.seek(pos_mimatrix)
    file_stream.write(np.array([mi_matrix_nbytes], dtype=np.uint32).tobytes())

    file_stream.seek(pos_ndims)
    file_stream.write(np.array([len_subsys_data], dtype=np.uint32).tobytes())

    file_stream.seek(pos_nbytes)
    file_stream.write(np.array([len_subsys_data], dtype=np.uint32).tobytes())

    file_stream.seek(0, 2)  # Seek to end
