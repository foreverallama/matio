"""Base class for MAT-file reading and writing"""

import struct
import sys
import time
from io import BytesIO

from matio.matio5 import read_matfile5, save_matfile5
from matio.matio7 import read_matfile7


def get_matfile_version(byte_data):
    """Reads subsystem MAT-file version and endianness from subsys byte stream"""

    maj_ind = int(byte_data[2] == b"I"[0])
    v_major = int(byte_data[maj_ind])
    v_minor = int(byte_data[1 - maj_ind])
    if v_major in (1, 2):
        return v_major, v_minor

    raise NotImplementedError(f"Unknown MAT-file version {v_major}.{v_minor}")


def load_from_mat(
    file_path,
    mdict=None,
    raw_data=False,
    add_table_attrs=False,
    *,
    spmatrix=True,
    **kwargs,
):
    """Loads variables from MAT-file
    Inputs
        1. file_path (str): Path to MAT-file
        2. mdict (dict): Dictionary to store loaded variables
        3. raw_data (bool): Whether to return raw data for objects
        4. add_table_attrs (bool): Add attributes to pandas DataFrame for MATLAB tables/timetables
        5. spmatrix (bool): Additional arguments for scipy.io.loadmat
        6. kwargs: Additional arguments for scipy.io.loadmat
    Returns:
        1. mdict (dict): Dictionary of loaded variables
    """

    with open(file_path, "rb") as f:
        f.seek(124)
        version_bytes = f.read(4)
        v_major, v_minor = get_matfile_version(version_bytes)

    if v_major == 1:
        matfile_dict = read_matfile5(
            file_path, raw_data, add_table_attrs, spmatrix, **kwargs
        )
    elif v_major == 2:
        matfile_dict = read_matfile7(
            file_path, raw_data, add_table_attrs, spmatrix, **kwargs
        )
    else:
        raise NotImplementedError(f"Unknown MAT-file version {v_major}.{v_minor}")

    # Update mdict if present
    if mdict is not None:
        mdict.update(matfile_dict)
    else:
        mdict = matfile_dict

    return mdict


def save_to_mat(
    file_path,
    mdict,
    version="v7",
    do_compression=True,
    global_vars=None,
    oned_as="row",
    use_strings=True,
):
    """Saves variables to MAT-file
    Inputs
        1. file_path (str): Path to MAT-file
        2. mdict (dict): Dictionary of variables to save
    Returns:
        None
    """

    mdict_copy = mdict.copy()
    # Copy as we are replacing objects in place with metadata arrays
    # This is because I'm using scipy.savemat to write data
    # Can be avoided with my own save implementation

    file_stream = BytesIO()
    write_file_header(file_stream, version)

    # FIXME: Should I include an argument for a saveobj return type?
    # Maybe as list of class names?

    if version == "v7":
        subsys_offset = save_matfile5(
            file_stream, mdict_copy, do_compression, global_vars, oned_as, use_strings
        )
    elif version == "v7.3":
        raise NotImplementedError("v7.3 MAT-file saving is not implemented")
    else:
        raise ValueError(f"Unknown MAT-file version '{version}' specified")

    if subsys_offset > 0:
        write_subsystem_offset(file_stream, subsys_offset)

    # FIXME: This might change when v7.3 gets implemented
    with open(file_path, "wb") as f:
        f.write(file_stream.getvalue())


def write_subsystem_offset(file_stream, offset=0):
    """Write 8 bytes of subsystem offset at byte 116"""

    file_stream.seek(116)
    file_stream.write(struct.pack("<Q", offset))
    file_stream.seek(0, 2)  # Seek to end


def write_version(file_stream, version):
    """Write version information"""

    if version == "v7":
        v_major, v_minor = 1, 0
    elif version == "v7.3":
        v_major, v_minor = 2, 0
    else:
        raise ValueError(f"Unknown version: {version}")

    is_little_endian = sys.byteorder == "little"

    if is_little_endian:
        file_stream.write(struct.pack("<BB", v_minor, v_major))
        file_stream.write(b"IM")
    else:
        file_stream.write(struct.pack(">BB", v_major, v_minor))
        file_stream.write(b"MI")


def write_file_header(file_stream, version):
    """Write MAT-file header"""

    current_time = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    description = (
        f"MATLAB 5.0 MAT-file Platform. "
        f"Created on: {current_time} by matio with scipy"
    )
    description_bytes = description.encode("ascii")[:116]  # Truncate if too long
    description_padded = description_bytes.ljust(116, b"\x20")

    file_stream.write(description_padded)
    write_subsystem_offset(file_stream)
    write_version(file_stream, version)
