import struct
import sys
import time
from enum import IntEnum, StrEnum

import h5py
import numpy as np

from matio.utils.matclass import MatReadError


class MAT_FILE_VERSIONS(IntEnum):
    """Supported MAT-file major versions."""

    V4 = 0
    V5 = 1
    HDF = 2


MAT_FILE_VERSIONS_STR = {
    "v7": MAT_FILE_VERSIONS.V5,
    "v7.3": MAT_FILE_VERSIONS.HDF,
    "v4": MAT_FILE_VERSIONS.V4,
}

MAT4_HEADER_SIZE_BYTES = 4
MAT4_HEADER_MOPT_MAX_VAL = 4052

MAT5_HEADER_SIZE_BYTES = 128
MAT5_MAX_ARR_BYTES = 2**32
MAT5_MAX_STRUCT_FIELDNAME_LEN = 64

MAT_HDF_USER_BLOCK_BYTES = 512
MAT_HDF_REFS_GROUP = "#refs#"
MAT_HDF_SUBSYS_GROUP = "#subsystem#"
MAT_HDF_COMPRESSION = "gzip"
MAT_HDF_COMPRESSION_OPTS = 3

MCOS_MAGIC_NUMBER = 0xDD000000


class MAT_HDF_ATTRS(StrEnum):
    """Enumeration for standard HDF5 attributes in MAT v7.3 files"""

    CLASS = "MATLAB_class"
    OBJECT_DECODE = "MATLAB_object_decode"
    INT_DECODE = "MATLAB_int_decode"
    EMPTY = "MATLAB_empty"
    GLOBAL = "MATLAB_global"
    SPARSE = "MATLAB_sparse"
    FIELDS = "MATLAB_fields"


def check_mat_v4_version(data):
    """Check if data corresponds to a valid MAT-file v4 header"""
    # v4 files typically have a zero in first 4 bytes
    arr = np.frombuffer(data, dtype=np.uint8)
    return bool(np.any(arr == 0))


def check_mat_version(data):
    """Reads MAT-file version from header data"""

    if data[2:] == b"IM":
        byte_order = "<"
    elif data[2:] == b"MI":
        byte_order = ">"
    else:
        raise MatReadError("Invalid endian indicator in MAT-file header")

    v_major, v_minor = int(data[1]), int(data[0])
    if byte_order != "<":
        v_major, v_minor = v_minor, v_major

    if v_major not in (MAT_FILE_VERSIONS.V5, MAT_FILE_VERSIONS.HDF):
        raise MatReadError(f"Unknown MAT-file version {v_major}.{v_minor}")

    return byte_order, v_major


def read_mat_header(file_path):
    """Reads MAT-file header and returns version information"""

    with open(file_path, "rb") as f:
        # v4 files may not have size MAT5_HEADER_SIZE_BYTES, so we check for v4 first
        data = f.read(MAT4_HEADER_SIZE_BYTES)
        if check_mat_v4_version(data):
            data_le = np.frombuffer(data[:4], dtype="<i4")[0]
            data_be = np.frombuffer(data[:4], dtype=">i4")[0]
            if 0 <= data_le <= MAT4_HEADER_MOPT_MAX_VAL:
                byte_order = "<"
            elif 0 <= data_be <= MAT4_HEADER_MOPT_MAX_VAL:
                byte_order = ">"
            else:
                raise MatReadError(
                    "Could not determine byte order for MAT-file v4 header"
                )

            return 0, MAT_FILE_VERSIONS.V4, byte_order

        f.seek(0)  # Reset position
        data = f.read(MAT5_HEADER_SIZE_BYTES)
        byte_order, v_major = check_mat_version(data[124:])

        subsystem_offset = np.frombuffer(data[116:124], dtype=byte_order + "u8")[0]
        if subsystem_offset == 0x2020202020202020:
            subsystem_offset = 0  # All spaces in BE Files

        return subsystem_offset, v_major, byte_order


def write_subsystem_offset(file_stream, offset=0):
    """Write 8 bytes of subsystem offset at byte 116"""

    file_stream.seek(116)
    file_stream.write(struct.pack("<Q", offset))


def write_version(file_stream, version):
    """Write version information"""

    v_major = version
    v_minor = 0

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
        f"MATLAB 5.0 MAT-file Platform. " f"Created on: {current_time} by matio"
    )
    if version == MAT_FILE_VERSIONS.V5:
        description += " with scipy"
    elif version == MAT_FILE_VERSIONS.HDF:
        description += f" using h5py v{h5py.__version__}"

    description_bytes = description.encode("ascii")[:116]  # Truncate if too long
    description_padded = description_bytes.ljust(116, b"\x20")

    file_stream.write(description_padded)
    write_subsystem_offset(file_stream)
    write_version(file_stream, version)
