import struct
import sys
import time

import h5py
import numpy as np

MAT5_HEADER_SIZE_BYTES = 128
MAT5_MAX_ARR_BYTES = 2**32
MAT5_MAX_STRUCT_FIELDNAME_LEN = 64
MAT_5_VERSION = 1

MAT_HDF_VERSION = 2
MAT_HDF_USER_BLOCK_BYTES = 512

MCOS_MAGIC_NUMBER = 0xDD000000

MAT_VERSIONS = {"v7": MAT_5_VERSION, "v7.3": MAT_HDF_VERSION}


def check_mat_version(data):
    """Reads MAT-file version from header data"""

    if data[2:] == b"IM":
        byte_order = "<"
    elif data[2:] == b"MI":
        byte_order = ">"
    else:
        raise ValueError("Invalid endian indicator in MAT-file header")

    v_major, v_minor = int(data[1]), int(data[0])
    if byte_order != "<":
        v_major, v_minor = v_minor, v_major

    if v_major not in (MAT_5_VERSION, MAT_HDF_VERSION):
        raise ValueError(f"Unknown MAT-file version {v_major}.{v_minor}")

    return byte_order, v_major


def read_mat_header(file_path):
    """Reads MAT-file header and returns version information"""

    with open(file_path, "rb") as f:
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
    if version == MAT_5_VERSION:
        description += " with scipy"
    elif version == MAT_HDF_VERSION:
        description += f" using h5py v{h5py.__version__}"

    description_bytes = description.encode("ascii")[:116]  # Truncate if too long
    description_padded = description_bytes.ljust(116, b"\x20")

    file_stream.write(description_padded)
    write_subsystem_offset(file_stream)
    write_version(file_stream, version)
