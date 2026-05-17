"""Classes for loading MAT-file v4 files.
v4 format only supports 2D double (real and complex), char and sparse arrays.
"""

# Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers.
# All rights reserved.
#
# Modified by foreverallama (c) 2025
# https://github.com/foreverallama/matio
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import warnings
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from scipy.sparse import coo_array

from matio.utils.matclass import MatReadError, MatReadWarning
from matio.utils.matheaders import MAT4_HEADER_MOPT_MAX_VAL
from matio.utils.matutils import decode_char_arrays


class MAT_V4_MATRIX_TYPE(IntEnum):
    IEEE_LE = 0
    IEEE_BE = 1
    VAX_D_FLOAT = 2
    VAX_G_FLOAT = 3
    CRAY = 4


class MAT_V4_MATRIX_PRECISION(IntEnum):
    DOUBLE = 0
    SINGLE = 1
    INT32 = 2
    INT16 = 3
    UINT16 = 4
    UINT8 = 5


class MAT_V4_DATATYPE(IntEnum):
    FULL = 0
    CHAR = 1
    SPARSE = 2


mattype_to_numpy = {
    MAT_V4_MATRIX_PRECISION.DOUBLE: "f8",
    MAT_V4_MATRIX_PRECISION.SINGLE: "f4",
    MAT_V4_MATRIX_PRECISION.INT32: "i4",
    MAT_V4_MATRIX_PRECISION.INT16: "i2",
    MAT_V4_MATRIX_PRECISION.UINT16: "u2",
    MAT_V4_MATRIX_PRECISION.UINT8: "u1",
}


@dataclass
class VarHeader4:
    """Header for a variable in a MAT-file v4 file"""

    name: str
    floating_point_format: int
    dtype: np.dtype
    mat_datatype: int
    dims: tuple
    is_complex: bool
    payload_byte_size: int
    classname: str


def loadmat4(file_path, byte_order, variable_names):
    """Load MAT-file v4 variables"""

    with open(file_path, "rb") as f:
        MR = MatFile4Reader(f, byte_order)
        matfile_dict = MR.get_variables(variable_names)

    return matfile_dict


def whosmat4(file_path, byte_order):
    """List variables in MAT-file v4 file"""

    with open(file_path, "rb") as f:
        MR = MatFile4Reader(f, byte_order)
        vars = MR.list_variables()

    return vars


class MatFile4Reader:
    """Reader for MAT-file v4."""

    def __init__(self, mat_stream, byte_order):
        """Initialize reader for MAT-v4 files"""
        self.mat_stream = mat_stream
        self.byte_order = byte_order

    def end_of_stream(self):
        curpos = self.mat_stream.tell()
        self.mat_stream.seek(0, 2)
        endpos = self.mat_stream.tell()
        self.mat_stream.seek(curpos)
        return curpos == endpos

    def read_numeric_array(self, header):
        """Read numeric array."""

        dt = header.dtype
        payload_bytes = header.payload_byte_size

        data = self.mat_stream.read(payload_bytes)
        if header.is_complex:
            real = (
                np.frombuffer(data[: payload_bytes // 2], dtype=dt)
                .reshape(header.dims, order="F")
                .astype(np.float64)
            )

            imag = (
                np.frombuffer(data[payload_bytes // 2 :], dtype=dt)
                .reshape(header.dims, order="F")
                .astype(np.float64)
            )

            arr = real + 1j * imag
        else:
            arr = (
                np.frombuffer(data, dtype=dt)
                .reshape(header.dims, order="F")
                .astype(np.float64)
            )

        return arr

    def read_char_array(self, header):
        """Read char array."""

        dt = header.dtype
        payload_bytes = header.payload_byte_size

        data = self.mat_stream.read(payload_bytes)
        arr = (
            np.frombuffer(data, dtype=dt)
            .astype(np.uint8)
            .reshape(header.dims, order="F")
        )
        return decode_char_arrays(arr, "latin-1")

    def read_sparse_array(self, header):
        """Read sparse array.
        sparse array data is stored in (mrow, ncol) matrix.
            * ncol = 3 for real sparse
            * ncol = 4 for complex sparse

        Each row corresponds to the entry for a non-zero value (COO format):
            * First two columns are the row and column indices (1-based).
            * Third column is the real value of the non-zero entry
            * Fourth column (if present) is the imaginary value of the non-zero entry.

        The last row of the sparse array data contains the shape of the output matrix.
        Last value (or two values for complex sparse) is a padding 0.

        Note: The imagf flag in header is not set for complex sparse.
        """
        data = self.read_numeric_array(header)
        is_complex = data.shape[1] == 4
        out_shape = tuple(int(s) for s in data[-1, :2])

        # scipy coo requires int and 0-based indexing
        coo_i = data[:-1, 0].astype(int) - 1
        coo_j = data[:-1, 1].astype(int) - 1
        real = data[:-1, 2]
        if is_complex:
            imag = data[:-1, 3]
            coo_v = real + 1j * imag
        else:
            coo_v = real

        return coo_array((coo_v, (coo_i, coo_j)), shape=out_shape).tocsc()

    def read_var_header(self):
        """Read variable header"""
        MAT_V4_HEADER_BYTES = 20
        data = self.mat_stream.read(MAT_V4_HEADER_BYTES)
        header_dtype = np.dtype(f"{self.byte_order}i4")
        mopt, mrows, ncols, imagf, namlen = np.frombuffer(data, dtype=header_dtype)

        if mopt < 0 or mopt > MAT4_HEADER_MOPT_MAX_VAL:
            raise ValueError("Could not determine byte order for MAT-file v4 variable.")

        M = mopt // 1000
        O = (mopt // 100) % 10
        P = (mopt // 10) % 10
        T = mopt % 10

        if (O != 0) or (M < 0 or M > 4) or (P < 0 or P > 5) or (T < 0 or T > 2):
            raise MatReadError("Cannot read MAT-file v4, variable header is malformed.")

        dims = (mrows, ncols)
        is_complex = imagf == 1
        dtype = np.dtype(f"{self.byte_order}{mattype_to_numpy[P]}")

        name = (
            self.mat_stream.read(namlen).strip(b"\x00").decode("ascii")
        )  # FIXME: Verify if I need to add +1 byte for terminating null
        payload_bytes = np.prod(dims) * dtype.itemsize

        if is_complex and not T == MAT_V4_DATATYPE.SPARSE:
            payload_bytes *= 2

        if T == MAT_V4_DATATYPE.FULL:
            if is_complex:
                classname = "complex double"
            else:
                classname = "double"
        elif T == MAT_V4_DATATYPE.CHAR:
            classname = "char"
        elif T == MAT_V4_DATATYPE.SPARSE:
            classname = "sparse"

        header = VarHeader4(
            name, M, dtype, T, dims, is_complex, payload_bytes, classname
        )

        return header

    def read_var_array(self, header):
        """Read variable payload."""
        mtype = header.mat_datatype
        if mtype == MAT_V4_DATATYPE.FULL:
            arr = self.read_numeric_array(header)
        elif mtype == MAT_V4_DATATYPE.CHAR:
            arr = self.read_char_array(header)
        elif mtype == MAT_V4_DATATYPE.SPARSE:
            arr = self.read_sparse_array(header)
        else:
            raise TypeError(f"Unknown datatype {mtype} in variable {header.name}")

        return arr

    def get_variables(self, variable_names=None):
        """Get variables from stream"""

        self.mat_stream.seek(0)

        mdict = {}
        while not self.end_of_stream():
            header = self.read_var_header()
            name = header.name
            next_pos = self.mat_stream.tell() + header.payload_byte_size

            if name == "":
                self.mat_stream.seek(next_pos)
                continue
            if variable_names is not None and name not in variable_names:
                self.mat_stream.seek(next_pos)
                continue

            if header.floating_point_format < 0 or header.floating_point_format > 4:
                warnings.warn(
                    f"Variable {name!r} has unknown floating point format {MAT_V4_MATRIX_TYPE(header.floating_point_format).name}, skipping variable.",
                    MatReadWarning,
                    stacklevel=2,
                )
                self.mat_stream.seek(next_pos)
                continue

            if name in mdict:
                msg = f"Duplicate variable name {name!r} in file. Overwriting previous."
                warnings.warn(msg, MatReadWarning, stacklevel=2)

            try:
                res = self.read_var_array(header)
            except MatReadError as err:
                warnings.warn(
                    f'Unreadable variable "{name}", because "{err}"',
                    Warning,
                    stacklevel=2,
                )
                res = f"Read error: {err}"

            self.mat_stream.seek(next_pos)
            mdict[name] = res

            if variable_names is not None:
                variable_names.remove(name)
                if len(variable_names) == 0:
                    break

        return mdict

    def _read_sparse_array_shape(self, header):
        """Read shape of sparse array.
        Used in whosmat.
        Does not read the entire sparse array data, only the last row which contains the shape of the output matrix.
        Data buffer is stored in column-major layout.
        """
        mrows = header.dims[0]
        dtype = header.dtype
        itemsize = header.dtype.itemsize

        cur_pos = self.mat_stream.tell()
        pos_out_row = cur_pos + (mrows - 1) * itemsize
        post_out_col = cur_pos + (2 * mrows - 1) * itemsize

        self.mat_stream.seek(pos_out_row)
        out_row = np.frombuffer(self.mat_stream.read(itemsize), dtype=dtype, count=1)[0]

        self.mat_stream.seek(post_out_col)
        out_col = np.frombuffer(self.mat_stream.read(itemsize), dtype=dtype, count=1)[0]

        shape = (int(out_row), int(out_col))
        return shape

    def list_variables(self):
        """List variables from stream"""
        self.mat_stream.seek(0)
        vars = {}
        while not self.end_of_stream():
            header = self.read_var_header()
            name = header.name
            next_pos = self.mat_stream.tell() + header.payload_byte_size

            if name == "":
                self.mat_stream.seek(next_pos)
                continue

            if header.mat_datatype == MAT_V4_DATATYPE.SPARSE:
                shape = self._read_sparse_array_shape(header)
            else:
                shape = tuple(int(s) for s in header.dims)

            vars[name] = (shape, header.classname)
            self.mat_stream.seek(next_pos)

        return vars
