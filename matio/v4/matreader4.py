"""Classes for loading MAT-file v4 files"""

import sys
import warnings
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import scipy.sparse

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
    dtype: np.dtype
    mat_datatype: int
    dims: tuple
    is_complex: bool
    payload_byte_size: int


def loadmat4(file_path, byte_order, variable_names):
    """Load MAT-file v4 variables"""

    with open(file_path, "rb") as f:
        MR = MatFile4Reader(f, byte_order)
        matfile_dict = MR.get_variables(variable_names)

    return matfile_dict


def whosmat4(file_path, byte_order):
    """List variables in MAT-file v4 file"""

    # TODO: Test
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
        """Read sparse array."""
        # TODO: Implement and Test
        pass

        # Notes
        # -----
        # MATLAB 4 real sparse arrays are saved in a N+1 by 3 array format, where
        # N is the number of non-zero values. Column 1 values [0:N] are the
        # (1-based) row indices of the each non-zero value, column 2 [0:N] are the
        # column indices, column 3 [0:N] are the (real) values. The last values
        # [-1,0:2] of the rows, column indices are shape[0] and shape[1]
        # respectively of the output matrix. The last value for the values column
        # is a padding 0. mrows and ncols values from the header give the shape of
        # the stored matrix, here [N+1, 3]. Complex data are saved as a 4 column
        # matrix, where the fourth column contains the imaginary component; the
        # last value is again 0. Complex sparse data do *not* have the header
        # ``imagf`` field set to True; the fact that the data are complex is only
        # detectable because there are 4 storage columns.

        # res = self.read_sub_array(hdr)
        # tmp = res[:-1,:]
        # # All numbers are float64 in Matlab, but SciPy sparse expects int shape
        # dims = (int(res[-1,0]), int(res[-1,1]))
        # I = np.ascontiguousarray(tmp[:,0],dtype='intc')  # fixes byte order also
        # J = np.ascontiguousarray(tmp[:,1],dtype='intc')
        # I -= 1  # for 1-based indexing
        # J -= 1
        # if res.shape[1] == 3:
        #     V = np.ascontiguousarray(tmp[:,2],dtype='float')
        # else:
        #     V = np.ascontiguousarray(tmp[:,2],dtype='complex')
        #     V.imag = tmp[:,3]
        # return scipy.sparse.coo_array((V,(I,J)), dims)

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
            if M not in (0, 1):
                raise NotImplementedError(
                    f"VAX and CRAY floating point formats are not supported."
                )
            else:
                raise MatReadError(
                    "Cannot read MAT-file v4 variable, variable header is malformed."
                )

        dims = (mrows, ncols)
        is_complex = imagf == 1
        dtype = np.dtype(f"{self.byte_order}{mattype_to_numpy[P]}")
        # TODO: Check dtype conversion for VAX and CRAY formats

        name = (
            self.mat_stream.read(namlen).strip(b"\x00").decode("ascii")
        )  # TODO: Verify if I need to add +1 byte for terminating null
        payload_bytes = np.prod(dims) * dtype.itemsize

        if is_complex and not T == MAT_V4_DATATYPE.SPARSE:
            payload_bytes *= 2

        header = VarHeader4(name, dtype, T, dims, is_complex, payload_bytes)

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

            if name in mdict:
                msg = f"Duplicate variable name {name!r} in file. Overwriting previous."
                warnings.warn(msg, MatReadWarning, stacklevel=2)
            if name == "":
                # TODO: Verify if v4 files can have variables with no names
                self.mat_stream.seek(next_pos)
                continue
            if variable_names is not None and name not in variable_names:
                self.mat_stream.seek(next_pos)
                continue

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

    def list_variables(self):
        """List variables from stream"""
        self.mat_stream.seek(0)
        vars = []
        while not self.end_of_stream():
            header = self.read_var_header()
            name = header.name
            next_pos = self.mat_stream.tell() + header.payload_byte_size

            if name == "":
                self.mat_stream.seek(next_pos)
                continue

            shape = self._matrix_reader.shape_from_header(header)
            if header.mat_datatype == MAT_V4_DATATYPE.SPARSE:
                info = "sparse"
            else:
                info = header.dtype.name
            vars.append((name, shape, info))
            self.mat_stream.seek(next_pos)
        return vars
