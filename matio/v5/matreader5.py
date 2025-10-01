"""Reader for MAT-5 version files"""

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

import sys
import warnings
from io import BytesIO

from matio.subsystem import MatSubsystem
from matio.utils.matclass import MatReadError, MatReadWarning
from matio.utils.matheaders import MAT5_HEADER_SIZE_BYTES, check_mat_version
from matio.utils.matutils import shape_from_metadata

# Constants and helper objects
from matio.v5.matio5_params import mclass_info, miTypes, mxTypes

# Reader object for matlab 5 format variables
from ._mio5_utils import VarReader5
from ._streams import ZlibInputStream

SYS_BYTE_ORDER = "<" if sys.byteorder == "little" else ">"


def loadmat5(
    file_path, subsystem_offset, byte_order, variable_names, raw_data, add_table_attrs
):
    """Load MATLAB file."""

    with open(file_path, "rb") as f:

        subsystem = read_subsystem(
            f,
            subsystem_offset,
            byte_order,
            raw_data=raw_data,
            add_table_attrs=add_table_attrs,
        )

        MR = MatFile5Reader(f, byte_order=byte_order, subsystem=subsystem)
        matfile_dict = MR.get_variables(variable_names)

    return matfile_dict


def whosmat5(file_path, byte_order):
    """List variables in MATLAB file."""

    with open(file_path, "rb") as f:
        MR = MatFile5Reader(f, byte_order=byte_order)
        vars = MR.list_variables()

    return vars


class MatFile5Reader:
    """Reader for Mat 5 mat files"""

    def __init__(
        self,
        mat_stream,
        byte_order=SYS_BYTE_ORDER,
        subsystem=None,
    ):
        """Initialize reader for Mat-5 files"""

        self.mat_stream = mat_stream
        self.dtypes = {}

        self.byte_order = byte_order
        self.uint16_codec = sys.getdefaultencoding()  # scipy arg

        self._file_reader = None
        self._matrix_reader = None
        self.subsystem = subsystem

    def end_of_stream(self):
        curpos = self.mat_stream.tell()
        self.mat_stream.seek(0, 2)
        endpos = self.mat_stream.tell()
        self.mat_stream.seek(curpos)
        return curpos == endpos

    def initialize_read(self):
        """Initialize file-level and matrix-level readers"""
        # reader for top level stream. We need this extra top-level
        # reader because we use the matrix_reader object to contain
        # compressed matrices (so they have their own stream)
        self._file_reader = VarReader5(self)
        self._matrix_reader = VarReader5(self)

    def read_var_header(self):
        """Read variable headers"""

        mdtype, byte_count = self._file_reader.read_full_tag()
        if not byte_count > 0:
            raise MatReadError("Did not read any bytes")
        next_pos = self.mat_stream.tell() + byte_count
        if mdtype == miTypes.miCOMPRESSED:
            # Make new stream from compressed data
            stream = ZlibInputStream(self.mat_stream, byte_count)
            self._matrix_reader.set_stream(stream)
            check_stream_limit = True
            mdtype, byte_count = self._matrix_reader.read_full_tag()
        else:
            check_stream_limit = False
            self._matrix_reader.set_stream(self.mat_stream)
        if not mdtype == miTypes.miMATRIX:
            raise TypeError(f"Expecting miMATRIX type here, got {mdtype}")
        header = self._matrix_reader.read_header(check_stream_limit)
        return header, next_pos

    def read_var_array(self, header, process=True):
        """Read variable array from stream"""
        return self._matrix_reader.array_from_header(header, process)

    def get_variables(self, variable_names=None):
        """Get variables from stream"""

        self.mat_stream.seek(MAT5_HEADER_SIZE_BYTES)
        self.initialize_read()

        mdict = {"__globals__": []}

        while not self.end_of_stream():
            hdr, next_position = self.read_var_header()
            name = hdr.name.decode("ascii")
            if name in mdict:
                msg = f"Duplicate variable name {name!r} in file. Overwriting previous."
                warnings.warn(msg, MatReadWarning, stacklevel=2)
            if name == "":
                # Skip subsystem
                self.mat_stream.seek(next_position)
                continue
            if variable_names is not None and name not in variable_names:
                self.mat_stream.seek(next_position)
                continue
            try:
                res = self.read_var_array(hdr, True)
            except MatReadError as err:
                warnings.warn(
                    f'Unreadable variable "{name}", because "{err}"',
                    Warning,
                    stacklevel=2,
                )
                res = f"Read error: {err}"
            self.mat_stream.seek(next_position)
            mdict[name] = res
            if hdr.is_global:
                mdict["__globals__"].append(name)
            if variable_names is not None:
                variable_names.remove(name)
                if len(variable_names) == 0:
                    break

        return mdict

    def list_variables(self):
        """List variables from stream"""
        self.mat_stream.seek(MAT5_HEADER_SIZE_BYTES)
        self.initialize_read()
        vars = []
        while not self.end_of_stream():
            hdr, next_position = self.read_var_header()
            name = hdr.name.decode("ascii")
            if name == "":
                # Skip subsystem
                self.mat_stream.seek(next_position)
                continue
            if hdr.mclass == mxTypes.mxOPAQUE_CLASS:
                shape = self.read_opaque_class_shape(hdr)
            else:
                shape = self._matrix_reader.shape_from_header(hdr)

            if hdr.is_logical:
                info = "logical"
            elif hdr.classname is not None:
                info = hdr.classname
            else:
                info = mclass_info.get(hdr.mclass, "unknown")

            vars.append((name, shape, info))

            self.mat_stream.seek(next_position)
        return vars

    def read_opaque_class_shape(self, hdr):
        """Read class info for OBJECT and OPAQUE types"""

        objmetadata = self._matrix_reader.array_from_header(hdr, False)
        shape = shape_from_metadata(objmetadata)
        return shape


def load_subsys_stream(f, subsystem_offset, byte_order):
    """Load subsystem stream if present"""

    f.seek(subsystem_offset)
    subsys_stream_reader = MatFile5Reader(f, byte_order=byte_order)
    subsys_stream_reader.initialize_read()
    hdr, _ = subsys_stream_reader.read_var_header()

    try:
        res = subsys_stream_reader.read_var_array(hdr, False)
    except MatReadError as err:
        raise MatReadError(f'Unreadable subsystem data because "{err}"')

    # Subsystem data is a file stream written as uint8 data
    # FIXME: Can this be optimized?
    return BytesIO(res)


def read_subsystem(f, subsystem_offset, byte_order, raw_data, add_table_attrs):
    """Read subsystem data if present"""

    if subsystem_offset <= 0:
        return None

    subsys_stream = load_subsys_stream(f, subsystem_offset, byte_order)
    subsys_stream.seek(0)
    byte_order, _ = check_mat_version(subsys_stream.read(4))
    subsys_stream.read(4)  # Discard Padding

    subsystem_reader = MatFile5Reader(subsys_stream, byte_order=byte_order)
    subsystem_reader.initialize_read()

    hdr, _ = subsystem_reader.read_var_header()
    try:
        subsys_data = subsystem_reader.read_var_array(hdr, False)
    except MatReadError as err:
        raise MatReadError(f'Unreadable subsystem data because "{err}"')

    subsystem = MatSubsystem(
        byte_order, raw_data=raw_data, add_table_attrs=add_table_attrs
    )
    subsystem.load_subsystem(subsys_data)

    return subsystem
