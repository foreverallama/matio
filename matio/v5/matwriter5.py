"""MAT-file 5 reading and writing"""

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

import math
import sys
import warnings
import zlib
from io import BytesIO

import numpy as np
import scipy.sparse

from matio.subsystem import MatSubsystem
from matio.utils.matclass import (
    MCOS_SUBSYSTEM_CLASS,
    EmptyMatStruct,
    MatlabCanonicalEmpty,
    MatlabEnumerationArray,
    MatlabFunction,
    MatlabObject,
    MatlabOpaque,
    MatlabOpaqueArray,
    MatWriteError,
    MatWriteWarning,
)
from matio.utils.matheaders import (
    MAT5_MAX_ARR_BYTES,
    MAT5_MAX_STRUCT_FIELDNAME_LEN,
    MAT_5_VERSION,
    write_file_header,
    write_subsystem_offset,
    write_version,
)
from matio.utils.matutils import (
    _get_string_arr_dtype,
    mat_numeric,
    matdims,
    strings_to_chars,
    to_writeable,
)

# Constants and helper objects
from matio.v5.matio5_params import (
    MDTYPES,
    NP_TO_MTYPES,
    NP_TO_MXTYPES,
    miTypes,
    mxTypes,
)

SYS_BYTE_ORDER = "<" if sys.byteorder == "little" else ">"

# Native byte ordered dtypes for convenience for writers
NDT_FILE_HDR = MDTYPES[SYS_BYTE_ORDER]["dtypes"]["file_header"]
NDT_TAG_FULL = MDTYPES[SYS_BYTE_ORDER]["dtypes"]["tag_full"]
NDT_TAG_SMALL = MDTYPES[SYS_BYTE_ORDER]["dtypes"]["tag_smalldata"]
NDT_ARRAY_FLAGS = MDTYPES[SYS_BYTE_ORDER]["dtypes"]["array_flags"]


def savemat5(file_path, mdict, global_vars, saveobj_classes, oned_as, do_compression):
    """Write data to MAT-5file."""

    with open(file_path, "wb") as f:
        write_file_header(f, version=MAT_5_VERSION)
        MW = MatFile5Writer(f, oned_as=oned_as)
        MW.subsystem = MatSubsystem(
            byte_order=SYS_BYTE_ORDER, oned_as=oned_as, saveobj_classes=saveobj_classes
        )
        MW.subsystem.init_save()

        MW.put_variables(mdict, global_vars, do_compression)
        subsystem = MW.subsystem.set_subsystem()
        if subsystem is None:
            return

        subsystem_offset = f.tell()
        subsys_stream = write_subsystem(subsystem, oned_as)

        temp_dict = {"__subsystem__": subsys_stream}
        MW.put_variables(temp_dict, global_vars=[], do_compression=do_compression)

        write_subsystem_offset(f, subsystem_offset)

    return


def write_subsystem(subsystem, oned_as):
    """Write subsystem data to subsys stream"""

    subsys_stream = BytesIO()
    subsys_stream.seek(0)
    write_version(subsys_stream, MAT_5_VERSION)
    subsys_stream.write(b"\x00" * 4)  # Padding

    MW = MatFile5Writer(subsys_stream, oned_as=oned_as)
    temp_dict = {"__subsystem__": subsystem}

    MW.put_variables(temp_dict, global_vars=[], do_compression=False)
    subsys_view = np.frombuffer(subsys_stream.getbuffer(), dtype=np.uint8)
    return subsys_view


class VarWriter5:
    """Generic matlab matrix writing class"""

    mat_tag = np.zeros((), NDT_TAG_FULL)
    mat_tag["mdtype"] = miTypes.miMATRIX

    def __init__(self, file_writer):
        self.file_stream = file_writer.file_stream
        self.oned_as = file_writer.oned_as
        self.subsystem = file_writer.subsystem

        # These are used for top level writes, and unset after
        self._var_name = None
        self._var_is_global = False

    def write_bytes(self, arr):
        self.file_stream.write(arr.tobytes(order="F"))

    def write_string(self, s):
        self.file_stream.write(s)

    def write_element(self, arr, mdtype=None):
        """write tag and data"""
        if mdtype is None:
            mdtype = NP_TO_MTYPES[arr.dtype.str[1:]]
        # Array needs to be in native byte order
        if arr.dtype.byteorder == (SYS_BYTE_ORDER == ">"):
            arr = arr.byteswap().view(arr.dtype.newbyteorder())
        byte_count = arr.size * arr.itemsize
        if byte_count <= 4:
            self.write_smalldata_element(arr, mdtype, byte_count)
        else:
            self.write_regular_element(arr, mdtype, byte_count)

    def write_smalldata_element(self, arr, mdtype, byte_count):
        # write tag with embedded data
        tag = np.zeros((), NDT_TAG_SMALL)
        tag["byte_count_mdtype"] = (byte_count << 16) + mdtype
        # if arr.tobytes is < 4, the element will be zero-padded as needed.
        tag["data"] = arr.tobytes(order="F")
        self.write_bytes(tag)

    def write_regular_element(self, arr, mdtype, byte_count):
        # write tag, data
        tag = np.zeros((), NDT_TAG_FULL)
        tag["mdtype"] = mdtype
        tag["byte_count"] = byte_count
        self.write_bytes(tag)
        self.write_bytes(arr)
        # pad to next 64-bit boundary
        bc_mod_8 = byte_count % 8
        if bc_mod_8:
            self.file_stream.write(b"\x00" * (8 - bc_mod_8))

    def write_header(self, shape, mclass, is_complex=False, is_logical=False, nzmax=0):
        """Write header for given data options
        shape : sequence
           array shape
        mclass      - mat5 matrix class
        is_complex  - True if matrix is complex
        is_logical  - True if matrix is logical
        nzmax        - max non zero elements for sparse arrays

        We get the name and the global flag from the object, and reset
        them to defaults after we've used them
        """
        # get name and is_global from one-shot object store
        name = self._var_name
        is_global = self._var_is_global
        # initialize the top-level matrix tag, store position
        self._mat_tag_pos = self.file_stream.tell()
        self.write_bytes(self.mat_tag)
        # write array flags (complex, global, logical, class, nzmax)
        af = np.zeros((), NDT_ARRAY_FLAGS)
        af["data_type"] = miTypes.miUINT32
        af["byte_count"] = 8
        flags = is_complex << 3 | is_global << 2 | is_logical << 1
        af["flags_class"] = mclass | flags << 8
        af["nzmax"] = nzmax
        self.write_bytes(af)
        # shape
        if shape is not None:
            self.write_element(np.array(shape, dtype="i4"))
        # write name
        name = np.asarray(name)
        if name == "" or name == b"":  # empty string zero-terminated
            self.write_smalldata_element(name, miTypes.miINT8, 0)
        else:
            self.write_element(name, miTypes.miINT8)
        # reset the one-shot store to defaults
        self._var_name = ""
        self._var_is_global = False

    def update_matrix_tag(self, start_pos):
        curr_pos = self.file_stream.tell()
        self.file_stream.seek(start_pos)
        byte_count = curr_pos - start_pos - 8
        if byte_count >= MAT5_MAX_ARR_BYTES:
            raise MatWriteError(
                "Matrix too large to save with MAT-5 file format. Use v7.3 format instead."
            )
        self.mat_tag["byte_count"] = byte_count
        self.write_bytes(self.mat_tag)
        self.file_stream.seek(curr_pos)

    def write_top(self, arr, name, is_global):
        """Write top-level variable"""
        # these are set before the top-level header write, and unset at
        # the end of the same write, because they do not apply for lower levels
        self._var_is_global = is_global
        self._var_name = name
        self.write(arr)

    def write(self, arr):
        """Write data element to MAT-file"""

        mat_tag_pos = self.file_stream.tell()

        narr = to_writeable(arr, self.oned_as)

        if scipy.sparse.issparse(narr):
            self.write_sparse(narr)
        elif isinstance(narr, MatlabCanonicalEmpty):
            self.write_canonical_empty(narr)
        elif isinstance(narr, MatlabObject):
            self.write_object(narr)
        elif isinstance(narr, MatlabFunction):
            self.write_function_handle(narr)
        elif isinstance(narr, EmptyMatStruct):
            self.write_empty_struct(narr)
        elif isinstance(
            narr, (MatlabOpaque, MatlabOpaqueArray, MatlabEnumerationArray)
        ):
            self.write_opaque(narr)
        elif narr.dtype.fields:  # struct array
            self.write_struct(narr)
        elif narr.dtype.hasobject:  # cell array
            self.write_cells(narr)
        elif narr.dtype.kind in ("U", "S"):
            self.write_char(narr)
        elif narr.dtype.kind in ("i", "u", "f", "c", "b"):
            self.write_numeric(narr)

        self.update_matrix_tag(mat_tag_pos)

    def write_numeric(self, arr):
        is_complex = arr.dtype.kind == "c"
        is_logical = arr.dtype.kind == "b"

        arr, _, _ = mat_numeric(arr, version=MAT_5_VERSION)
        mclass = NP_TO_MXTYPES[arr.dtype.str[1:]]

        self.write_header(
            matdims(arr, self.oned_as),
            mclass,
            is_complex=is_complex,
            is_logical=is_logical,
        )

        if is_complex:
            self.write_element(arr.real)
            self.write_element(arr.imag)
        else:
            self.write_element(arr)

    def write_char(self, arr, codec="ascii"):
        """Write char array (from numpy string array)"""

        if arr.size == 0 or np.all(arr == ""):
            # Empty String Array
            # MATLAB only recognizes 0-D char array
            shape = (0,) * np.max([arr.ndim, 2])
            self.write_header(shape, mxTypes.mxCHAR_CLASS)
            self.write_smalldata_element(arr, miTypes.miUTF8, 0)
            return

        # Convert string to char array
        arr = strings_to_chars(arr)
        # We have to write the shape directly, because we are going
        # recode the characters, and the resulting stream of chars
        # may have a different length
        shape = arr.shape
        self.write_header(shape, mxTypes.mxCHAR_CLASS)
        if arr.dtype.kind == "U" and arr.size:
            # Make one long string from all the characters. We need to
            # transpose here, because we're flattening the array, before
            # we write the bytes. The bytes have to be written in
            # Fortran order.
            n_chars = math.prod(shape)
            st_arr = np.ndarray(
                shape=(), dtype=_get_string_arr_dtype(arr, n_chars), buffer=arr.T.copy()
            )  # Fortran order
            # Recode with codec to give byte string
            st = st_arr.item().encode(codec)
            # Reconstruct as 1-D byte array
            arr = np.ndarray(shape=(len(st),), dtype="S1", buffer=st)
        self.write_element(arr, mdtype=miTypes.miUTF8)

    def write_sparse(self, arr):
        """Sparse matrices are 2D"""
        A = arr.tocsc()  # convert to sparse CSC format
        A.sort_indices()  # MATLAB expects sorted row indices
        is_complex = A.dtype.kind == "c"
        is_logical = A.dtype.kind == "b"
        nz = A.nnz
        self.write_header(
            matdims(arr, self.oned_as),
            mxTypes.mxSPARSE_CLASS,
            is_complex=is_complex,
            is_logical=is_logical,
            # matlab won't load file with 0 nzmax
            nzmax=1 if nz == 0 else nz,
        )
        self.write_element(A.indices.astype("i4"))
        self.write_element(A.indptr.astype("i4"))
        self.write_element(A.data.real)
        if is_complex:
            self.write_element(A.data.imag)

    def write_cells(self, arr):
        """Write cell array"""
        self.write_header(matdims(arr, self.oned_as), mxTypes.mxCELL_CLASS)
        # loop over data, column major
        A = np.atleast_2d(arr).flatten("F")
        for el in A:
            self.write(el)

    def write_empty_struct(self, arr):
        """Write empty struct array"""
        self.write_header(arr.shape, mxTypes.mxSTRUCT_CLASS)
        # max field name length set to 1 in an example matlab struct
        self.write_element(np.array(1, dtype=np.int32))
        # Field names element is empty
        self.write_element(np.array([], dtype=np.int8))

    def write_struct(self, arr):
        """Write struct array"""
        self.write_header(matdims(arr, self.oned_as), mxTypes.mxSTRUCT_CLASS)
        self._write_items(arr)

    def _write_items(self, arr):
        """Write struct fields"""
        # write fieldnames
        fieldnames = [f[0] for f in arr.dtype.descr]
        length = max([len(fieldname) for fieldname in fieldnames]) + 1
        if length > MAT5_MAX_STRUCT_FIELDNAME_LEN:
            raise ValueError(
                f"Field names are restricted to {MAT5_MAX_STRUCT_FIELDNAME_LEN - 1} characters"
            )
        self.write_element(np.array([length], dtype="i4"))
        self.write_element(
            np.array(fieldnames, dtype=f"S{length}"), mdtype=miTypes.miINT8
        )
        A = np.atleast_2d(arr).flatten("F")
        for el in A:
            for f in fieldnames:
                self.write(el[f])

    def write_object(self, arr):
        """Write MatlabObject"""
        self.write_header(matdims(arr, self.oned_as), mxTypes.mxOBJECT_CLASS)
        self.write_element(np.array(arr.classname, dtype="S"), mdtype=miTypes.miINT8)
        self._write_items(arr)

    def write_function_handle(self, arr):
        """Write MatlabFunction"""
        self.write_header(matdims(arr, self.oned_as), mxTypes.mxFUNCTION_CLASS)
        self.write(arr.view(np.ndarray))

    def write_opaque(self, arr):
        """Array Flags, Var Name, Type System, Class Name, Metadata"""
        self.write_header(None, mxTypes.mxOPAQUE_CLASS)
        self.write_element(np.array(arr.type_system, dtype="S"), mdtype=miTypes.miINT8)
        self.write_element(np.array(arr.classname, dtype="S"), mdtype=miTypes.miINT8)
        if arr.classname == MCOS_SUBSYSTEM_CLASS:
            self.write(arr.properties)
        else:
            if isinstance(arr, MatlabEnumerationArray):
                objmetadata = self.subsystem.set_enumeration_metadata(arr)
            else:
                objmetadata = self.subsystem.set_object_metadata(arr)
            self.write(objmetadata)

    def write_canonical_empty(self, arr):
        """Write canonical empty array"""
        self._mat_tag_pos = self.file_stream.tell()
        self.write_bytes(self.mat_tag)  # 0 byte miMATRIX Tag


class MatFile5Writer:
    """Class for writing mat5 files"""

    def __init__(self, file_stream, oned_as):
        """Initialize writer for MAT-5 format"""
        self.file_stream = file_stream
        self.oned_as = oned_as
        self._matrix_writer = None
        self.subsystem = None

    def put_variables(self, mdict, global_vars, do_compression):
        """Write variables to MATLAB file."""

        self._matrix_writer = VarWriter5(self)

        for name, var in mdict.items():
            if name == "__subsystem__":
                name = ""
            elif name[0] in "_0123456789":
                msg = f"Variable names cannot start with '{name[0]}'. Skipping '{name}'"
                warnings.warn(msg, MatWriteWarning, stacklevel=2)
                continue
            is_global = name in global_vars
            if do_compression:
                stream = BytesIO()
                self._matrix_writer.file_stream = stream
                self._matrix_writer.write_top(var, name.encode("latin1"), is_global)
                out_str = zlib.compress(stream.getvalue())
                tag = np.empty((), NDT_TAG_FULL)
                tag["mdtype"] = miTypes.miCOMPRESSED
                tag["byte_count"] = len(out_str)
                self.file_stream.write(tag.tobytes())
                self.file_stream.write(out_str)
            else:
                self._matrix_writer.write_top(var, name.encode("latin1"), is_global)
