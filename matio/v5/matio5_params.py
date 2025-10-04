"""Constants and classes for matlab 5 read and write"""

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

from enum import IntEnum

import numpy as np

__all__ = [
    "MDTYPES",
    "NP_TO_MTYPES",
    "NP_TO_MXTYPES",
    "codecs_template",
    "mclass_dtypes_template",
    "mclass_info",
    "mdtypes_template",
    "miTypes",
    "mxTypes",
]


class miTypes(IntEnum):
    miINT8 = 1
    miUINT8 = 2
    miINT16 = 3
    miUINT16 = 4
    miINT32 = 5
    miUINT32 = 6
    miSINGLE = 7
    miDOUBLE = 9
    miINT64 = 12
    miUINT64 = 13
    miMATRIX = 14
    miCOMPRESSED = 15
    miUTF8 = 16
    miUTF16 = 17
    miUTF32 = 18


class mxTypes(IntEnum):
    mxCELL_CLASS = 1
    mxSTRUCT_CLASS = 2
    mxOBJECT_CLASS = 3
    mxCHAR_CLASS = 4
    mxSPARSE_CLASS = 5
    mxDOUBLE_CLASS = 6
    mxSINGLE_CLASS = 7
    mxINT8_CLASS = 8
    mxUINT8_CLASS = 9
    mxINT16_CLASS = 10
    mxUINT16_CLASS = 11
    mxINT32_CLASS = 12
    mxUINT32_CLASS = 13
    mxINT64_CLASS = 14
    mxUINT64_CLASS = 15
    mxFUNCTION_CLASS = 16
    mxOPAQUE_CLASS = 17


mdtypes_template = {
    miTypes.miINT8: "i1",
    miTypes.miUINT8: "u1",
    miTypes.miINT16: "i2",
    miTypes.miUINT16: "u2",
    miTypes.miINT32: "i4",
    miTypes.miUINT32: "u4",
    miTypes.miSINGLE: "f4",
    miTypes.miDOUBLE: "f8",
    miTypes.miINT64: "i8",
    miTypes.miUINT64: "u8",
    miTypes.miUTF8: "u1",
    miTypes.miUTF16: "u2",
    miTypes.miUTF32: "u4",
    "file_header": [
        ("description", "S116"),
        ("subsystem_offset", "i8"),
        ("version", "u2"),
        ("endian_test", "S2"),
    ],
    "tag_full": [("mdtype", "u4"), ("byte_count", "u4")],
    "tag_smalldata": [("byte_count_mdtype", "u4"), ("data", "S4")],
    "array_flags": [
        ("data_type", "u4"),
        ("byte_count", "u4"),
        ("flags_class", "u4"),
        ("nzmax", "u4"),
    ],
    "U1": "U1",
}

mclass_dtypes_template = {
    mxTypes.mxINT8_CLASS: "i1",
    mxTypes.mxUINT8_CLASS: "u1",
    mxTypes.mxINT16_CLASS: "i2",
    mxTypes.mxUINT16_CLASS: "u2",
    mxTypes.mxINT32_CLASS: "i4",
    mxTypes.mxUINT32_CLASS: "u4",
    mxTypes.mxINT64_CLASS: "i8",
    mxTypes.mxUINT64_CLASS: "u8",
    mxTypes.mxSINGLE_CLASS: "f4",
    mxTypes.mxDOUBLE_CLASS: "f8",
}

mclass_info = {
    mxTypes.mxINT8_CLASS: "int8",
    mxTypes.mxUINT8_CLASS: "uint8",
    mxTypes.mxINT16_CLASS: "int16",
    mxTypes.mxUINT16_CLASS: "uint16",
    mxTypes.mxINT32_CLASS: "int32",
    mxTypes.mxUINT32_CLASS: "uint32",
    mxTypes.mxINT64_CLASS: "int64",
    mxTypes.mxUINT64_CLASS: "uint64",
    mxTypes.mxSINGLE_CLASS: "single",
    mxTypes.mxDOUBLE_CLASS: "double",
    mxTypes.mxCELL_CLASS: "cell",
    mxTypes.mxSTRUCT_CLASS: "struct",
    mxTypes.mxOBJECT_CLASS: "object",
    mxTypes.mxCHAR_CLASS: "char",
    mxTypes.mxSPARSE_CLASS: "sparse",
    mxTypes.mxFUNCTION_CLASS: "function_handle",
    mxTypes.mxOPAQUE_CLASS: "opaque",
}

NP_TO_MTYPES = {
    "f8": miTypes.miDOUBLE,
    "c32": miTypes.miDOUBLE,
    "c24": miTypes.miDOUBLE,
    "c16": miTypes.miDOUBLE,
    "f4": miTypes.miSINGLE,
    "c8": miTypes.miSINGLE,
    "i8": miTypes.miINT64,
    "i4": miTypes.miINT32,
    "i2": miTypes.miINT16,
    "i1": miTypes.miINT8,
    "u8": miTypes.miUINT64,
    "u4": miTypes.miUINT32,
    "u2": miTypes.miUINT16,
    "u1": miTypes.miUINT8,
    "S1": miTypes.miUINT8,
    "U1": miTypes.miUTF16,
    "b1": miTypes.miUINT8,
}


NP_TO_MXTYPES = {
    "f8": mxTypes.mxDOUBLE_CLASS,
    "c32": mxTypes.mxDOUBLE_CLASS,
    "c24": mxTypes.mxDOUBLE_CLASS,
    "c16": mxTypes.mxDOUBLE_CLASS,
    "f4": mxTypes.mxSINGLE_CLASS,
    "c8": mxTypes.mxSINGLE_CLASS,
    "i8": mxTypes.mxINT64_CLASS,
    "i4": mxTypes.mxINT32_CLASS,
    "i2": mxTypes.mxINT16_CLASS,
    "i1": mxTypes.mxINT8_CLASS,
    "u8": mxTypes.mxUINT64_CLASS,
    "u4": mxTypes.mxUINT32_CLASS,
    "u2": mxTypes.mxUINT16_CLASS,
    "u1": mxTypes.mxUINT8_CLASS,
    "S1": mxTypes.mxUINT8_CLASS,
    "b1": mxTypes.mxUINT8_CLASS,
}

# Before release v7.1 (release 14) matlab (TM) used the system
# default character encoding scheme padded out to 16-bits. Release 14
# and later use Unicode. When saving character data, R14 checks if it
# can be encoded in 7-bit ascii, and saves in that format if so.

codecs_template = {
    miTypes.miUTF8: {"codec": "utf_8", "width": 1},
    miTypes.miUTF16: {"codec": "utf_16", "width": 2},
    miTypes.miUTF32: {"codec": "utf_32", "width": 4},
}


def _convert_codecs(template, byte_order):
    """Convert codec template mapping to byte order

    Set codecs not on this system to None

    Parameters
    ----------
    template : mapping
       key, value are respectively codec name, and root name for codec
       (without byte order suffix)
    byte_order : {'<', '>'}
       code for little or big endian

    Returns
    -------
    codecs : dict
       key, value are name, codec (as in .encode(codec))
    """
    codecs = {}
    postfix = byte_order == "<" and "_le" or "_be"
    for k, v in template.items():
        codec = v["codec"]
        try:
            " ".encode(codec)
        except LookupError:
            codecs[k] = None
            continue
        if v["width"] > 1:
            codec += postfix
        codecs[k] = codec
    return codecs.copy()


def convert_dtypes(dtype_template, order_code):
    """Convert dtypes in mapping to given order

    Parameters
    ----------
    dtype_template : mapping
       mapping with values returning numpy dtype from ``np.dtype(val)``
    order_code : str
       an order code suitable for using in ``dtype.newbyteorder()``

    Returns
    -------
    dtypes : mapping
       mapping where values have been replaced by
       ``np.dtype(val).newbyteorder(order_code)``

    """
    dtypes = dtype_template.copy()
    for k in dtypes:
        dtypes[k] = np.dtype(dtypes[k]).newbyteorder(order_code)
    return dtypes


MDTYPES = {}
for _bytecode in "<>":
    _def = {
        "dtypes": convert_dtypes(mdtypes_template, _bytecode),
        "classes": convert_dtypes(mclass_dtypes_template, _bytecode),
        "codecs": _convert_codecs(codecs_template, _bytecode),
    }
    MDTYPES[_bytecode] = _def
