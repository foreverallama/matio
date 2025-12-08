"""MATLAB MAT-file version 7.3 (HDF5) reader."""

import warnings

import h5py
import numpy as np
from scipy.sparse import csc_array

from matio.subsystem import MatSubsystem
from matio.utils.matclass import (
    ENUMERATION_INSTANCE_TAG,
    MCOS_SUBSYSTEM_CLASS,
    EmptyMatStruct,
    IntegerDecodingHint,
    MatlabCanonicalEmpty,
    MatlabFunction,
    MatlabObject,
    MatReadWarning,
    ObjectDecodingHint,
)
from matio.utils.matconvert import guess_type_system
from matio.utils.matheaders import (
    MAT_HDF_ATTRS,
    MAT_HDF_REFS_GROUP,
    MAT_HDF_SUBSYS_GROUP,
    MCOS_MAGIC_NUMBER,
)
from matio.utils.matutils import (
    chars_to_strings,
    matlab_class_to_dtype,
    shape_from_metadata,
)


def loadmat7(
    file_path,
    byte_order,
    variable_names,
    raw_data,
    add_table_attrs,
):
    """Reads MAT-file version 7.3 (HDF5) files."""

    with h5py.File(file_path, "r") as f:
        subsystem = read_subsystem(f, byte_order, raw_data, add_table_attrs)
        mat_reader = MatRead7(f, subsystem)
        matfile_dict = mat_reader.get_variables(variable_names)

    return matfile_dict


def whosmat7(file_path):
    """Lists variables in MAT-file version 7.3 (HDF5) files."""

    with h5py.File(file_path, "r") as f:
        vars = []
        for var in f:
            if var in (MAT_HDF_REFS_GROUP, MAT_HDF_SUBSYS_GROUP):
                continue

            class_name = f[var].attrs.get(MAT_HDF_ATTRS.CLASS, None)
            object_decode = f[var].attrs.get(MAT_HDF_ATTRS.OBJECT_DECODE, -1)
            is_empty = f[var].attrs.get(MAT_HDF_ATTRS.EMPTY, 0)
            is_sparse = f[var].attrs.get(MAT_HDF_ATTRS.SPARSE, -1)

            if class_name is not None:
                class_name = class_name.decode("ascii")

            if is_empty:
                shape = tuple(int(x) for x in f[var][()])
            elif is_sparse >= 0:
                ncols = f[var]["jc"].size - 1
                shape = (int(is_sparse), ncols)
                class_name = "sparse"
            elif object_decode > 0:
                if object_decode == ObjectDecodingHint.OPAQUE_HINT:
                    if isinstance(f[var], h5py.Dataset):
                        objmetadata = f[var][()].T
                        shape = shape_from_metadata(objmetadata)
                    else:
                        if MAT_HDF_ATTRS.FIELDS in f[var].attrs:
                            # Enumeration Array
                            shape = f[var]["ValueIndices"].shape[::-1]
                        else:
                            warnings.warn(
                                f"Cannot determine shape of opaque object {var}",
                                MatReadWarning,
                            )
                            shape = ()
                else:
                    shape = (1, 1)
            elif class_name == "struct":
                keys = list(f[var].keys())
                if len(keys) == 0:
                    # Struct with no fields
                    shape = (1, 1)
                f[var][keys[0]]
                if isinstance(f[var][keys[0]], h5py.Dataset) and f[var][
                    keys[0]
                ].dtype == np.dtype("O"):
                    # Struct array
                    shape = f[var][keys[0]].shape[::-1]
                else:
                    # Scalar struct
                    shape = (1, 1)
            else:
                shape = f[var].shape[::-1]

            vars.append((var, shape, class_name))

        return vars


class MatRead7:
    """Reads MAT-file version 7.3 (HDF5) files."""

    def __init__(self, file_stream, subsystem=None):
        """Initializes the MAT-file reader.
        Parameters
        ----------
            file_stream : h5py.File object
            raw_data : bool, optional
            add_table_attrs : bool, optional
            chars_as_strings : bool, optional
        """
        self.h5stream = file_stream
        self.subsystem = subsystem

    def read_numeric(self, obj, is_empty=0):
        """Reads MATLAB numeric arrays from the v7.3 MAT-file."""

        int_decode = obj.attrs.get(MAT_HDF_ATTRS.INT_DECODE, None)

        if is_empty:
            obj_class = obj.attrs.get(MAT_HDF_ATTRS.CLASS, None)
            dtype = matlab_class_to_dtype(obj_class, obj.dtype)
            return np.empty(shape=obj[()], dtype=dtype)

        arr = obj[()]
        if arr.dtype.names:
            # complex number
            real = arr["real"]
            imag = arr["imag"]
            arr = np.empty(shape=arr.shape, dtype=np.complex128)
            arr.real = real
            arr.imag = imag

        if int_decode == IntegerDecodingHint.LOGICAL_HINT:
            arr = arr.astype(np.bool_)

        return arr.T

    def read_char(self, obj, is_empty=0):
        """Decodes MATLAB char arrays from the v7.3 MAT-file."""

        decode_type = obj.attrs.get(MAT_HDF_ATTRS.INT_DECODE, None)
        raw = obj[()].T

        if is_empty:
            return chars_to_strings(np.empty(shape=raw, dtype=np.str_))

        if decode_type == IntegerDecodingHint.UTF16_HINT:
            codec = "utf-16"
        else:
            warnings.warn(
                f"MatRead7:read_char:{MAT_HDF_ATTRS.INT_DECODE}={decode_type} not supported. "
                "This may lead to unexpected behaviour",
                MatReadWarning,
            )
            codec = "utf-8"

        decoded_arr = np.array(list(raw.tobytes().decode(codec))).reshape(raw.shape)
        return chars_to_strings(decoded_arr)

    def is_struct_matrix(self, hdf5_group):
        """Check if the HDF5 struct group is a struct matrix or scalar"""

        # Scalar structs are stored directly as members of a group (can be nested)
        # Struct arrays are stored as datasets of HDF5 references
        for key in hdf5_group:
            obj = hdf5_group[key]
            if isinstance(obj, h5py.Group):
                return False

            if isinstance(obj, h5py.Dataset):
                class_name = obj.attrs.get(MAT_HDF_ATTRS.CLASS, None)
                if class_name is not None:
                    return False

        return True

    def read_struct(self, obj, is_empty=0):
        """Reads MATLAB struct arrays from the v7.3 MAT-file."""

        if is_empty:
            return EmptyMatStruct(np.empty(shape=obj[()], dtype=object))

        fields = list(obj.keys())
        field_order = obj.attrs.get(MAT_HDF_ATTRS.FIELDS, None)
        if field_order is not None:
            if isinstance(field_order, h5py.Reference):
                # MATLAB writes large field lists as a reference to dset
                fields = ["".join(x.astype(str)) for x in self.h5stream[field_order]]
            else:
                fields = ["".join(x.astype(str)) for x in field_order]

        if self.is_struct_matrix(obj):
            is_scalar = False
            shape = next(iter(obj.values())).shape
        else:
            is_scalar = True
            shape = (1, 1)

        dt = [(name, object) for name in fields]
        arr = np.empty(shape=shape, dtype=dt)

        for field in obj:
            obj_field = obj[field]
            for idx in np.ndindex(arr.shape):
                if is_scalar:
                    arr[idx][field] = self.read_h5_data(obj_field)
                else:
                    arr[idx][field] = self.read_h5_data(self.h5stream[obj_field[idx]])
        return arr.T

    def read_cell(self, obj, is_empty=0):
        """Reads MATLAB cell arrays from the v7.3 MAT-file."""

        if is_empty:
            return np.empty(shape=obj[()], dtype=object)

        arr = np.empty(shape=obj.shape, dtype=object)
        for idx in np.ndindex(obj.shape):
            ref_data = self.h5stream[obj[idx]]
            arr[idx] = self.read_h5_data(ref_data)
        return arr.T

    def read_sparse(self, obj, nrows):
        """Reads MATLAB sparse arrays from the v7.3 MAT-file."""

        jc = obj["jc"][()]
        ncols = jc.size - 1

        if "data" in obj:
            # Exists only if sparse matrix contains non-zero elements
            data = self.read_numeric(obj["data"])
            matlab_class = obj.attrs.get(MAT_HDF_ATTRS.CLASS, None)
            if data.dtype.kind != "c":
                dtype = matlab_class_to_dtype(matlab_class, data.dtype)
                data = data.astype(dtype)
            ir = obj["ir"][()]
        else:
            data = np.array([], dtype=np.float64)
            ir = np.array([], dtype=np.int32)

        return csc_array((data, ir, jc), shape=(nrows, ncols))

    def read_function_handle(self, obj, object_decode):
        """Reads MATLAB function handles from the v7.3 MAT-file."""

        if object_decode != ObjectDecodingHint.FUNCTION_HINT:
            raise ValueError(
                f"Function handle with object decode {object_decode} not supported. Data may be corrupted"
            )

        data = self.read_struct(obj)
        return MatlabFunction(data)

    def read_object(self, obj, classname):
        """Reads mxOBJECT_CLASS variables from the v7.3 MAT-file."""

        fields = self.read_struct(obj)
        return MatlabObject(fields, classname)

    def read_opaque(self, obj, object_decode):
        """Reads MATLAB opaque objects from the v7.3 MAT-file."""

        classname = obj.attrs.get(MAT_HDF_ATTRS.CLASS, None)
        if classname is not None:
            classname = classname.decode("ascii")

        if object_decode == ObjectDecodingHint.OBJECT_HINT:
            return self.read_object(obj, classname)

        if object_decode != ObjectDecodingHint.OPAQUE_HINT:
            raise ValueError(
                f"Opaque object with object decode {object_decode} not supported. Data may be corrupted"
            )

        if classname == MCOS_SUBSYSTEM_CLASS:
            return self.read_cell(obj)

        type_system = guess_type_system(classname)

        # Check Enumeration Instances
        fields = obj.attrs.get(MAT_HDF_ATTRS.FIELDS, None)
        if fields is not None:
            fields = ["".join(x.astype(str)) for x in fields]

            if ENUMERATION_INSTANCE_TAG in fields:
                metadata = self.read_struct(obj)
                if metadata[0, 0][ENUMERATION_INSTANCE_TAG] != MCOS_MAGIC_NUMBER:
                    return metadata
                return self.subsystem.load_opaque_object(metadata, type_system)
            else:
                metadata = obj[()].T
        else:
            metadata = obj[()].T
        return self.subsystem.load_opaque_object(metadata, type_system, classname)

    def read_h5_data(self, obj):
        """Reads data from the HDF5 object."""

        matlab_class = obj.attrs.get(MAT_HDF_ATTRS.CLASS, None)
        is_empty = obj.attrs.get(MAT_HDF_ATTRS.EMPTY, 0)
        object_decode = obj.attrs.get(MAT_HDF_ATTRS.OBJECT_DECODE, -1)
        matlab_sparse = obj.attrs.get(MAT_HDF_ATTRS.SPARSE, -1)

        if matlab_sparse >= 0:
            arr = self.read_sparse(obj, matlab_sparse)
        elif matlab_class == b"char":
            arr = self.read_char(obj, is_empty)
        elif matlab_class in (
            b"int8",
            b"uint8",
            b"int16",
            b"uint16",
            b"int32",
            b"uint32",
            b"int64",
            b"uint64",
            b"single",
            b"double",
            b"logical",
        ):
            arr = self.read_numeric(obj, is_empty)
        elif matlab_class == b"struct":
            arr = self.read_struct(obj, is_empty)
        elif matlab_class == b"cell":
            arr = self.read_cell(obj, is_empty)
        elif matlab_class == b"function_handle":
            arr = self.read_function_handle(obj, object_decode)
        elif matlab_class == b"canonical empty" and is_empty:
            arr = MatlabCanonicalEmpty()
        elif object_decode >= 0:
            arr = self.read_opaque(obj, object_decode)
        else:
            raise NotImplementedError(f"MATLAB class {matlab_class} not supported")

        return arr

    def get_variables(self, variable_names):
        """Reads variables from the HDF5 file."""

        mdict = {}
        mdict["__globals__"] = []

        for var in self.h5stream:
            obj = self.h5stream[var]
            if var in (MAT_HDF_REFS_GROUP, MAT_HDF_SUBSYS_GROUP):
                continue
            if variable_names is not None and var not in variable_names:
                continue
            try:
                data = self.read_h5_data(obj)
            except Exception as err:
                raise ValueError(f"Error reading variable {var}: {err}") from err
            mdict[var] = data
            is_global = obj.attrs.get(MAT_HDF_ATTRS.GLOBAL, 0)
            if is_global:
                mdict["__globals__"].append(var)
            if variable_names is not None:
                variable_names.remove(var)
                if len(variable_names) == 0:
                    break

        return mdict


def read_subsystem(f, byte_order, raw_data, add_table_attrs):
    """Read subsystem data if present"""

    if MAT_HDF_SUBSYS_GROUP not in f:
        return None

    subsys_group = f[MAT_HDF_SUBSYS_GROUP]
    MR7 = MatRead7(f, None)
    subsys_data = MR7.read_struct(subsys_group)
    subsystem = MatSubsystem(byte_order, raw_data, add_table_attrs)
    subsystem.load_subsystem(subsys_data)

    return subsystem
