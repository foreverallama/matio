"""MAT-file version 7.3 (HDF5) writer."""

import string
import sys
import warnings
from itertools import count, product

import h5py
import numpy as np
from scipy.sparse import issparse

from matio.subsystem import MatSubsystem
from matio.utils.matclass import (
    MCOS_SUBSYSTEM_CLASS,
    EmptyMatStruct,
    IntegerDecodingHint,
    MatlabCanonicalEmpty,
    MatlabClasses,
    MatlabEnumerationArray,
    MatlabFunction,
    MatlabObject,
    MatlabOpaque,
    MatlabOpaqueArray,
    MatWriteWarning,
    ObjectDecodingHint,
)
from matio.utils.matheaders import (
    MAT_HDF_ATTRS,
    MAT_HDF_COMPRESSION,
    MAT_HDF_COMPRESSION_OPTS,
    MAT_HDF_REFS_GROUP,
    MAT_HDF_SUBSYS_GROUP,
    MAT_HDF_USER_BLOCK_BYTES,
    MAT_HDF_VERSION,
    write_file_header,
)
from matio.utils.matutils import mat_numeric, strings_to_chars, to_writeable

SYS_BYTE_ORDER = "<" if sys.byteorder == "little" else ">"


def savemat7(file_path, mdict, global_vars, saveobj_classes, oned_as):
    """Write data to MAT-5file."""

    with h5py.File(file_path, "w", userblock_size=MAT_HDF_USER_BLOCK_BYTES) as f:
        MW7 = MatWrite7(f, oned_as=oned_as)
        MW7.subsystem = MatSubsystem(
            byte_order=SYS_BYTE_ORDER, oned_as=oned_as, saveobj_classes=saveobj_classes
        )
        MW7.subsystem.init_save()
        MW7.put_variables(mdict, global_vars)

        subsystem = MW7.subsystem.set_subsystem()
        if subsystem is not None:
            MW7.write_struct_array(f, MAT_HDF_SUBSYS_GROUP, subsystem)

    with open(file_path, "r+b") as f:
        f.seek(0)
        write_file_header(f, version=MAT_HDF_VERSION)

    return


class MatWrite7:
    """Writes MAT-file version 7.3 (HDF5) files."""

    def __init__(self, h5file, oned_as="col"):
        self.h5file = h5file
        self.oned_as = oned_as
        self.subsystem = None
        self.refs_group = MAT_HDF_REFS_GROUP
        self._name_gen = self._matlab_refname_generator()

    def _matlab_refname_generator(self):
        """Generates MATLAB style refs subgroup names."""
        yield from string.ascii_lowercase
        yield from string.ascii_uppercase
        alphabet = string.ascii_lowercase + string.ascii_uppercase
        for size in count(2):
            for letters in product(alphabet, repeat=size):
                yield "".join(letters)

    def _next_refname(self):
        """Gets the next MATLAB style refs subgroup name."""
        return next(self._name_gen)

    def get_empty_array(self, shape):
        """MATLAB stores empty arrays as uint64 containing shape"""
        return np.array(shape, dtype=np.uint64)

    def add_empty_attribute(self, dset):
        """Adds MATLAB empty attribute to the dataset."""
        dset.attrs.create(MAT_HDF_ATTRS.EMPTY, np.int32(1))

    def add_classname_attr(self, dset, classname):
        """Adds MATLAB class name attribute to the dataset."""
        dset.attrs.create(MAT_HDF_ATTRS.CLASS, np.bytes_(classname))

    def add_int_decode_attr(self, dset, int_decode):
        """Adds MATLAB integer decoding hint attribute to the dataset."""
        dset.attrs.create(MAT_HDF_ATTRS.INT_DECODE, np.int32(int_decode))

    def add_object_decode_attr(self, dset, object_decode):
        """Adds MATLAB object decoding hint attribute to the dataset."""
        dset.attrs.create(MAT_HDF_ATTRS.OBJECT_DECODE, np.int32(object_decode))

    def add_sparse_attr(self, dset, nrows):
        """Adds MATLAB sparse nnz attribute to the dataset."""
        dset.attrs.create(MAT_HDF_ATTRS.SPARSE, np.int32(nrows))

    def write_numeric_dset(self, parent, var_name, data):
        """Writes a numeric dataset to the HDF5 file."""

        data, classname, int_decode = mat_numeric(data, version=MAT_HDF_VERSION)
        if data.size == 0:
            data_empty = self.get_empty_array(data.shape)
            dset = parent.create_dataset(var_name, data=data_empty)
            self.add_empty_attribute(dset)
        else:
            dset = parent.create_dataset(
                var_name,
                data=data.T,
                compression=MAT_HDF_COMPRESSION,
                compression_opts=MAT_HDF_COMPRESSION_OPTS,
                chunks=True,
            )

        self.add_classname_attr(dset, classname)
        if int_decode is not None and int_decode == IntegerDecodingHint.LOGICAL_HINT:
            self.add_int_decode_attr(dset, int_decode)

        return dset

    def write_char_dset(self, parent, var_name, data):
        """Writes a char dataset to the HDF5 file."""

        if data.size == 0 or np.all(data == ""):
            # Empty String Array
            # MATLAB only recognizes 0-D char array
            shape = (0,) * np.max([data.ndim, 2])
            data_empty = self.get_empty_array(shape)
            dset = parent.create_dataset(var_name, data=data_empty)
            self.add_empty_attribute(dset)
        else:
            data = strings_to_chars(data)
            data = data.view(np.uint32).astype(np.uint16)
            dset = parent.create_dataset(
                var_name,
                data=data.T,
                compression=MAT_HDF_COMPRESSION,
                compression_opts=MAT_HDF_COMPRESSION_OPTS,
                chunks=True,
            )

        self.add_classname_attr(dset, MatlabClasses.CHAR)
        self.add_int_decode_attr(dset, IntegerDecodingHint.UTF16_HINT)

        return dset

    def write_cell_array(self, parent, var_name, data):
        """Writes a cell array to the HDF5 file."""

        if data.size == 0:
            data = self.get_empty_array(data.shape)
            dset = parent.create_dataset(var_name, data=data.T)
            self.add_classname_attr(dset, MatlabClasses.CELL)
            self.add_empty_attribute(dset)

        else:
            ref_array = np.empty(data.shape, dtype=h5py.ref_dtype)

            for idx, item in np.ndenumerate(data):
                dset_name = self._next_refname()
                dset = self.write_variable(dset_name, item, group=self.refs_group)
                ref_array[idx] = dset.ref

            dset = parent.create_dataset(
                var_name, data=ref_array.T, dtype=h5py.ref_dtype
            )

        self.add_classname_attr(dset, MatlabClasses.CELL)
        return dset

    def write_empty_struct(self, parent, var_name, data):
        """Writes an empty struct to the HDF5 file."""

        data = self.get_empty_array(data.shape)
        dset = parent.create_dataset(var_name, data=data)
        self.add_classname_attr(dset, MatlabClasses.STRUCT)
        self.add_empty_attribute(dset)
        return dset

    def write_struct_array(self, parent, var_name, data, object_decode=0):
        """Writes a struct array to the HDF5 file."""

        fields = data.dtype.names
        if data.size == 0:
            # Handle empty struct arrays with fields
            empty_data = self.get_empty_array(data.shape)
            struct_group = parent.create_dataset(var_name, data=empty_data)
            self.add_empty_attribute(struct_group)
        elif data.size == 1:
            struct_group = parent.create_group(var_name)
            if fields is not None:
                for field in fields:
                    self.write_variable(
                        field, data[field][0, 0], group=struct_group.name
                    )
        else:
            # Struct arrays are stored as dset of references to each struct
            struct_group = parent.create_group(var_name)
            for field in fields:
                field_refs = np.empty(data.shape, dtype=h5py.ref_dtype)
                for idx, item in np.ndenumerate(data):
                    dset_name = self._next_refname()
                    dset = self.write_variable(
                        dset_name, item[field], group=self.refs_group
                    )
                    field_refs[idx] = dset.ref
                dset = struct_group.create_dataset(
                    field, data=field_refs.T, dtype=h5py.ref_dtype
                )

        if object_decode == ObjectDecodingHint.FUNCTION_HINT:
            self.add_object_decode_attr(struct_group, object_decode)
            self.add_classname_attr(struct_group, MatlabClasses.FUNCTION)
        elif object_decode == ObjectDecodingHint.OBJECT_HINT:
            self.add_object_decode_attr(struct_group, object_decode)
            self.add_classname_attr(struct_group, data.classname)
        else:
            self.add_classname_attr(struct_group, MatlabClasses.STRUCT)

        # Add field names attribute
        # MATLAB uses an assertion to ensure sub-group/dataset names matches with
        # field names in attributes
        dt = h5py.special_dtype(vlen=np.dtype("S1"))
        matlab_fields = np.empty(shape=(len(fields),), dtype=dt)

        total_characters = 0
        for i, field_name in enumerate(fields):
            encoded_field = np.array(
                [c.encode("ascii") for c in field_name], dtype="S1"
            )
            matlab_fields[i] = encoded_field
            total_characters += len(field_name)

        if matlab_fields.size > 0:
            # If total length of fields >= 4096 then its written as a reference
            if total_characters < 4096:
                struct_group.attrs.create(MAT_HDF_ATTRS.FIELDS, matlab_fields)
            else:
                # Write as a dset under #refs# group
                fields_refname = self._next_refname()
                parent = self.h5file.require_group(self.refs_group)
                parent.create_dataset(fields_refname, data=matlab_fields, dtype=dt)

        return struct_group

    def write_function_handle(self, parent, var_name, data):
        """Writes a function handle to the HDF5 file."""

        if data.size == 0:
            warnings.warn(
                "Empty function handle not supported. Skipping.", MatWriteWarning
            )
            return

        dset = self.write_struct_array(
            parent, var_name, data, object_decode=ObjectDecodingHint.FUNCTION_HINT
        )
        return dset

    def write_matlab_object(self, parent, var_name, data):
        """Writes a MATLAB object to the HDF5 file."""

        if data.size == 0:
            warnings.warn(
                "Empty MATLAB object not supported. Skipping.", MatWriteWarning
            )
            return

        dset = self.write_struct_array(
            parent, var_name, data, object_decode=ObjectDecodingHint.OBJECT_HINT
        )
        return dset

    def write_sparse_array(self, parent, var_name, data):
        """Writes a sparse array to the HDF5 file."""

        A = data.tocsc()
        A.sort_indices()
        ir = A.indices.astype("uint64")
        jc = A.indptr.astype("uint64")

        data, classname, int_decode = mat_numeric(A.data, version=MAT_HDF_VERSION)

        sparse_group = parent.create_group(var_name)
        sparse_group.create_dataset(
            "jc",
            data=jc,
            compression=MAT_HDF_COMPRESSION,
            compression_opts=MAT_HDF_COMPRESSION_OPTS,
            chunks=True,
        )
        if data.size > 0:
            sparse_group.create_dataset(
                "data",
                data=data,
                compression=MAT_HDF_COMPRESSION,
                compression_opts=MAT_HDF_COMPRESSION_OPTS,
                chunks=True,
            )
            sparse_group.create_dataset(
                "ir",
                data=ir,
                compression=MAT_HDF_COMPRESSION,
                compression_opts=MAT_HDF_COMPRESSION_OPTS,
                chunks=True,
            )

        self.add_classname_attr(sparse_group, classname)
        self.add_sparse_attr(sparse_group, A.shape[0])
        if int_decode is not None and int_decode == IntegerDecodingHint.LOGICAL_HINT:
            self.add_int_decode_attr(sparse_group, int_decode)
        return sparse_group

    def write_opaque_object(self, parent, var_name, data):
        """Writes an opaque object to the HDF5 file."""

        if data.classname == MCOS_SUBSYSTEM_CLASS:
            dset = self.write_cell_array(parent, var_name, data.properties)
        else:
            if isinstance(data, MatlabEnumerationArray):
                metadata = self.subsystem.set_enumeration_metadata(data)
                dset = self.write_struct_array(
                    parent,
                    var_name,
                    metadata,
                    object_decode=ObjectDecodingHint.OPAQUE_HINT,
                )
            else:
                metadata = self.subsystem.set_object_metadata(data)
                dset = parent.create_dataset(var_name, data=metadata.T)

        classname = data.classname

        dset.attrs.create(MAT_HDF_ATTRS.CLASS, np.bytes_(classname))
        dset.attrs.create(
            MAT_HDF_ATTRS.OBJECT_DECODE, np.int32(ObjectDecodingHint.OPAQUE_HINT)
        )
        return dset

    def write_canonical_empty(self, parent, var_name, data):
        """Writes a canonical empty array to the HDF5 file."""

        data_empty = self.get_empty_array((0, 0))
        dset = parent.create_dataset(var_name, data=data_empty)
        self.add_classname_attr(dset, MatlabClasses.EMPTY)
        self.add_empty_attribute(dset)
        return dset

    def write_variable(self, var_name, data, group=None):
        """Writes a variable to the HDF5 file."""

        if group is None:
            parent = self.h5file
        else:
            parent = self.h5file.require_group(group)

        data = to_writeable(data, self.oned_as)

        if issparse(data):
            dset = self.write_sparse_array(parent, var_name, data)
        elif isinstance(data, MatlabCanonicalEmpty):
            dset = self.write_canonical_empty(parent, var_name, data)
        elif isinstance(
            data, (MatlabOpaque, MatlabOpaqueArray, MatlabEnumerationArray)
        ):
            dset = self.write_opaque_object(parent, var_name, data)
        elif isinstance(data, MatlabFunction):
            dset = self.write_function_handle(parent, var_name, data)
        elif isinstance(data, MatlabObject):
            dset = self.write_matlab_object(parent, var_name, data)
        elif isinstance(data, EmptyMatStruct):
            dset = self.write_empty_struct(parent, var_name, data)

        # At this point it must be a numpy array
        elif data.dtype.kind in ("i", "u", "f", "c", "b"):
            dset = self.write_numeric_dset(parent, var_name, data)
        elif data.dtype.kind in ("U", "S"):
            dset = self.write_char_dset(parent, var_name, data)
        elif data.dtype.hasobject:
            if data.dtype.names:
                dset = self.write_struct_array(parent, var_name, data)
            else:
                dset = self.write_cell_array(parent, var_name, data)
        else:
            warnings.warn(
                f"Data type {data.dtype} not supported for variable {var_name}. Skipping.",
                MatWriteWarning,
            )
            return None

        if group is not None:
            dset.attrs.create("H5PATH", np.bytes_(dset.name))

        return dset

    def put_variables(self, mdict, global_vars):
        """Writes variables to the HDF5 file."""

        for var_name, data in mdict.items():
            if var_name[0] == "_":
                msg = (
                    f"Starting field name with a " f"underscore ({var_name}) is ignored"
                )
                warnings.warn(msg, MatWriteWarning, stacklevel=2)
                continue

            dset = self.write_variable(var_name, data)
            if dset is not None and var_name in global_vars:
                dset.attrs.create(MAT_HDF_ATTRS.GLOBAL, np.int32(1))

        return
