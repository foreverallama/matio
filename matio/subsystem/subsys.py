"""Methods to handle MATLAB FileWrapper__ data"""

import warnings
from enum import IntEnum, StrEnum

import numpy as np

from matio.utils.matclass import (
    EmptyMatStruct,
    MatlabOpaque,
    MatlabOpaqueArray,
    MatReadError,
    MatReadWarning,
)

from ..utils.matclass import OpaqueType, PropertyType
from ..utils.matconvert import convert_mat_to_py, mat_to_enum, matlab_classdef_types
from ..utils.matheaders import MCOS_MAGIC_NUMBER

SYSTEM_BYTE_ORDER = "<" if np.little_endian else ">"

FILEWRAPPER_VERSION = 4


class MatSubsystem:
    """Representation class for MATLAB FileWrapper__ data"""

    def __init__(self, byte_order, raw_data, add_table_attrs):
        self.byte_order = "<u4" if byte_order[0] == "<" else ">u4"
        self.raw_data = raw_data
        self.add_table_attrs = add_table_attrs

        self.mcos_object_cache = {}

        self.version = None
        self.num_names = 0
        self.mcos_names = []

        # Metadata Regions
        self.class_id_metadata = []
        self.object_id_metadata = []
        self.saveobj_metadata = []
        self.nobj_metadata = []
        self.dynprop_metadata = []
        self._u6_metadata = None  # Unknown Object Metadata
        self._u7_metadata = None  # Unknown Object Metadata

        self.mcos_props_saved = []

        # Class Template Data
        self._c3 = None  # Unknown Class Template
        self._c2 = None  # Unknown Class Template
        self.mcos_props_defaults = None

        self._handle_data = None
        self._java_data = None

        # Counters for object serialization
        self.saveobj_counter = 0
        self.nobj_counter = 0
        self.class_id_counter = 0
        self.object_id_counter = 0

    def load_subsystem(self, subsystem_data):
        """Parse and cache subsystem data"""

        for field in subsystem_data.dtype.names:
            if field == OpaqueType.JAVA:
                self._java_data = subsystem_data[0, 0][field]
            if field == OpaqueType.HANDLE:
                self._handle_data = subsystem_data[0, 0][field]
            if field == OpaqueType.MCOS:
                self.load_mcos_data(subsystem_data[0, 0][field])

    def load_fwrap_metadata(self, fwrap_metadata):
        """Parse and cache FileWrapper__ metadata"""

        self.version = np.frombuffer(
            fwrap_metadata, dtype=self.byte_order, count=1, offset=0
        )[0]
        if not 1 < self.version <= FILEWRAPPER_VERSION:
            raise MatReadError(f"FileWrapper version {self.version} is not supported")

        # Number of unique property and class names
        self.num_names = np.frombuffer(
            fwrap_metadata, dtype=self.byte_order, count=1, offset=4
        )[0]

        # 8 offsets to different regions within this cell
        region_offsets = np.frombuffer(
            fwrap_metadata, dtype=self.byte_order, count=8, offset=8
        )

        # A list of null terminated Property and Class Names
        data = fwrap_metadata[40 : region_offsets[0]].tobytes()
        raw_strings = data.split(b"\x00")
        self.mcos_names = [s.decode("ascii") for s in raw_strings if s]

        # Region 1: Class ID Metadata
        self.class_id_metadata = np.frombuffer(
            fwrap_metadata,
            dtype=self.byte_order,
            count=(region_offsets[1] - region_offsets[0]) // 4,
            offset=region_offsets[0],
        )

        # Region 2: Saveobj Prop Metadata
        self.saveobj_metadata = np.frombuffer(
            fwrap_metadata,
            dtype=self.byte_order,
            count=(region_offsets[2] - region_offsets[1]) // 4,
            offset=region_offsets[1],
        )

        # Region 3: Object ID Metadata
        self.object_id_metadata = np.frombuffer(
            fwrap_metadata,
            dtype=self.byte_order,
            count=(region_offsets[3] - region_offsets[2]) // 4,
            offset=region_offsets[2],
        )

        # Region 4: Object Prop Metadata
        self.nobj_metadata = np.frombuffer(
            fwrap_metadata,
            dtype=self.byte_order,
            count=(region_offsets[4] - region_offsets[3]) // 4,
            offset=region_offsets[3],
        )

        # Region 5: Dynamic Prop Metadata
        self.dynprop_metadata = np.frombuffer(
            fwrap_metadata,
            dtype=self.byte_order,
            count=(region_offsets[5] - region_offsets[4]) // 4,
            offset=region_offsets[4],
        )

        # Following may be reserved in some versions
        # Unknown data, kept raw
        if region_offsets[6] > 0:
            self._u6_metadata = fwrap_metadata[region_offsets[5] : region_offsets[6]]
        if region_offsets[7] > 0:
            self._u7_metadata = fwrap_metadata[region_offsets[6] : region_offsets[7]]

    def load_mcos_data(self, fwrap_data):
        """Parse and cache MCOS FileWrapper__ data"""

        fwrap_metadata = fwrap_data[0, 0]
        self.load_fwrap_metadata(fwrap_metadata)

        if self.version == 2:
            self.mcos_props_saved = fwrap_data[2:-1, 0]
        elif self.version == 3:
            self.mcos_props_saved = fwrap_data[2:-2, 0]
            self._c2 = fwrap_data[-2, 0]
        else:
            self.mcos_props_saved = fwrap_data[2:-3, 0]
            self._c3 = fwrap_data[-3, 0]
            self._c2 = fwrap_data[-2, 0]

        self.mcos_props_defaults = fwrap_data[-1, 0]

    def init_save(self):
        """Initializes save with metadata for object ID = 0"""

        self.class_id_metadata.extend([0, 0, 0, 0])
        self.object_id_metadata.extend([0, 0, 0, 0, 0, 0])
        self.saveobj_prop_metadata.extend([0, 0])
        self.obj_prop_metadata.extend([0, 0])
        self.dynprop_metadata.extend([0, 0])

    def is_valid_mcos_enumeration(self, metadata):
        """Checks if property value is a valid MCOS enumeration metadata array"""

        if metadata.dtype.names:
            if "EnumerationInstanceTag" in metadata.dtype.names:
                if (
                    metadata[0, 0]["EnumerationInstanceTag"].dtype == np.uint32
                    and metadata[0, 0]["EnumerationInstanceTag"].size == 1
                    and metadata[0, 0]["EnumerationInstanceTag"] == MCOS_MAGIC_NUMBER
                ):
                    return True

        return False

    def is_valid_mcos_object(self, metadata):
        """Checks if property value is a valid MCOS metadata array"""

        if not (
            metadata.dtype == np.uint32
            and metadata.ndim == 2
            and metadata.shape[1] == 1
            and metadata.size >= 3
        ):
            return False

        if metadata[0, 0] != MCOS_MAGIC_NUMBER:
            return False

        return True

    def is_valid_opaque_object(self, metadata):
        """Checks if property value is a valid opaque object metadata array"""

        # TODO: Add checks for other opaque objects
        return self.is_valid_mcos_object(metadata)

    def check_prop_for_opaque(self, prop):
        """Check if a property value in FileWrapper__ contains opaque objects during load"""

        if not isinstance(prop, np.ndarray):
            return prop

        if prop.dtype.hasobject:
            if prop.dtype.names:
                # Iterate though struct array
                # Also handles MatlabObject, MatlabFunction
                if self.is_valid_mcos_enumeration(prop):
                    return self.load_mcos_enumeration(prop, type_system=OpaqueType.MCOS)
                else:
                    for idx in np.ndindex(prop.shape):
                        for name in prop.dtype.names:
                            field_val = prop[idx][name]
                            prop[idx][name] = self.check_prop_for_opaque(field_val)

            else:
                # Iterate through cell arrays
                # NOTE: Function Handles to classdef methods have an MCOS identifier
                # But I don't think there's anything to read them as opaque objects
                for idx in np.ndindex(prop.shape):
                    cell_item = prop[idx]
                    prop[idx] = self.check_prop_for_opaque(cell_item)

        elif self.is_valid_opaque_object(prop):
            # MCOS class names are derived from subsystem
            # For other types, it may be derived from metadata instead
            # So we use a placeholder classname
            prop = self.load_opaque_object(prop, type_system=OpaqueType.MCOS)

        return prop

    def get_classname(self, class_id):
        """Extracts class name with namespace qualifier for a given object from its class ID."""

        namespace_idx = self.class_id_metadata[class_id * 4]
        classname_idx = self.class_id_metadata[class_id * 4 + 1]
        # Remaining two fields are unknowns

        if namespace_idx == 0:
            namespace = ""
        else:
            namespace = self.mcos_names[namespace_idx - 1] + "."

        classname = namespace + self.mcos_names[classname_idx - 1]
        return classname

    def get_object_metadata(self, object_id):
        """Extracts object dependency IDs for a given object."""

        class_id, _, _, saveobj_id, normobj_id, dep_id = self.object_id_metadata[
            object_id * 6 : object_id * 6 + 6
        ]
        # Ignored fields are unknowns
        return class_id, saveobj_id, normobj_id, dep_id

    def get_default_properties(self, class_id):
        """Returns the default properties (as dict) for a given class ID"""

        prop_arr = self.mcos_props_defaults[class_id, 0]
        prop_map = {}
        if prop_arr.dtype.names:
            for prop_name in prop_arr.dtype.names:
                prop_map[prop_name] = self.check_prop_for_opaque(
                    prop_arr[prop_name][0, 0]
                )

        return prop_map

    def get_property_idxs(self, obj_type_id, saveobj_ret_type):
        """Returns the property (name, type, value) metadata for an object ID"""

        prop_field_idxs = (
            self.saveobj_metadata if saveobj_ret_type else self.nobj_metadata
        )

        nfields = 3
        offset = prop_field_idxs[0]
        for _ in range(obj_type_id):
            nprops = prop_field_idxs[offset]
            offset += 1 + nfields * nprops
            offset += offset % 2  # Padding

        nprops = prop_field_idxs[offset]
        offset += 1
        return prop_field_idxs[offset : offset + nprops * nfields].reshape(
            (nprops, nfields)
        )

    def get_saved_properties(self, obj_type_id, saveobj_ret_type):
        """Returns the saved properties (as dict) for an object ID"""

        save_prop_map = {}

        prop_field_idxs = self.get_property_idxs(obj_type_id, saveobj_ret_type)
        for prop_idx, prop_type, prop_value in prop_field_idxs:
            prop_name = self.mcos_names[prop_idx - 1]
            if prop_type == PropertyType.MATLAB_ENUMERATION:
                save_prop_map[prop_name] = self.mcos_names[prop_value - 1]
            elif prop_type == PropertyType.PROPERTY_VALUE:
                save_prop_map[prop_name] = self.check_prop_for_opaque(
                    self.mcos_props_saved[prop_value]
                )
            elif prop_type == PropertyType.INTEGER_VALUE:
                save_prop_map[prop_name] = prop_value
            else:
                warnings.warn(
                    f'Unknown property type {prop_type} for property "{prop_name}"',
                    Warning,
                    stacklevel=3,
                )
                save_prop_map[prop_name] = prop_value

        return save_prop_map

    def get_dyn_object_id(self, normobj_id):
        """Gets the object ID from normobj ID for dynamicprops objects"""

        num_objects = len(self.object_id_metadata) // 6

        for object_id in range(num_objects):
            block_start = object_id * 6
            block_nobj_id = self.object_id_metadata[block_start + 4]

            if block_nobj_id == normobj_id:
                return object_id

        return None

    def get_dynamic_properties(self, dep_id):
        """Returns dynamicproperties (as dict) for a given object based on dependency ID"""

        offset = self.dynprop_metadata[0]
        for i in range(dep_id):
            nprops = self.dynprop_metadata[offset]
            offset += 1 + nprops
            offset += offset % 2  # Padding

        ndynprops = self.dynprop_metadata[offset]
        offset += 1
        dyn_prop_type2_ids = self.dynprop_metadata[offset : offset + ndynprops]

        if ndynprops == 0:
            return {}

        dyn_prop_map = {}
        for i, dyn_prop_id in enumerate(dyn_prop_type2_ids):
            dyn_obj_id = self.get_dyn_object_id(dyn_prop_id)
            dyn_class_id = self.get_object_metadata(dyn_obj_id)[0]
            classname = self.get_classname(dyn_class_id)
            dynobj = MatlabOpaque(
                None, type_system=OpaqueType.MCOS, classname=classname
            )
            dynobj.properties = self.get_properties(dyn_obj_id)
            dyn_prop_map[f"__dynamic_property__{i + 1}"] = dynobj

        return dyn_prop_map

    def get_properties(self, object_id):
        """Returns the properties as a dict for a given object ID"""

        if object_id == 0:
            # Matlab uses an object ID=0
            # MATLAB seems to keep references to deleted objects
            # Observed this in fig files I think? Don't remember
            # objectID=0 may be a placeholder for such cases
            return None

        class_id, saveobj_id, normobj_id, dep_id = self.get_object_metadata(object_id)
        if saveobj_id != 0:
            saveobj_ret_type = True
            obj_type_id = saveobj_id
        else:
            saveobj_ret_type = False
            obj_type_id = normobj_id

        prop_map = self.get_default_properties(class_id)
        prop_map.update(self.get_saved_properties(obj_type_id, saveobj_ret_type))
        prop_map.update(self.get_dynamic_properties(dep_id))

        return prop_map

    def load_mcos_enumeration(self, metadata, type_system):
        """Loads MATLAB MCOS enumeration instance array"""

        classname = self.get_classname(metadata[0, 0]["ClassName"].item())
        builtin_class_idx = metadata[0, 0]["BuiltinClassName"].item()
        if builtin_class_idx != 0:
            builtin_class_name = self.get_classname(builtin_class_idx)
        else:
            builtin_class_name = np.str_("")

        value_names = [
            self.mcos_names[val - 1] for val in metadata[0, 0]["ValueNames"].ravel()
        ]

        enum_vals = []
        value_idx = metadata[0, 0]["ValueIndices"]
        mmdata = metadata[0, 0]["Values"]  # Array is N x 1 shape
        if mmdata.size != 0:
            mmdata_map = mmdata[value_idx]
            for val in np.nditer(mmdata_map, flags=["refs_ok"], op_flags=["readonly"]):
                obj_array = self.load_mcos_object(val.item(), "MCOS")
                enum_vals.append(obj_array)

        if not self.raw_data:
            return mat_to_enum(
                enum_vals,
                value_names,
                classname,
                value_idx.shape,
            )

        metadata[0, 0]["BuiltinClassName"] = builtin_class_name
        metadata[0, 0]["ClassName"] = classname
        metadata[0, 0]["ValueNames"] = np.array(value_names).reshape(
            value_idx.shape, order="F"
        )
        metadata[0, 0]["ValueIndices"] = value_idx
        metadata[0, 0]["Values"] = np.array(enum_vals).reshape(
            value_idx.shape, order="F"
        )

        return MatlabOpaque(metadata, type_system, classname)

    def load_mcos_object(self, metadata, type_system=OpaqueType.MCOS):
        """Loads MCOS object"""

        metadata = np.atleast_2d(metadata)

        ndims = metadata[1, 0]
        dims = metadata[2 : 2 + ndims, 0]
        nobjects = np.prod(dims)
        object_ids = metadata[2 + ndims : 2 + ndims + nobjects, 0]

        class_id = metadata[-1, 0]
        classname = self.get_classname(class_id)

        is_array = nobjects > 1
        array_objs = []
        for object_id in object_ids:
            if object_id in self.mcos_object_cache:
                obj = self.mcos_object_cache[object_id]
            else:
                if not self.raw_data and classname in matlab_classdef_types:
                    obj_props = self.get_properties(object_id)
                    obj = convert_mat_to_py(
                        obj_props,
                        classname,
                        byte_order=self.byte_order,
                        add_table_attrs=self.add_table_attrs,
                    )
                    self.mcos_object_cache[object_id] = (
                        obj  # Caching here is probably unnecessary but safer
                    )
                else:
                    obj = MatlabOpaque(None, type_system, classname)
                    self.mcos_object_cache[object_id] = obj
                    obj.properties = self.get_properties(object_id)
            array_objs.append(obj)

        if is_array:
            obj_arr = np.empty((nobjects,), dtype=object)
            obj_arr[:] = array_objs
            obj_arr = obj_arr.reshape(dims, order="F")
            obj_arr = MatlabOpaqueArray(obj_arr, type_system, classname)
        else:
            obj_arr = array_objs[0]

        return obj_arr

    def load_opaque_object(self, metadata, type_system, classname=None):
        """Loads opaque object"""

        if type_system != OpaqueType.MCOS:
            warnings.warn(
                f"Opaque object of type {type_system} is not supported",
                MatReadWarning,
                stacklevel=2,
            )
            return MatlabOpaque(metadata, type_system, classname)

        if metadata.dtype.names:
            return self.load_mcos_enumeration(metadata, type_system)
        else:
            return self.load_mcos_object(metadata, type_system)
