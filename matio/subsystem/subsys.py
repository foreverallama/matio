"""Methods to handle MATLAB FileWrapper__ data"""

import warnings

import numpy as np

from matio.utils.matclass import (
    DYNAMIC_PROPERTY_PREFIX,
    ENUMERATION_INSTANCE_TAG,
    MCOS_SUBSYSTEM_CLASS,
    EmptyMatStruct,
    MatlabCanonicalEmpty,
    MatlabEnumerationArray,
    MatlabOpaque,
    MatlabOpaqueArray,
    MatReadError,
    MatReadWarning,
    MatWriteWarning,
    OpaqueType,
    PropertyType,
)
from matio.utils.matconvert import (
    convert_mat_to_py,
    mat_to_enum,
    matlab_classdef_types,
    matlab_saveobj_ret_types,
)
from matio.utils.matheaders import MAT_HDF_VERSION, MCOS_MAGIC_NUMBER
from matio.utils.matutils import to_writeable

SYSTEM_BYTE_ORDER = "<" if np.little_endian else ">"

FILEWRAPPER_VERSION = 4
MIN_FILEWRAPPER_VERSION = 2


class MatSubsystem:
    """Representation class for MATLAB FileWrapper__ data"""

    def __init__(
        self,
        byte_order,
        raw_data=False,
        add_table_attrs=False,
        oned_as="col",
        saveobj_ret_classes=[],  # TODO: Allow as user arg
    ):
        self.byte_order = "<u4" if byte_order[0] == "<" else ">u4"
        self.raw_data = raw_data
        self.add_table_attrs = add_table_attrs
        self.oned_as = oned_as

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

        if isinstance(saveobj_ret_classes, str):
            saveobj_ret_classes = [saveobj_ret_classes]
        self.saveobj_class_names = matlab_saveobj_ret_types + saveobj_ret_classes

    def init_save(self):
        """Initializes save with metadata for object ID = 0"""

        self.version = FILEWRAPPER_VERSION
        self.class_id_metadata.extend([0, 0, 0, 0])
        self.dynprop_metadata.extend([0, 0])

        # These metadata fields are written per object before unrolling
        # nested properties. Each list represents one object which is mutable
        # This does not matter for class ID
        self.object_id_metadata.append([0, 0, 0, 0, 0, 0])
        self.saveobj_metadata.append([0, 0])
        self.nobj_metadata.append([0, 0])

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
        if not MIN_FILEWRAPPER_VERSION <= self.version <= FILEWRAPPER_VERSION:
            raise MatReadError(
                f"{MCOS_SUBSYSTEM_CLASS} version {self.version} is not supported"
            )

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

    def is_valid_mcos_enumeration(self, metadata):
        """Checks if property value is a valid MCOS enumeration metadata array"""

        if metadata.dtype.names:
            if ENUMERATION_INSTANCE_TAG in metadata.dtype.names:
                if (
                    metadata[0, 0][ENUMERATION_INSTANCE_TAG].dtype == np.uint32
                    and metadata[0, 0][ENUMERATION_INSTANCE_TAG].size == 1
                    and metadata[0, 0][ENUMERATION_INSTANCE_TAG] == MCOS_MAGIC_NUMBER
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
                    MatReadWarning,
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
            dyn_prop_map[f"{DYNAMIC_PROPERTY_PREFIX}{i + 1}"] = dynobj

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

    def set_mcos_name(self, name):
        """Gets or creates index for a name in mcos_names"""
        try:
            return self.mcos_names.index(name) + 1  # 1-based
        except ValueError:
            # Name doesn't exist, add it
            self.mcos_names.append(name)
            self.num_names += 1
            return self.num_names

    def set_class_id(self, classname):
        """Sets the class ID for a given class name (including namespace)"""

        for class_id in range(1, self.class_id_counter + 1):
            existing_name = self.get_classname(class_id)
            if existing_name == classname:
                return class_id

        self.class_id_counter += 1

        # Add new class ID metadata
        namespace, _, cname = classname.rpartition(".")
        if namespace:
            namespace_idx = self.set_mcos_name(namespace)
        else:
            namespace_idx = 0
        cname_idx = self.set_mcos_name(cname)

        metadata = [namespace_idx, cname_idx, 0, 0]
        self.class_id_metadata.extend(metadata)

        return self.class_id_counter

    def serialize_nested_props(self, prop_value):
        """Recursively serializes nested properties of an MCOS object"""

        if isinstance(prop_value, MatlabEnumerationArray):
            raise NotImplementedError("MatlabEnumerationArray is not yet supported")

        elif (
            isinstance(prop_value, np.ndarray)
            and prop_value.dtype.hasobject
            and prop_value.dtype.kind != "T"
        ):
            # Iterate through cell arrays and struct arrays
            # Ignore StringDType(): Used for MATLAB strings which uses 'O' internally

            # Make a copy to avoid modifying the original
            # Should be cheap as we're only copying object arrays
            prop_value = prop_value.copy()

            if prop_value.dtype.names:
                # Struct array
                for idx in np.ndindex(prop_value.shape):
                    for name in prop_value.dtype.names:
                        field_val = prop_value[idx][name]
                        field_val = to_writeable(field_val)
                        prop_value[idx][name] = self.serialize_nested_props(field_val)
            else:
                # Cell array
                for idx in np.ndindex(prop_value.shape):
                    cell_item = prop_value[idx]
                    cell_item = to_writeable(cell_item)
                    prop_value[idx] = self.serialize_nested_props(cell_item)

        else:
            prop_value = to_writeable(prop_value)

        if isinstance(prop_value, (MatlabOpaque, MatlabOpaqueArray)):
            prop_value = self.set_object_metadata(prop_value)

        return prop_value

    def serialize_object_props(self, prop_map, obj_prop_metadata):
        """Serializes the properties of an MCOS object"""

        object_id = self.object_id_counter
        nprops = len(prop_map)
        obj_prop_metadata.extend([nprops])

        for prop_name, prop_value in prop_map.items():
            if prop_name.startswith(DYNAMIC_PROPERTY_PREFIX):
                warnings.warn(
                    f"Dynamic property '{prop_name}' cannot be serialized. Skipping",
                    MatWriteWarning,
                    stacklevel=3,
                )
                continue

            if prop_name[0] in "_0123456789":
                msg = f"Property names cannot start with '{prop_name[0]}'. Skipping '{prop_name}'"
                warnings.warn(msg, MatWriteWarning, stacklevel=3)
                continue

            field_name_idx = self.set_mcos_name(prop_name)
            prop_value = self.serialize_nested_props(prop_value)

            cell_number_idx = len(self.mcos_props_saved)
            self.mcos_props_saved.append(prop_value)
            obj_prop_metadata.extend(
                [field_name_idx, PropertyType.PROPERTY_VALUE, cell_number_idx]
            )

        if len(obj_prop_metadata) % 2 == 1:
            obj_prop_metadata.append(0)  # Padding

        # TODO: Add dynamicprop support
        dynprop_entry = [0, 0]
        self.dynprop_metadata.extend(dynprop_entry)

        ndeps = self.object_id_counter - object_id
        return ndeps

    def set_object_id(self, obj, saveobj_ret_type=False):
        """Sets the object ID for a given object key (id(obj) or obj_key)"""

        if obj.properties is None:
            # Deleted object
            class_id = self.set_class_id(obj.classname)
            obj_id = 0
            return obj_id, class_id

        if obj in self.mcos_object_cache:
            class_id = self.set_class_id(obj.classname)
            return self.mcos_object_cache[obj], class_id

        self.object_id_counter += 1
        mat_obj_id = self.object_id_counter
        self.mcos_object_cache[obj] = mat_obj_id

        obj_prop_metadata = []
        saveobj_id = 0
        nobj_id = 0
        if saveobj_ret_type:
            self.saveobj_counter += 1
            saveobj_id = self.saveobj_counter
            self.saveobj_metadata.append(obj_prop_metadata)
        else:
            self.nobj_counter += 1
            nobj_id = self.nobj_counter
            self.nobj_metadata.append(obj_prop_metadata)

        obj_id_metadata = [0] * 6
        self.object_id_metadata.append(obj_id_metadata)

        # * Can we optimize here?
        if saveobj_ret_type and not (
            len(obj.properties) == 1 and set(obj.properties.keys()) == {"any"}
        ):
            raise MatReadError(
                f"Object of class {obj.classname} marked with a saveobj return type must have a single property 'any' containing the return value of its saveobj method"
            )

        ndeps = self.serialize_object_props(obj.properties, obj_prop_metadata)

        class_id = self.set_class_id(obj.classname)
        obj_id_metadata[0] = class_id
        if saveobj_ret_type:
            obj_id_metadata[3] = saveobj_id
        else:
            obj_id_metadata[4] = nobj_id

        obj_id_metadata[5] = mat_obj_id + ndeps
        for i in range(1, ndeps + 1):
            obj_id = self.object_id_counter - ndeps + i
            self.object_id_metadata[obj_id][5] -= 1

        return mat_obj_id, class_id

    def set_mcos_object_metadata(self, obj):
        """Sets metadata for a single MCOS object"""

        arr_ids = []
        if isinstance(obj, MatlabOpaque):
            classname = obj.classname
        elif isinstance(obj, MatlabOpaqueArray):
            classname = obj.flat[0].classname
        saveobj_ret_type = classname in self.saveobj_class_names

        if isinstance(obj, MatlabOpaqueArray):
            for obj_elem in obj.ravel(order="F"):
                object_id, class_id = self.set_object_id(obj_elem, saveobj_ret_type)
                arr_ids.append(object_id)
            dims = obj.shape
        else:
            object_id, class_id = self.set_object_id(obj, saveobj_ret_type)
            arr_ids.append(object_id)
            dims = (1, 1)

        return self.create_mcos_metadata(dims, arr_ids, class_id)

    def create_mcos_metadata(self, dims, arr_ids, class_id):
        """Creates MCOS metadata array"""

        ndims = len(dims)
        values = [MCOS_MAGIC_NUMBER, ndims] + list(dims) + list(arr_ids) + [class_id]
        return np.array(values, dtype=np.uint32).reshape(-1, 1)

    def set_object_metadata(self, obj):
        """Sets metadata for a MatioOpaque object"""

        if isinstance(obj, MatlabOpaqueArray):
            obj0 = obj.flat[0]
            type_system = obj0.type_system
        else:
            type_system = obj.type_system

        if type_system != OpaqueType.MCOS:
            warnings.warn(
                "subsystem:set_object_metadata: Only MCOS objects are supported currently. This item will be skipped",
                MatWriteWarning,
            )
            return np.empty((0, 0), dtype=np.uint8)

        return self.set_mcos_object_metadata(obj)

    def set_fwrap_metadata(self):
        """Create FileWrapper Metadata Array"""

        regions = []
        regions.append(np.array([self.version], dtype=np.uint32).view(np.uint8))
        regions.append(np.array([self.num_names], dtype=np.uint32).view(np.uint8))

        # Region offsets (uint32 array)
        region_offsets = np.zeros(8, dtype=np.uint32)
        regions.append(region_offsets.view(np.uint8))

        # Names string (null-terminated)
        names_bytes = (
            b"\x00".join(name.encode("ascii") for name in self.mcos_names) + b"\x00"
        )
        pad_len = (8 - len(names_bytes) % 8) % 8
        names_bytes += b"\x00" * pad_len
        names_uint8 = np.frombuffer(names_bytes, dtype=np.uint8)
        regions.append(names_uint8)
        region_offsets[0] = 40 + names_uint8.size

        # Region 1 - Class ID Metadata
        region1 = np.array(self.class_id_metadata, dtype=np.uint32).view(np.uint8)
        regions.append(region1)
        region_offsets[1] = region_offsets[0] + region1.size

        # Region 2 - Saveobj Metadata
        self.saveobj_metadata = [x for sub in self.saveobj_metadata for x in sub]
        region2 = (
            np.array(self.saveobj_metadata, dtype=np.uint32).view(np.uint8)
            if len(self.saveobj_metadata) > 2
            else np.empty(0, dtype=np.uint8)
        )
        regions.append(region2)
        region_offsets[2] = region_offsets[1] + region2.size

        # Region 3 - Object ID Metadata
        self.object_id_metadata = [x for sub in self.object_id_metadata for x in sub]
        region3 = np.array(self.object_id_metadata, dtype=np.uint32).view(np.uint8)
        regions.append(region3)
        region_offsets[3] = region_offsets[2] + region3.size

        # Region 4 - Object Prop Metadata
        self.nobj_metadata = [x for sub in self.nobj_metadata for x in sub]
        region4 = (
            np.array(self.nobj_metadata, dtype=np.uint32).view(np.uint8)
            if len(self.nobj_metadata) > 0
            else np.empty(0, dtype=np.uint8)
        )
        regions.append(region4)
        region_offsets[4] = region_offsets[3] + region4.size

        # Region 5 - Dynamic Prop Metadata
        region5 = np.array(self.dynprop_metadata, dtype=np.uint32).view(np.uint8)
        regions.append(region5)
        region_offsets[5] = region_offsets[4] + region5.size

        # Region 6 - Unknown
        region6 = np.empty(0, dtype=np.uint8)
        regions.append(region6)
        region_offsets[6] = region_offsets[5] + region6.size

        # Region 7 - Unknown
        region7 = np.zeros(1, np.uint64).view(np.uint8)
        regions.append(region7)
        region_offsets[7] = region_offsets[6] + region7.size

        fwrap_metadata = np.concatenate(regions)
        return fwrap_metadata.reshape(-1, 1)

    def set_fwrap_data(self):
        """Create FileWrapper Cell Array"""

        if self.class_id_counter == 0:
            # No MCOS objects to save
            return None

        array_len = 5 + len(self.mcos_props_saved)
        fwrap_data = np.empty((array_len, 1), dtype=object)

        fwrap_data[0, 0] = self.set_fwrap_metadata()
        fwrap_data[1, 0] = MatlabCanonicalEmpty()

        for i, prop_val in enumerate(self.mcos_props_saved):
            fwrap_data[2 + i, 0] = prop_val

        # Write Unknowns
        u3_arr = np.empty((self.class_id_counter + 1, 1), dtype=object)
        for i in range(self.class_id_counter + 1):
            u3_arr[i, 0] = EmptyMatStruct(np.empty((1, 0), dtype=object))
        fwrap_data[-3, 0] = u3_arr
        fwrap_data[-1, 0] = u3_arr

        fwrap_data[-2, 0] = np.zeros(
            shape=(self.class_id_counter + 1, 1), dtype=np.int32
        )

        fwrapper = MatlabOpaque(fwrap_data, OpaqueType.MCOS, MCOS_SUBSYSTEM_CLASS)
        return fwrapper

    def set_subsystem(self):
        """Creates subsystem struct array"""

        fwrap_data = self.set_fwrap_data()
        if fwrap_data is None:
            return None

        dtype = [(OpaqueType.MCOS, object)]
        subsys_metadata = np.empty((1, 1), dtype=dtype)
        subsys_metadata[0, 0][OpaqueType.MCOS] = fwrap_data

        return subsys_metadata

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
