"""Reads MCOS subsystem data from MAT files"""

import warnings
from contextlib import contextmanager

import numpy as np
from scipy.io.matlab._mio5 import EmptyStructMarker
from scipy.io.matlab._mio5_params import (
    OPAQUE_DTYPE,
    MatlabFunction,
    MatlabObject,
    MatlabOpaque,
)

from matio.mat_opaque_tools import (
    MatioOpaque,
    convert_mat_to_py,
    convert_py_to_mat,
    guess_class_name,
    mat_to_enum,
)

FILEWRAPPER_INSTANCE = None
OBJECT_CACHE = {}

FILEWRAPPER_VERSION = 4
SYSTEM_BYTE_ORDER = "<u4" if np.little_endian else ">u4"


@contextmanager
def get_matio_context():
    """Context manager for both FileWrapper and object cache"""
    global FILEWRAPPER_INSTANCE, OBJECT_CACHE

    FILEWRAPPER_INSTANCE = None
    OBJECT_CACHE = {}

    try:
        yield
    finally:
        FILEWRAPPER_INSTANCE = None
        OBJECT_CACHE.clear()


def set_file_wrapper():
    """Set global FileWrapper instance"""
    global FILEWRAPPER_INSTANCE
    if FILEWRAPPER_INSTANCE is not None:
        raise RuntimeError(
            "Subsystem data was not cleaned up. Use get_matio_context() to reset"
        )
    FILEWRAPPER_INSTANCE = FileWrapper()
    return FILEWRAPPER_INSTANCE


def get_file_wrapper():
    """Get global FileWrapper instance"""
    if FILEWRAPPER_INSTANCE is None:
        raise RuntimeError(
            "No FileWrapper instance is active. Use get_matio_context() first."
        )
    return FILEWRAPPER_INSTANCE


def get_object_cache():
    """Get global object cache"""
    return OBJECT_CACHE


def is_valid_opaque_object(metadata):
    """Checks if property value is a valid opaque object metadata array"""

    if is_valid_mcos_object(metadata):
        return True

    # TODO: Add checks for other opaque objects

    return False


def is_valid_mcos_object(metadata):
    """Checks if property value is a valid MCOS metadata array"""

    if metadata.dtype.names:
        if "EnumerationInstanceTag" in metadata.dtype.names:
            if (
                metadata[0, 0]["EnumerationInstanceTag"].dtype == np.uint32
                and metadata[0, 0]["EnumerationInstanceTag"].size == 1
                and metadata[0, 0]["EnumerationInstanceTag"] == 0xDD000000
            ):
                return True
        return False

    if not (
        metadata.dtype == np.uint32
        and metadata.ndim == 2
        and metadata.shape == (metadata.shape[0], 1)
        and metadata.size >= 3
    ):
        return False

    if metadata[0, 0] != 0xDD000000:
        return False

    return True


def check_prop_for_opaque(prop):
    """Recursively check if a property value in FileWrapper__ contains opaque objects during load"""

    if not isinstance(prop, np.ndarray):
        return prop

    if is_valid_opaque_object(prop):
        prop = load_opaque_object(prop, classname="", type_system="MCOS")

    elif prop.dtype == object:
        # Iterate through cell arrays
        for idx in np.ndindex(prop.shape):
            cell_item = prop[idx]
            prop[idx] = check_prop_for_opaque(cell_item)

    elif prop.dtype.names:
        # Iterate though struct array
        for idx in np.ndindex(prop.shape):
            for name in prop.dtype.names:
                field_val = prop[idx][name]
                prop[idx][name] = check_prop_for_opaque(field_val)

    return prop


class FileWrapper:
    """Representation class for MATLAB FileWrapper__ data"""

    def __init__(self):
        self.byte_order = SYSTEM_BYTE_ORDER
        self.raw_data = False
        self.add_table_attrs = False
        self._use_strings = True  # Use in future?

        self.version = FILEWRAPPER_VERSION
        self.num_names = 0
        self.region_offsets = None
        self.mcos_names = []
        self.class_id_metadata = []
        self.object_id_metadata = []
        self.saveobj_prop_metadata = []
        self.obj_prop_metadata = []
        self.dynprop_metadata = []
        self._u6_metadata = None  # Unknown Object Metadata
        self._u7_metadata = None  # Unknown Object Metadata
        self.prop_vals_saved = []
        self._c3 = None  # Unknown Class Template
        self._c2 = None  # Unknown Class Template
        self.prop_vals_defaults = None

        # Counters for object creation
        self.saveobj_object_counter = 0
        self.normal_object_counter = 0
        self.class_id_counter = 0
        self.object_id_counter = 0

    def init_load(
        self,
        fwrap_data,
        byte_order=SYSTEM_BYTE_ORDER,
        raw_data=False,
        add_table_attrs=False,
    ):
        """Initializes the FileWrapper instance with metadata from MATLAB FileWrapper__"""
        self.byte_order = "<u4" if byte_order[0] == "<" else ">u4"
        self.raw_data = raw_data
        self.add_table_attrs = add_table_attrs

        fwrap_metadata = fwrap_data[0, 0]

        fromfile_version = np.frombuffer(
            fwrap_metadata, dtype=self.byte_order, count=1, offset=0
        )[0]
        if not 1 < fromfile_version <= self.version:
            raise RuntimeError(
                f"FileWrapper version {fromfile_version} is not supported"
            )

        self.num_names = np.frombuffer(
            fwrap_metadata, dtype=self.byte_order, count=1, offset=4
        )[0]

        self.region_offsets = np.frombuffer(
            fwrap_metadata, dtype=self.byte_order, count=8, offset=8
        )

        # Property and Class Names
        data = fwrap_metadata[40 : self.region_offsets[0]].tobytes()
        raw_strings = data.split(b"\x00")
        self.mcos_names = [s.decode("ascii") for s in raw_strings if s]

        # Class ID Metadata
        self.class_id_metadata = np.frombuffer(
            fwrap_metadata,
            dtype=self.byte_order,
            count=(self.region_offsets[1] - self.region_offsets[0]) // 4,
            offset=self.region_offsets[0],
        )

        # Saveobj Prop Metadata
        self.saveobj_prop_metadata = np.frombuffer(
            fwrap_metadata,
            dtype=self.byte_order,
            count=(self.region_offsets[2] - self.region_offsets[1]) // 4,
            offset=self.region_offsets[1],
        )

        # Object ID Metadata
        self.object_id_metadata = np.frombuffer(
            fwrap_metadata,
            dtype=self.byte_order,
            count=(self.region_offsets[3] - self.region_offsets[2]) // 4,
            offset=self.region_offsets[2],
        )

        # Object Prop Metadata
        self.obj_prop_metadata = np.frombuffer(
            fwrap_metadata,
            dtype=self.byte_order,
            count=(self.region_offsets[4] - self.region_offsets[3]) // 4,
            offset=self.region_offsets[3],
        )

        # Dynamic Prop Metadata
        self.dynprop_metadata = np.frombuffer(
            fwrap_metadata,
            dtype=self.byte_order,
            count=(self.region_offsets[5] - self.region_offsets[4]) // 4,
            offset=self.region_offsets[4],
        )

        # Unknown Region 6 Metadata
        if self.region_offsets[6] > 0:
            # Some versions its reserved
            self._u6_metadata = fwrap_metadata[
                self.region_offsets[5] : self.region_offsets[6]
            ]

        # Unknown Region 7 Metadata
        if self.region_offsets[7] > 0:
            # Some versions its reserved
            self._u7_metadata = fwrap_metadata[
                self.region_offsets[6] : self.region_offsets[7]
            ]

        if fromfile_version == 2:
            self.prop_vals_saved = fwrap_data[2:-1, 0]
        elif fromfile_version == 3:
            self.prop_vals_saved = fwrap_data[2:-2, 0]
            self._c2 = fwrap_data[-2, 0]
        else:
            self.prop_vals_saved = fwrap_data[2:-3, 0]
            self._c3 = fwrap_data[-3, 0]
            self._c2 = fwrap_data[-2, 0]

        self.prop_vals_defaults = fwrap_data[-1, 0]

    def init_save(self, use_strings=True):
        """Initializes save with metadata for object ID = 0"""

        self._use_strings = use_strings
        self.class_id_metadata.extend([0, 0, 0, 0])
        self.object_id_metadata.extend([0, 0, 0, 0, 0, 0])
        self.saveobj_prop_metadata.extend([0, 0])
        self.obj_prop_metadata.extend([0, 0])
        self.dynprop_metadata.extend([0, 0])

    def get_classname(self, class_id):
        """Extracts class name with namespace qualifier for a given object from its class ID"""

        namespace_idx = self.class_id_metadata[class_id * 4]
        classname_idx = self.class_id_metadata[class_id * 4 + 1]

        if namespace_idx == 0:
            namespace = ""
        else:
            namespace = self.mcos_names[namespace_idx - 1] + "."

        classname = namespace + self.mcos_names[classname_idx - 1]
        return classname

    def get_object_metadata(self, object_id):
        """Extracts object dependency IDs for a given object"""

        class_id, _, _, saveobj_id, normobj_id, dep_id = self.object_id_metadata[
            object_id * 6 : object_id * 6 + 6
        ]
        return class_id, saveobj_id, normobj_id, dep_id

    def get_default_properties(self, class_id):
        """Returns the default properties (as dict) for a given class ID"""

        prop_arr = self.prop_vals_defaults[class_id, 0]
        prop_map = {}
        if prop_arr.dtype.names:
            for prop_name in prop_arr.dtype.names:
                prop_map[prop_name] = check_prop_for_opaque(prop_arr[prop_name][0, 0])

        return prop_map

    def get_property_idxs(self, obj_type_id, saveobj_ret_type):
        """Returns the property field indices for an object"""

        prop_field_idxs = (
            self.saveobj_prop_metadata if saveobj_ret_type else self.obj_prop_metadata
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
        """Returns the saved properties (as dict) for an object"""

        save_prop_map = {}

        prop_field_idxs = self.get_property_idxs(obj_type_id, saveobj_ret_type)
        for prop_idx, prop_type, prop_value in prop_field_idxs:
            prop_name = self.mcos_names[prop_idx - 1]
            if prop_type == 0:
                save_prop_map[prop_name] = self.mcos_names[prop_value - 1]
            elif prop_type == 1:
                save_prop_map[prop_name] = check_prop_for_opaque(
                    self.prop_vals_saved[prop_value]
                )
            elif prop_type == 2:
                save_prop_map[prop_name] = prop_value
            else:
                raise ValueError(
                    f"Unknown property type ID:{prop_type} encountered during deserialization"
                )

        return save_prop_map

    def get_dyn_object_id(self, normobj_id):
        """Gets the object ID from normobj ID for dynamicprops objects"""

        num_objects = len(self.object_id_metadata) // 6

        for object_id in range(num_objects):
            block_start = object_id * 6
            block_normobj_id = self.object_id_metadata[block_start + 4]

            if block_normobj_id == normobj_id:
                return object_id

        raise ValueError(f"No dynamicproperty found with ID {normobj_id}")

    def get_dynamic_properties(self, dep_id):
        """Returns dynamicproperties (as dict) for a given object based on dependency ID"""

        offset = self.dynprop_metadata[0]
        for i in range(dep_id):
            nprops = self.dynprop_metadata[offset]
            offset += 1 + nprops
            offset += offset % 2

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
            obj = MatioOpaque(classname=classname, type_system="MCOS")
            obj.properties = self.get_properties(dyn_obj_id)
            dyn_prop_map[f"__dynamic_property__{i + 1}"] = obj

        return dyn_prop_map

    def get_properties(self, object_id):
        """Returns the properties as a dict for a given object ID"""
        if object_id == 0:
            return {}

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

    def get_class_id(self, classname):
        """Gets the class ID for a given class name (including namespace)"""

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

    def set_mcos_name(self, name):
        """Gets or creates index for a name in mcos_names"""
        try:
            return self.mcos_names.index(name) + 1  # 1-based
        except ValueError:
            # Name doesn't exist, add it
            self.mcos_names.append(name)
            self.num_names += 1
            return self.num_names

    def set_mcos_object_metadata(self, obj, class_id, saveobj_ret_type=False):
        """Sets the metadata for a MATLAB MCOS object"""

        object_cache = get_object_cache()

        obj_key = id(obj)
        cache_id = object_cache.get(obj_key, (0, 0))[0]
        if cache_id == 0:
            cache_id = len(object_cache) + 1
            object_cache[obj_key] = (cache_id, obj)
            new_entry = True
        else:
            new_entry = False

        if not new_entry:
            return cache_id

        prop_map = obj.properties
        self.object_id_counter += 1

        saveobj_object_id = 0
        normal_object_id = 0

        if saveobj_ret_type:
            self.saveobj_object_counter += 1
            saveobj_object_id = self.saveobj_object_counter
        else:
            self.normal_object_counter += 1
            normal_object_id = self.normal_object_counter

        object_id_entry = [class_id, 0, 0, saveobj_object_id, normal_object_id, 0]

        object_id_metadata_pos = len(self.object_id_metadata)
        saveobj_metadata_pos = len(self.saveobj_prop_metadata)
        normobj_metadata_pos = len(self.obj_prop_metadata)

        # Process Properties
        prop_names = list(prop_map.keys())
        num_props = len(prop_names)
        prop_fields = [num_props]

        for prop_name in prop_names:
            field_name_idx = self.set_mcos_name(prop_name)

            prop_value = prop_map[prop_name]
            prop_value = find_matio_opaque(prop_value, in_subsystem=True)

            cell_number_idx = len(self.prop_vals_saved)
            self.prop_vals_saved.append(prop_value)
            prop_fields.extend([field_name_idx, 1, cell_number_idx])
            # TODO:Add support for different property types

        if (num_props * 3 + 1) % 2 != 0:
            prop_fields.append(0)  # Padding

        if saveobj_ret_type:
            self.saveobj_prop_metadata[saveobj_metadata_pos:saveobj_metadata_pos] = (
                prop_fields
            )
        else:
            self.obj_prop_metadata[normobj_metadata_pos:normobj_metadata_pos] = (
                prop_fields
            )

        self.object_id_metadata[object_id_metadata_pos:object_id_metadata_pos] = (
            object_id_entry
        )

        # TODO: Add dynamicprop support
        dynprop_entry = [0, 0]
        self.dynprop_metadata = np.append(
            self.dynprop_metadata, np.array(dynprop_entry, dtype=np.uint32)
        )

        return cache_id

    def write_fwrap_metadata(self):
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

        # Region 1
        region1 = np.array(self.class_id_metadata, dtype=np.uint32).view(np.uint8)
        regions.append(region1)
        region_offsets[1] = region_offsets[0] + region1.size

        # Region 2
        region2 = (
            np.array(self.saveobj_prop_metadata, dtype=np.uint32).view(np.uint8)
            if len(self.saveobj_prop_metadata) > 2
            else np.empty(0, dtype=np.uint8)
        )
        regions.append(region2)
        region_offsets[2] = region_offsets[1] + region2.size

        # Region 3
        region3 = np.array(self.object_id_metadata, dtype=np.uint32).view(np.uint8)
        regions.append(region3)
        region_offsets[3] = region_offsets[2] + region3.size

        # Region 4
        region4 = (
            np.array(self.obj_prop_metadata, dtype=np.uint32).view(np.uint8)
            if len(self.obj_prop_metadata) > 0
            else np.empty(0, dtype=np.uint8)
        )
        regions.append(region4)
        region_offsets[4] = region_offsets[3] + region4.size

        # Region 5
        region5 = np.array(self.dynprop_metadata, dtype=np.uint32).view(np.uint8)
        regions.append(region5)
        region_offsets[5] = region_offsets[4] + region5.size

        # Region 6
        region6 = np.empty(0, dtype=np.uint8)
        regions.append(region6)
        region_offsets[6] = region_offsets[5] + region6.size

        # Region 7
        region7 = np.zeros(1, np.uint64).view(np.uint8)
        regions.append(region7)
        region_offsets[7] = region_offsets[6] + region7.size

        fwrap_metadata = np.concatenate(regions)
        return fwrap_metadata.reshape(-1, 1)

    def write_subsys_data(self):
        """Create FileWrapper Cell Array"""

        if self.class_id_counter == 0:
            return None

        array_len = 5 + len(self.prop_vals_saved)
        fwrap_data = np.empty((array_len, 1), dtype=object)

        fwrap_data[0, 0] = self.write_fwrap_metadata()
        fwrap_data[1, 0] = np.empty((0, 0), dtype=object)  # Empty Placeholder

        for i, prop_val in enumerate(self.prop_vals_saved):
            fwrap_data[2 + i, 0] = prop_val

        # Write Unknowns
        u3_arr = np.empty((self.class_id_counter + 1, 1), dtype=object)
        for i in range(self.class_id_counter + 1):
            u3_arr[i, 0] = np.empty((1, 0), dtype=object)

        # Write U3 Array
        fwrap_data[-3, 0] = u3_arr
        fwrap_data[-1, 0] = u3_arr

        fwrap_data[-2, 0] = np.zeros(
            shape=(self.class_id_counter + 1, 1), dtype=np.int32
        )
        return fwrap_data


def load_mcos_enumeration(metadata, type_system):
    """Loads MATLAB MCOS enumeration instance array"""

    file_wrapper = get_file_wrapper()

    classname = file_wrapper.get_classname(metadata[0, 0]["ClassName"].item())
    builtin_class_idx = metadata[0, 0]["BuiltinClassName"].item()
    if builtin_class_idx != 0:
        builtin_class_name = file_wrapper.get_classname(builtin_class_idx)
    else:
        builtin_class_name = np.str_("")

    value_names = [
        file_wrapper.mcos_names[val - 1] for val in metadata[0, 0]["ValueNames"].ravel()
    ]

    enum_vals = []
    value_idx = metadata[0, 0]["ValueIndices"]
    mmdata = metadata[0, 0]["Values"]  # Array is N x 1 shape
    if mmdata.size != 0:
        mmdata_map = mmdata[value_idx]
        for val in np.nditer(mmdata_map, flags=["refs_ok"], op_flags=["readonly"]):
            obj_array = load_mcos_object(val.item(), "MCOS")
            enum_vals.append(obj_array)

    if not file_wrapper.raw_data:
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
    metadata[0, 0]["Values"] = np.array(enum_vals).reshape(value_idx.shape, order="F")

    return MatioOpaque(
        properties=metadata, classname=classname, type_system=type_system
    )


def load_mcos_object(metadata, type_system):
    """Loads MCOS object"""

    file_wrapper = get_file_wrapper()
    object_cache = get_object_cache()

    if metadata.dtype.names:
        return load_mcos_enumeration(metadata, type_system)

    ndims = metadata[1, 0]
    dims = metadata[2 : 2 + ndims, 0]
    nobjects = np.prod(dims)
    object_ids = metadata[2 + ndims : 2 + ndims + nobjects, 0]

    class_id = metadata[-1, 0]
    classname = file_wrapper.get_classname(class_id)

    obj_arr = np.empty((nobjects, 1), dtype=object)

    for i, object_id in enumerate(object_ids):
        if object_id in object_cache:
            obj_arr[i] = object_cache[object_id]
        elif object_id == 0:
            # Empty object, return empty MatioOpaque
            obj = MatioOpaque(classname=classname, type_system=type_system)
            obj.properties = {}
            obj_arr[i] = obj
        else:
            obj = MatioOpaque(classname=classname, type_system=type_system)
            object_cache[object_id] = obj
            obj.properties = file_wrapper.get_properties(object_id)

            if not file_wrapper.raw_data:
                obj = convert_mat_to_py(
                    obj,
                    byte_order=file_wrapper.byte_order,
                    add_table_attrs=file_wrapper.add_table_attrs,
                )

            obj_arr[i, 0] = obj

    if nobjects == 1:
        return obj_arr[0, 0]

    for obj in obj_arr.flat:
        obj.is_array = True
    return obj_arr.reshape(dims, order="F")


def load_opaque_object(metadata, classname, type_system):
    """Loads opaque object"""

    if type_system != "MCOS":
        warnings.warn(
            f"subsystem:load_mcos_object: Loading opaque objects of type "
            f"{type_system} is not supported. Returning metadata"
        )
        obj = MatioOpaque(metadata, classname, type_system)
        return obj

    return load_mcos_object(metadata, type_system)


def create_mcos_metadata(dims, arr_ids, class_id):
    """Creates MCOS metadata array"""

    ndims = len(dims)
    values = [0xDD000000, ndims] + list(dims) + list(arr_ids) + [class_id]
    return np.array(values, dtype=np.uint32).reshape(-1, 1)


def set_object_metadata(obj, saveobj_ret_type=False):
    """Sets metadata for a MatioOpaque object"""

    if isinstance(obj, np.ndarray):
        obj0 = obj.flat[0]
        type_system = obj0.type_system
        classname = obj0.classname
    else:
        type_system = obj.type_system
        classname = obj.classname

    if type_system != "MCOS":
        warnings.warn(
            "subsystem:set_object_metadata: Only MCOS objects are supported currently. This item will be skipped"
        )
        return np.empty((), dtype=np.uint32)

    file_wrapper = get_file_wrapper()

    arr_ids = []
    class_id = file_wrapper.get_class_id(classname)

    if isinstance(obj, np.ndarray):
        # Object Arrays
        for idx in np.ndindex(obj.shape):
            object_id = file_wrapper.set_mcos_object_metadata(obj[idx], class_id)
            arr_ids.append(object_id)
        dims = obj.shape
    else:
        if not isinstance(obj.properties, dict) or obj.classname == "containers.Map":
            obj.properties = convert_py_to_mat(
                obj.properties,
                obj.classname,
            )

        if classname in ("string", "timetable"):
            saveobj_ret_type = True

        object_id = file_wrapper.set_mcos_object_metadata(
            obj, class_id, saveobj_ret_type
        )
        arr_ids.append(object_id)
        dims = (1, 1)

    return create_mcos_metadata(dims, arr_ids, class_id)


def wrap_matlab_opaque(obj):
    """Wrap data in scipy's Matlab Opaque for writing to MAT-files"""

    if isinstance(obj, np.ndarray):
        obj0 = obj.flat[0]
        classname = obj0.classname
        type_system = obj0.type_system
    else:
        classname = obj.classname
        type_system = obj.type_system

    data_new = np.empty((1,), dtype=OPAQUE_DTYPE)
    data_new0 = data_new[0]
    data_new0["_Class"] = classname
    data_new0["_TypeSystem"] = type_system
    data_new0["_ObjectMetadata"] = set_object_metadata(obj)

    data_new = MatlabOpaque(data_new)

    return data_new


def find_matio_opaque(data, in_subsystem=False):
    """Recursively find and wrap MatioOpaque objects in data."""

    if isinstance(data, MatioOpaque):
        if data.classname is None:
            data.classname = guess_class_name(data.properties)
            data.type_system = "MCOS"
        return set_object_metadata(data) if in_subsystem else wrap_matlab_opaque(data)

    if isinstance(data, np.generic):
        # For compatibility
        data = np.asarray(data).reshape(1, 1)

    if isinstance(data, np.ndarray):

        # 2a: datetime
        if np.issubdtype(data.dtype, np.datetime64):
            tmp_obj = MatioOpaque(
                properties=data, classname="datetime", type_system="MCOS"
            )
            return (
                set_object_metadata(tmp_obj)
                if in_subsystem
                else wrap_matlab_opaque(tmp_obj)
            )

        # 2b: duration
        if np.issubdtype(data.dtype, np.timedelta64):
            tmp_obj = MatioOpaque(
                properties=data, classname="duration", type_system="MCOS"
            )
            return (
                set_object_metadata(tmp_obj)
                if in_subsystem
                else wrap_matlab_opaque(tmp_obj)
            )

        if data.dtype.kind in {"U", "S"}:
            if data.dtype.str.endswith("1") or data.size == 1:
                # FIXME: Need a better way to distinguish between char arrays and strings
                return data

            # Read as string array
            tmp_obj = MatioOpaque(
                properties=data, classname="string", type_system="MCOS"
            )
            return (
                set_object_metadata(tmp_obj)
                if in_subsystem
                else wrap_matlab_opaque(tmp_obj)
            )

        # 2b: Cell Arrays
        if data.dtype == object:
            if data.size == 0:
                return data
            if all(
                isinstance(item, MatioOpaque) and item.is_array for item in data.flat
            ):
                return (
                    set_object_metadata(data)
                    if in_subsystem
                    else wrap_matlab_opaque(data)
                )
            for idx in np.ndindex(data.shape):
                data[idx] = find_matio_opaque(data[idx], in_subsystem)
            return data

        # 2c: struct arrays
        if data.dtype.names:
            if set(data.dtype.names) == {"months", "days", "millis"}:
                # calendarDuration
                if not all(
                    np.issubdtype(data[name].item().dtype, np.timedelta64)
                    for name in ("months", "days", "millis")
                ):
                    return data  # If calendarDuration is converted into prop map
                tmp_obj = MatioOpaque(
                    properties=data, classname="calendarDuration", type_system="MCOS"
                )
                return (
                    set_object_metadata(tmp_obj)
                    if in_subsystem
                    else wrap_matlab_opaque(tmp_obj)
                )
            for idx in np.ndindex(data.shape):
                for name in data.dtype.names:
                    data[idx][name] = find_matio_opaque(data[idx][name], in_subsystem)
            return data

        return data

    if isinstance(
        data,
        (
            MatlabObject,
            MatlabFunction,
            MatlabOpaque,
            str,
            list,
            bool,
            int,
            float,
            EmptyStructMarker,
        ),
    ):
        # For scipy compatibility
        # Scipy also accepts tuple, but here it is used for MATLAB dictionary
        return data

    # Everything else: attempt to wrap into MatioOpaque
    classname = guess_class_name(data)
    tmp_obj = MatioOpaque(properties=data, classname=classname, type_system="MCOS")
    return set_object_metadata(tmp_obj) if in_subsystem else wrap_matlab_opaque(tmp_obj)


def create_subsystem_metadata(fwrap_data, java_data=None, handle_data=None):
    """Creates subsystem struct array"""

    if fwrap_data is None and java_data is None and handle_data is None:
        return None

    field_names = []
    field_data = []

    if fwrap_data is not None:
        field_names.append("MCOS")
        data_new = np.empty((1,), dtype=OPAQUE_DTYPE)
        data_new0 = data_new[0]
        data_new0["_Class"] = "FileWrapper__"
        data_new0["_TypeSystem"] = "MCOS"
        data_new0["_ObjectMetadata"] = fwrap_data
        data_new = MatlabOpaque(data_new)
        field_data.append(data_new)

    dtype = [(name, object) for name in field_names]
    subsys_metadata = np.empty((1, 1), dtype=dtype)

    for field_name, data in zip(field_names, field_data):
        subsys_metadata[0, 0][field_name] = data

    return subsys_metadata


def parse_input_dict(mdict):
    """Parses opaque class metadata"""

    file_wrapper = get_file_wrapper()

    for var, data in mdict.items():
        if var[0] == "_":
            continue
        try:
            mdict[var] = find_matio_opaque(data)
        except ValueError as e:
            raise ValueError(f"Error processing variable '{var}': {str(e)}") from e

    fwrap_data = file_wrapper.write_subsys_data()
    subsys_data = create_subsystem_metadata(fwrap_data)
    mdict["__subsystem__"] = subsys_data

    return mdict
