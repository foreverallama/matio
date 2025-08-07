"""Reads MCOS subsystem data from MAT files"""

from contextlib import contextmanager

import numpy as np

from matio.mat_opaque_tools import CLASS_TO_FUNCTION, MatOpaque, mat_to_enum

FILEWRAPPER_INSTANCE = None
OBJECT_CACHE = {}

FILEWRAPPER_VERSION = 4


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


def set_file_wrapper(fwrap_data, byte_order, raw_data, add_table_attrs):
    """Set global FileWrapper instance"""
    global FILEWRAPPER_INSTANCE
    if FILEWRAPPER_INSTANCE is not None:
        raise RuntimeError(
            "Subsystem data was not cleaned up. Use get_matio_context() to reset"
        )
    FILEWRAPPER_INSTANCE = FileWrapper(byte_order, raw_data, add_table_attrs)
    FILEWRAPPER_INSTANCE.init_load(fwrap_data)


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


class FileWrapper:
    """Representation class for MATLAB FileWrapper__ data"""

    def __init__(self, byte_order, raw_data, add_table_attrs):
        self.byte_order = "<u4" if byte_order == "<" else ">u4"

        self.raw_data = raw_data
        self.add_table_attrs = add_table_attrs

        self.version = FILEWRAPPER_VERSION
        self.num_names = 0
        self.region_offsets = None
        self.mcos_names = None
        self.class_id_metadata = None
        self.object_id_metadata = None
        self.saveobj_prop_metadata = None
        self.obj_prop_metadata = None
        self.dynprop_metadata = None
        self._u6_metadata = None  # Unknown Object Metadata
        self._u7_metadata = None  # Unknown Object Metadata
        self.prop_vals_saved = None
        self._c3 = None  # Unknown Class Template
        self._c2 = None  # Unknown Class Template
        self.prop_vals_defaults = None

    def init_load(self, fwrap_data):
        """Initializes the FileWrapper instance with metadata from MATLAB FileWrapper__"""
        fwrap_metadata = fwrap_data[0, 0]

        fromfile_version = np.frombuffer(
            fwrap_metadata, dtype=self.byte_order, count=1, offset=0
        )[0]
        if fromfile_version > self.version:
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
        self._u6_metadata = fwrap_metadata[
            self.region_offsets[5] : self.region_offsets[6]
        ]

        # Unknown Region 7 Metadata
        self._u7_metadata = fwrap_metadata[
            self.region_offsets[6] : self.region_offsets[7]
        ]

        self.prop_vals_saved = fwrap_data[2:-3, 0]
        self._c3 = fwrap_data[-3, 0]  # Unknown
        self._c2 = fwrap_data[-2, 0]  # Unknown
        self.prop_vals_defaults = fwrap_data[-1, 0]

    def is_valid_mcos_object(self, metadata):
        """Checks if property value is a valid MCOS metadata array"""

        if not isinstance(metadata, np.ndarray):
            return False

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

    def check_prop_for_mcos(self, prop):
        """Recursively check if a property value in FileWrapper__ contains MCOS objects"""

        if not isinstance(prop, np.ndarray) or isinstance(prop, MatOpaque):
            return prop

        if self.is_valid_mcos_object(prop):
            prop = load_mcos_object(prop, "MCOS")

        elif prop.dtype == object:
            # Iterate through cell arrays
            for idx in np.ndindex(prop.shape):
                cell_item = prop[idx]
                if self.is_valid_mcos_object(cell_item):
                    prop[idx] = load_mcos_object(cell_item, "MCOS")
                else:
                    self.check_prop_for_mcos(cell_item)

        elif prop.dtype.names:
            # Iterate though struct array
            for idx in np.ndindex(prop.shape):
                for name in prop.dtype.names:
                    field_val = prop[idx][name]
                    if self.is_valid_mcos_object(field_val):
                        prop[idx][name] = load_mcos_object(field_val, "MCOS")
                    else:
                        self.check_prop_for_mcos(field_val)

        return prop

    def get_classname(self, class_id):
        """Extracts class name for a given object from its class ID"""

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
                prop_map[prop_name] = self.check_prop_for_mcos(
                    prop_arr[prop_name][0, 0]
                )

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
                save_prop_map[prop_name] = self.check_prop_for_mcos(
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

        raise ValueError(f"No object found with normobj_id {normobj_id}")

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
            classname = self.get_classname(dyn_obj_id)
            obj = MatOpaque(classname, "MCOS")
            obj.properties = self.get_properties(dyn_obj_id)
            dyn_prop_map[f"__dynamic_property__{i + 1}"] = obj

        return dyn_prop_map

    def get_properties(self, object_id):
        """Returns the properties as a dict for a given object ID"""
        if object_id == 0:
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


def load_mcos_enumeration(metadata, type_system):
    """Loads MATLAB MCOS enumeration instance array"""

    file_wrapper = get_file_wrapper()

    classname = file_wrapper.get_classname(metadata[0, 0]["ClassName"].item())
    builtin_class_idx = metadata[0, 0]["BuiltinClassName"].item()
    if builtin_class_idx != 0:
        builtin_class_name = file_wrapper.get_classname(builtin_class_idx)
    else:
        builtin_class_name = None

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

    return MatOpaque(classname, type_system, metadata)


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
            # Empty object, return empty MatOpaque
            obj = MatOpaque(classname, type_system)
            obj.properties = {}
            obj_arr[i] = obj
        else:
            obj = MatOpaque(classname, type_system)
            object_cache[object_id] = obj
            obj.properties = file_wrapper.get_properties(object_id)

            if not file_wrapper.raw_data:
                convert_func = CLASS_TO_FUNCTION.get(classname)
                if convert_func is not None:
                    obj = convert_func(
                        obj.properties,
                        byte_order=file_wrapper.byte_order,
                        add_table_attrs=file_wrapper.add_table_attrs,
                    )

            obj_arr[i, 0] = obj

    if nobjects == 1:
        return obj_arr[0, 0]
    return obj_arr.reshape(dims, order="F")


def load_opaque_object(metadata, classname, type_system):
    """Loads opaque object"""

    if type_system != "MCOS":
        # Return raw metadata for this case
        obj = MatOpaque(classname, type_system)
        obj.properties = metadata
        return obj

    return load_mcos_object(metadata, type_system)
