# Mat-IO Module

The `mat-io` module provides tools for loading and saving MAT-files, including MATLAB's classdef-based datatypes such as `datetime`, `table` and `string`. It supports almost all MATLAB object types, including user-defined objects and handle class objects. Additionally, it includes utilities to convert the following MATLAB datatypes into their respective _Pythonic_ objects, and vice versa:

- `string`
- `datetime`, `duration` and `calendarDuration`
- `table` and `timetable`
- `containers.Map` and `dictionary`
- `categorical`
- Enumeration Instance Arrays

MAT-file versions `v6`, `v7` and `v7.3` are supported.

- Versions `v6` and `v7` uses a modified version of `scipy.io` under the hood
- Version `v7.3` uses `h5py` to write in the HDF5 format.

Data is returned in the same format as `scipy.io.loadmat` does.

## Installation

```bash
pip install mat-io
```

## Usage

### Loading MAT-files

```python
from matio import load_from_mat

file_path = "path/to/your/file.mat"
data = load_from_mat(
    file_path,
    raw_data=False,
    add_table_attrs=False,
    mdict=None,
    variable_names=None,
)
```

- `raw_data`: If `True`, returns raw property maps of Opaque class objects.
- `add_table_attrs`: If `True`, adds custom Matlab Table or Timetable properties as `pandas.DataFrame` attributes.
- `mdict`: If provided, this dictionary will be updated with the data from a MAT-file.
- `variable_names`: A list of variable names to load from file.

### Saving MAT-files

```python
from matio import save_to_mat

file_path = "path/to/your/file.mat"
mdict = {"var1": data1, "var2": data2}
save_to_mat(
    file_path,
    mdict=mdict,
    version="v7.3",
    global_vars=None,
    oned_as="col",
    do_compression=True,
)
```

- `file_path`: The file to save data.
- `mdict`: A dictionary of `{var_name: var_data}` to write to MAT-files.
- `version`: The MAT-file version to save to. Supported versions are `v7.3` and `v7`. Defaults to `v7.3` which is based on the HDF5 format.
- `global_vars`: A list of variable names that are to be marked as global variables.
- `saveobj_classes`: A list of class names that implement `saveobj` methods.
- `oned_as`: The 2D shape to apply to unit `numpy.ndarrays`. Either `row` or `col`. Defaults to `col`.
- `do_compression`: If `False`, does not compress data when saving.

### List variables in a MAT-file

```python
from matio import whosmat

file_path = "path/to/your/file.mat"
vars = whosmat(file_path)
# Returns (variable_name, dims, datatype/classname)
print(vars)
```

## Opaque Class Objects

Opaque class objects are what MATLAB calls object instances. Opaque objects have different types. The most common is `MCOS`, which is used for all user-defined classdefs, enumeration classes, as well as most MATLAB datatypes like `string`, `datetime` and `table.`

Opaque objects are returned as an instance of class `MatlabOpaque` with the following attributes:

- `classname`: The class name, including [namespace qualifiers](https://www.mathworks.com/help/matlab/matlab_oop/namespaces.html) (if any).
- `type_system`: An interal MATLAB type identifier. Usually `MCOS`, but could also be `java` or `handle`.
- `properties`: A dictionary containing the property names and property values.
- `class_alias`: This is an optional attribute containing [class aliases](https://www.mathworks.com/help/matlab/matlab_oop/class-aliasing.html), if any.

If the `raw_data` parameter is set to `False`, then `load_from_mat` converts these objects into a corresponding Pythonic datatype, if available. For a list of conversion rules between MATLAB and Python datatypes, see the [documentation](./docs/field_contents.md).

When writing objects, `matio` tries to guess the class name of the object. For example, `pandas.DataFrames` could be read in as `table` or `timetable`. User-defined objects must contain a dictionary of property name, value pairs wrapped around a `MatlabOpaque` instance.

```python
from matio import save_to_mat
from matio.utils import MatlabOpaque

prop_map = {"prop1": val1, "prop2": val2}
mat_obj = MatlabOpaque(properties=prop_map, classname="MyClass")
mdict = {"var1": mat_obj}
data = save_to_mat(file_path="temp.mat", mdict=mdict)
```

If the class implements a custom `saveobj` method, this can be specified in the `saveobj_classes` argument in `save_to_mat`. The property map for such classes **must** contain a single property called `any`. Typically, the value of this property would be in a format output by the `saveobj` method. This would be the input for the `loadobj` method in the class, if defined. More information [here](https://www.mathworks.com/help/matlab/matlab_oop/code-patterns-for-saveobj-and-loadobj.html).

## Wrapper Classes

This package uses wrapper classes to represent Matlab object data to help distinguish from basic datatypes. These are as follows:

- `MatlabOpaque`: A wrapper class for all opaque objects with three attributes: `properties`, `classname`, `type_system`. `properties` is a name-value pair dictionary for each property of the class saved to a MAT-file
- `MatlabOpaqueArray`: A wrapper class subclassed from `numpy.ndarray` to represent object arrays. Each item in this array is a `MatlabOpaque` object.
- `MatlabEnumerationArray`: A wrapper class subclassed from `numpy.ndarray` to represent enumeration instance arrays. Each item in this array is of type `enum.Enum`.
- `MatlabContainerMap`: A wrapper class subclassed from `collections.UserDict` to represent `container.Map` objects. During save, dictionaries are converted to a `struct`. Wrap dictionaries around `MatlabContainerMap` to write to `container.Map` instead.

To save these types to a MAT-file, data must be wrapped around the relevant wrapper class. These can be imported from `matio.utils`. An example is shown below:

```python
# Save dictionary as container.Map
from matio.utils import MatlabContainerMap
from matio import save_to_mat

map = {"a": 1, "b": 2}
map = MatlabContainerMap(map)
var_dict = {"myVar": map}
save_to_mat("file.mat", var_dict)
```

## Contribution

Feel free to create a PR if you'd like to add something, or open up an issue if you'd like to discuss!

## Acknowledgement

Huge thanks to [mahalex](https://github.com/mahalex/MatFileHandler) for their breakdown of MAT-files. A lot of this wouldn't be possible without it.
