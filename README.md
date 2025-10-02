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

- `classname`: The class name, including [namespace qualifiers](https://in.mathworks.com/help/matlab/matlab_oop/namespaces.html) (if any).
- `type_system`: An interal MATLAB type identifier. Usually `MCOS`, but could also be `java` or `handle`.
- `properties`: A dictionary containing the property names and property values.

If the `raw_data` parameter is set to `False`, then `load_from_mat` converts these objects into a corresponding Pythonic datatype, if available.

When writing objects, `matio` tries to guess the class name of the object. For example, `pandas.DataFrames` could be read in as `table` or `timetable`. User-defined objects must contain a dictionary of property name, value pairs wrapped around a `MatlabOpaque` instance.

```python
from matio import MatlabOpaque, save_to_mat

prop_map = {"prop1": val1, "prop2": val2}
mat_obj = MatlabOpaque(prop_map, classname="MyClass")
mdict = {"var1": mat_obj}
data = save_to_mat(file_path="temp.mat", mdict=mdict)
```

## Notes

This package uses wrapper classes to represent Matlab object data to help distinguish from basic datatypes. These are as follows:

- `MatlabOpaque`: A wrapper class for all opaque objects with three attributes: `classname`, `type_system`, `properties`. `properties` is a name-value pair dictionary for each property of the class saved to a MAT-file
- `MatlabOpaqueArray`: A wrapper class subclassed from `numpy.ndarray` to represent object arrays. Each item in this array is a `MatlabOpaque` object. MATLAB creates a separate object instance for each object in an array. The same is followed here.
- `MatlabEnumerationArray`: A wrapper class subclassed from `numpy.ndarray` to represent enumeration instance arrays. Each item in this array is of type `enum.Enum`.
- `MatlabContainerMap`: A wrapper class subclassed from `collections.UserDict` to represent `container.Map` objects. During save, dictionaries are converted to a `struct`. Wrap dictionaries around `MatlabContainerMap` to write to `container.Map` instead.

For conversion rules between MATLAB and Python datatypes, see the [documentation](./docs/field_contents.md).

## Contribution

Feel free to create a PR if you'd like to add something, or open up an issue if you'd like to discuss!

## Acknowledgement

Huge thanks to [mahalex](https://github.com/mahalex/MatFileHandler) for their detailed breakdown of MAT-files. A lot of this wouldn't be possible without it.
