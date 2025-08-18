# Mat-IO Module

The `mat-io` module provides tools for load & save with `.mat` files, particularly perfomring read-write operations on user-defined objects or MATLAB datatypes such as `datetime`, `table` and `string`. It uses a wrapper built around `scipy.io` to read-write subsystem data in MAT-files, which is where MATLAB stores serialized object data. MAT-file versions `v7` to `v7.3` are supported.

`mat-io` can load and save almost all types of objects from MAT-files, including user-defined objects and handle class objects. Additionally, it includes utilities to convert the following MATLAB datatypes into their respective _Pythonic_ objects, and vice versa:

- `string`
- `datetime`, `duration` and `calendarDuration`
- `table` and `timetable`
- `containers.Map` and `dictionary`
- `categorical`
- Enumeration Instance Arrays

**Note**: `load_from_mat()` uses a modified version of `scipy`. The modifications include a few minor changes to `scipy.io` to process `mxOPAQUE_CLASS` variables. You can view the changes under `patches/` and apply it manually. Note that you might need to rebuild as parts of the Cython code was modified. Follow the instructions on the [official SciPy documentation](https://scipy.github.io/devdocs/building/index.html#building-from-source).

## Usage

Install using pip

```bash
pip install mat-io
```

### Available Commands

To read MATLAB objects from a `.mat` file:

```python
from matio import load_from_mat

file_path = "path/to/your/file.mat"
data = load_from_mat(
    file_path,
    raw_data=False,
    add_table_attrs=False,
    mdict=None,
    variable_names=None,
    **kwargs
)
print(data)
```

MATLAB Opaque objects are returned as an instance of class `MatioOpaque` with the following attributes:

- `classname`: The class name, including namespace qualifiers (if any).
- `type_system`: An interal MATLAB type identifier. Usually `MCOS`, but could also be `java` or `handle`.
- `properties`: A dictionary containing the property names and property values.

These objects are contained within `numpy.ndarray` in case of object arrays. If the `raw_data` parameter is set to `False`, then `load_from_mat` converts these objects into a corresponding Pythonic datatype.

---

To write Python objects to a `.mat` file:

```python
from matio import save_to_mat

file_path = "path/to/your/file.mat"
mdict = {"var1": data1, "var2": data2}
save_to_mat(
    file_path,
    mdict=mdict,
    version="v7",
    do_compression=True,
    global_vars=None,
    oned_as="row",
)
```

When writing objects, `matio` tries to guess the class name of the object based on its datatype. For example, `pandas.DataFrames` could be read in as `table` or `timetable`. You can also explicitly mention the object by wrapping your data around an instance of `MatioOpaque` as follows:

```python
from matio import MatioOpaque, save_to_mat

df = some_dataframe
mat_df = MatioOpaque(properties=df, classname="table")
mdict = {"table1": mat_df}
data = save_to_mat(file_path="temp.mat", mdict=mdict)
```

For user-defined classes, a dictionary of property name-value pairs must be wrapped around a `MatioOpaque` instance. In case of arrays, these objects should be contained within a `numpy.ndarray` with `dtype=object` and each `MatioOpaque` instance should be flagged with the `is_array` attribute. All user-defined classes default to the `MCOS` type system.

### Notes

- Extra keyword arguments (`**kwargs`) are passed directly to [`scipy.io.loadmat`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.loadmat.html) or [`scipy.io.savemat`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.savemat.html).

- For conversion rules between MATLAB and Python datatypes, see the [documentation](./docs/field_contents.md).

- To differentiate between MATLAB character and string arrays, it would be ideal to wrap a numpy string array around a `MatioOpaque` instance. However, this is not feasible all the time, especially as MATLAB seems to be using strings in most of their new datatypes. Hence, this module uses Scipy's convention of converting Python `str`, `list(str)` or `numpy.array(dtype=<U1/S1)` to a character array. Numpy string arrays with multiple strings is converted to a MATLAB string

## Contribution

Feel free to create a PR if you'd like to add something, or open up an issue if you'd like to discuss! I've also opened an [issue](https://github.com/scipy/scipy/issues/22736) with `scipy.io` detailing some of the workflow, as well as a [PR](https://github.com/scipy/scipy/pull/22847) to develop this iteratively. Please feel free to contribute there as well!

## Acknowledgement

Big thanks to [mahalex](https://github.com/mahalex/MatFileHandler) for their detailed breakdown of MAT-files. A lot of this wouldn't be possible without it.
