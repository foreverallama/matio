# Mat-IO Module

The `mat-io` module provides tools for reading `.mat` files, particularly for extracting contents from user-defined objects or MATLAB datatypes such as `datetime`, `table` and `string`. It uses a wrapper built around `scipy.io` to extract raw subsystem data from MAT-files, which is then parsed and interpreted to extract object data. MAT-file versions `v7` to `v7.3` are supported.

`mat-io` can read almost all types of objects from MAT-files, including user-defined objects and handle class objects. Additionally, it includes utilities to convert the following MATLAB datatypes into their respective _Pythonic_ objects:

- `string`
- `datetime`, `duration` and `calendarDuration`
- `table` and `timetable`
- `containers.Map` and `dictionary`
- `categorical`
- Enumeration Instance Arrays

**Note**: `load_from_mat()` uses a modified fork of `scipy`. The fork currently contains a few minor changes to `scipy.io` to return variable names and object metadata for all objects in a MAT-file. You can view the changes under `patches/` and apply it manually. Note that you might need to rebuild as parts of the Cython code was modified. Follow the instruction on the [official SciPy documentation](https://scipy.github.io/devdocs/building/index.html#building-from-source).

## Usage

Install using pip

```bash
pip install mat-io
```

### Example

To read subsystem data from a `.mat` file:

```python
from matio import load_from_mat

file_path = "path/to/your/file.mat"
data = load_from_mat(file_path, raw_data=False, add_table_attrs=False)
print(data)
```

#### Parameters

- **`file_path`**: `str`
  Full path to the MAT-file.

- **`raw_data`**: `bool`, *optional*
  - If `False` (default), returns object data as raw object data
  - If `True`, converts data into respective Pythonic datatypes (e.g., `string`, `datetime` and `table`).

- **`add_table_attrs`**: `bool`, *optional*
  If `True`, additional properties of MATLAB `table` and `timetable` are attached to the resultant `pandas.DataFrame`. Works only if `raw_data = False`

- **`mdict`**: `dict`, *optional*
  Dictionary into which MATLAB variables will be inserted. If `None`, a new dictionary is created and returned.

- **`variable_names`**: `list` or `string`, *optional*
  The variable names to load from the MAT-file

- **`**kwargs`**:
  Additional keyword arguments passed to [`scipy.io.loadmat`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.loadmat.html). These are only used to call `scipy.io.loadmat` and are not used natively by this package.
  - `spmatrix`
  - `byte_order`
  - `mat_dtype`
  - `chars_as_strings`
  - `verify_compressed_data_integrity`

### MATLAB Opaque Objects

MATLAB Opaque objects are returned as an instance of class `MatioOpaque` with the following attributes:

- `classname`: The class name, including namespace qualifiers (if any).
- `type_system`: An interal MATLAB type identifier. Usually `MCOS`, but could also be `java` or `handle`.
- `properties`: A dictionary containing the property names and property values.

These objects are contained within `numpy.ndarray` in case of object arrays. If the `raw_data` parameter is set to `False`, then `load_from_mat` converts these objects into a corresponding Pythonic datatype. This conversion is [detailed here](https://github.com/foreverallama/matio/tree/main/docs).

## Contribution

Feel free to create a PR if you'd like to add something, or open up an issue if you'd like to discuss! I've also opened an [issue](https://github.com/scipy/scipy/issues/22736) with `scipy.io` detailing some of the workflow, as well as a [PR](https://github.com/scipy/scipy/pull/22847) to develop this iteratively. Please feel free to contribute there as well!

## Acknowledgement

Big thanks to [mahalex](https://github.com/mahalex/MatFileHandler) for their detailed breakdown of MAT-files. A lot of this wouldn't be possible without it.
