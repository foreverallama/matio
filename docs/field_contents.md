# Field Contents

This document describes in detail the saved properties of some common MATLAB datatypes. In MATLAB, most datatypes are saved using default load and save processes. However, a few such as `string` or `timetable` use a `saveobj` and `loadobj` method which can use a custom save value.

<!--TOC-->

- [MATLAB to Python Conversion](#matlab-to-python-conversion)
- [`datetime`](#datetime)
- [`duration`](#duration)
- [`calendarDuration`](#calendarduration)
- [`string`](#string)
- [`table`](#table)
- [`timetable`](#timetable)
- [`containers.Map`](#containersmap)
- [`dictionary`](#dictionary)
- [`categorical`](#categorical)
- [Enumeration Instance Arrays](#enumeration-instance-arrays)
- [Others](#others)

<!--TOC-->

## MATLAB to Python Conversion

The `matio` package attempts to convert some common MATLAB datatypes into a Pythonic datatype, and vice versa. These are detailed below:

| MATLAB Type                       | Python Equivalent                          |
|-----------------------------------|---------------------------------------------|
| `datetime`                        | `numpy.datetime64`|
| `duration`                        | `numpy.timedelta64`|
| `calendarDuration`                | `numpy.ndarray` with fields `{months, days, millis}` |
| `string`                          | `numpy.str_`            |
| `table`                           | `pandas.DataFrame`                          |
| `timetable`                       | `pandas.DataFrame` with datetime or duration index      |
| `containers.Map`                  | `dict`                                      |
| `dictionary`                      | `(keys, values)`                                     |
| `categorical`                     | `pandas.Categorical`                        |
| Enumeration Instance Arrays       | `numpy.ndarray` of `enum.Enum`         |
| User-Defined Classes              | `MatioOpaque` instance with property map `dict` |

## `datetime`

Objects of this class contain the following properties:

- `data`: A complex double precision number. The real part contains the milliseconds, and the imaginary part contains the microseconds. The date/time is calculated from the UNIX epoch
- `tmz` or Timezone: A UTF-8 string
- `fmt` or Format: The display format to use. e.g. `YYYY-MM-DD HH:MM:SS`

`matio.load_from_mat` converts these objects into `numpy.datetime64[ms]` arrays if `raw_data` is set to `False`. Timezone offsets, if present, is automatically applied using `zoneinfo`.

## `duration`

Objects of this class contain the following properties:

- `millis`: A real double precision number containing time in milliseconds
- `fmt` or Format: The display format to use. e.g. `s`, `m`, `h`, `d` for `seconds`, `minutes`, `hours`, `days`

`matio.load_from_mat` converts these objects into `numpy.timedelta64` arrays if `raw_data` is set to `False`. The `dtype` of the array is set according to `fmt`. It defaults to `[ms]` if `fmt` could not be parsed.

## `calendarDuration`

Objects of this class contains a single property `components` which is defined as a `struct` array with the following fields:

1. `months`
2. `days`
3. `millis`
4. `fmt`: Character Array

`matio.load_from_mat` converts this into a structured `numpy.ndarray` with fields `months`, `days` and `millis` which contains the above properties as `numpy.timedelta64`.

## `string`

Objects of this class contains only one property called `any` indicating the use of a `saveobj` method. This `saveobj` method returns a `uint64` integer array in the following format:

- First integer most likely indicates the version of the saved `string` object
- Second integer specifies the number of dimensions `ndims`
- The next `ndims` integers specify the size of each dimension
- The next `K` integers specify the number of characters in each string in the array. Here `K` is the total number of strings in the array (which is the product of all the dimensions)
- The remaining bytes store the string contents. However, these remaining bytes are to be read as `UTF-16` characters

`matio.load_from_mat` converts these objects into `numpy.str_` arrays.

## `table`

Objects of this class contain the following properties:

1. `data`: A cell array, with each cell containing the values for that column
2. `ndims`: A double precision number describing the dimensions of the table, which is usually 2
3. `nrows`: A double precision number describing the number of rows in the table
4. `rownames`: A cell array. Each cell is a character array containing the row name (similar to `df.index`). If no row names are specified, this is an empty cell array
5. `nvars`: A double precision number describing the number of columns/variables
6. `varnames`: A cell array. Each cell is a character array containing the column name.
7. `props`: A `1x1` struct with the following fields, mostly containing some extended metadata:
   1. `useVariableNamesOriginal`
   2. `useDimensionNamesOriginal`
   3. `CustomProps`
   4. `VariableCustomProps`
   5. `versionSavedFrom`
   6. `minCompatibleVersion`
   7. `incompatibilityMsg`
   8. `VersionSavedFrom`
   9. `Description`
   10. `VariableNamesOriginal`
   11. `DimensionNames`
   12. `DimensionNamesOriginal`
   13. `UserData`
   14. `VariableDescriptions`
   15. `VariableUnits`
   16. `VariableContinuity`

`matio.load_from_mat` converts these objects into `pandas.DataFrame` objects.

## `timetable`

Objects of this class contains a single property `any` indicating the use of a `saveobj` method. This `saveobj` method returns a `struct` array with the following fields:

1. `arrayProps`: A `1x1` struct with the following fields:
   1. `Description`
   2. `UserData`
   3. `TableCustomProperties`
2. `data`: A cell array, with each cell containing a variable/column values
3. `numDims`: A double precision number describing the number of dimensions of the timetable, which is usually 2
4. `dimNames`: A cell array. Each cell is a character array containing the dimension names
5. `varNames`: A cell array. Each cell is a character array containing the variable/column names.
6. `numRows`: A double precision number containing the number of rows in the timetable
7. `numVars`: A double precision number containign the number of variables/columns in the timetable
8. `varUnits`: A cell array. Each cell is a character array containing the units of the variables used
9. `rowTimes`: An array of times (typically `duration` or `datetime`). This would be an object reference to the relevant object.

The remaining fields are metadata fields:

1. `customProps`
2. `VariableCustomProps`
3. `versionSavedFrom`
4. `minCompatibleVersion`
5. `incompatibilityMsg`
6. `useVarNamesOrig`
7. `useDimNamesOrig`
8. `dimNamesOrig`
9. `varNamesOrig`
10. `varDescriptions`
11. `timeEvents`
12. `varContinuity`

`matio.load_from_mat` converts these objects into `pandas.DataFrame` objects with row indices indicating the timesteps.

## `containers.Map`

Objects of this class contains a single property `serialization`, which is defined as a `struct` array with the following fields:

1. `keys`: A cell array containing the key names
2. `values`: A cell array containing the values
3. `uniformity`: bool, indicating uniform values
4. `keyType`: Char array indicating `dtype` of keys
5. `valueType`: Char array indicating `dtype` of values

`matio.load_from_mat` converts these objects into a dictionary of `{key: value}`.

## `dictionary`

Objects of this class contain a single property `data` which is defined as a `struct` array with the following fields:

1. `Version`
2. `IsKeyCombined`: bool, unknown indicator, presumably for optimization purposes
3. `IsValueCombined`: bool, unknown indicator, presumably for optimization purposes
4. `Key`
5. `Value`

Since keys can be of any MATLAB datatype, including object instances, `matio.load_from_mat` converts this into a tuple `(keys, values)`. Each value in the tuple contains all the keys and values for the dictionary. These values are not split up as MATLAB optimizes on object representation.

## `categorical`

Objects of this class contain the following properties:

1. `categoryNames`: A cell array of character arrays or strings
2. `codes`: An unsigned integer array specifying the category codes
3. `isProtected`: bool, indicates protected categories
4. `isOrdinal`: bool, indicates ordered variables

`matio.load_from_mat` converts these objects into `pandas.Categorical` objects.

## Enumeration Instance Arrays

Enumeration instance arrays are stored as `mxOPAQUE_CLASS` arrays of `MCOS` type. The object metadata for this (in the main part of the MAT-file) is returned as a `struct` array containing the following fields:

1. `EnumerationInstanceTag`: Contains the reference value `0xDD00000000`.
2. `ClassName`: A metadata indicator to extract the enumeration class name from subsystem.
3. `ValueNames`: A metadata indicator to extract the property names of the enumeration class from subsystem.
4. `Values`: Contains an array of object references, which are used to extract the contents of each instance of the enumeration array from subsytem. If the properties of the enumeration class are not initialized/instantiated, then this is an empty array.
5. `ValueIndices`: The value indices of the enumeration array. This also indicates the dimensions of the enumeration array.
6. `BuiltinClassName`: This is set if the enumeration class specifies a superclass. The value is a metadata indicator to extract the name of the superclass.

Enumerations arrays are returned as a `numpy` array of members of `enum.Enum` class.

## Others

To add:

1. `function_handle`
2. `timeseries`
