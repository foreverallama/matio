# Unknowns

This document details the unknown parts of the subsystem, and provides some hints to what they could be.

## Class ID Metadata

This is the part marked by offset 1 inside the first cell of the cell array inside the subsystem. This part usually has the format `(namespace_index, class_name_index, 0, 0)`.
`class_name_index` and `namespace_index` point to class names from the list of all field names and class names. The remaining zeros are of unknown purpose, but are possibly used during desrialization.

## Object ID Metadata

This is the part marked by offset 3 inside the first cell of the cell array inside the subsystem. This part usually has the format `(class_id, 0, 0, saveobj_id, normalobj_id, dependency_id)`.

- `dependency_id` basically tells us how many objects the current object __depends__ on, i.e., if the property of the object is an object itself.
- `saveobj_id` is set if the class definition defines a `saveobj` method with a return type that is not an object of the same class. Otherwise `normalobj_id` is set.
- The zeros are of unknown purpose, but are possible utilized during deserialization.

## Field Contents Metadata

This is the part marked by offsets 2 and 4 inside the first cell of the cell array inside the subsystem. This part usually has the format `(field_index, field_type, field_value)`. If the class defines a `saveobj` method with a custom return type, then this metadata is entered into region 2, else region 4.

- `field_name_index` points to the field name from the list of all field names and class names
- `field_type` indicates if the field is a property (1) or an attribute (2). It is unclear if there are more types.
- `field_value` depends on the flag set by `field_type`
  - If `field_value = 0`, then it points to an enumeration member name in the list of class and property names
  - If `field_value = 1`, then it points to the cell array containing the property value
  - If `field_value = 2`, then it is the raw integer value

## Offset Regions 6, 7 of Cell 1 Metadata

These are the parts marked by offsets 6, and 7 inside the first cell of the cell array inside the subsystem. In all the examples I've studied so far, these were always all zeros.

- Region 6: Never seen this to contain any data. However, my hypothesis is that this field is set under some specific condition similar to the `saveobj_id` flag. One of the unknown flags in object ID metadata might be responsible for setting this.
- Region 7: This was always the last 8 bytes at the end of the cell. These bytes are usually all zeros. Their purpose is unknown.

## Cell[-3] and Cell[-2]

Cell[-3] has the same structure as Cell[-1], i.e., it consists of a `(num_classes + 1, 1)` cell array, where `num_classes` is the number of classes in the MAT-file. Going by Cell[-1], it can be deduced that these structs are ordered by `class_id`, with an first cell being empty. Each cell is in turn a `struct`. Its contents are unknown, but likely contains some kind of class related instantiations. This field is present since `FileWrapper__ v4`.

Cell[-2] is a `mxINT32` array with dimensions `(num_classes + 1, 1)`. Its purpose is unknown. This field is present since `FileWrapper__ v3`. I have only seen one example where its set - when saving an object that subclasses from `matlab.unittest.TestCase`. For this object, the values are set for objects of class `matlab.unittest.Verbosity` and `matlab.unittest.diagnostics.DiagnosticData` as `12` and `20` respectively. However, modifying these values did not affect load in any way. My guess is it relates to some kind of internal schema or versioning. In any case, you can reproduce it using the steps below (example from [MATLAB documentation](https://www.mathworks.com/help/matlab/ref/matlab.unittest.testcase-class.html)):

```MATLAB
%% classdef
classdef SampleTest < matlab.unittest.TestCase
    methods
        function testCase = SampleTest % Constructor method not recommended
            testCase.verifyEqual(1,2)  % Does not produce a test failure
        end
    end

    methods (Test)
       function testSize(testCase)
           testCase.verifySize([1 2 3; 4 5 6],[2 4]) % Produces a test failure
       end
    end
end

%% Create and save object
obj = SampleTest
save('example.mat')
```

## Why do all regions of the subsystem start with zeros or empty arrays?

This is a tricky question to answer. If you've noticed, all of the offset region starts with a bunch of zeros. In fact, within the whole of the subsystem data, there is a general trend of repeating empty elements. Some speculations below:

- Maybe someone forgot to switch between 0 and 1 indexing (lol)
- They are using some kind of recursive method to write each object metadata to the file. The recursive loop ends when no more objects are available to write, resulting in a bunch of zeros.
- Its possibly used to identify and cache some kind of placeholder objects internally that can be re-written later
- MATLAB appears to keep weak references to objects. Maybe an ID of zero allows to load back a deleted weak reference?
