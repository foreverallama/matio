# Unknowns

This document details the unknown parts of the subsystem, and provides some hints to what they could be.

## Class ID Metadata

This is the part marked by offset 1 inside the first cell of the cell array inside the subsystem. This part usually has the format `(namespace_index, class_name_index, 0, 0)`.
`class_name_index` and `namespace_index` point to class names from the list of all field names and class names. The remaining zeros are of unknown purpose, but are possibly used during desrialization.

## Object ID Metadata

This is the part marked by offset 3 inside the first cell of the cell array inside the subsystem. This part usually has the format `(class_id, 0, 0, saveobj_id, normalobj_id, dependency_id)`. The zeros are of unknown purpose, but are mostly utilized during deserialization. One of them might be linked to the unknown metadata encoded in Region 6.

## Offset Regions 6, 7 of Cell 1 Metadata

These are the parts marked by offsets 6, and 7 inside the first cell of the cell array inside the subsystem. In all the examples I've studied so far, these were always all zeros.

- Region 6: Never seen this to contain any data. However, my hypothesis is that this field is set under some specific condition similar to the `saveobj_id` flag. One of the unknown flags in object ID metadata might be responsible for setting this.
- Region 7: This is always the last 8 bytes at the end of the cell. These bytes are usually all zeros. Their purpose is unknown.

## Cell[-3]

Cell[-3] has the same structure as Cell[-1], i.e., it consists of a `(num_classes + 1, 1)` cell array, where `num_classes` is the number of classes in the MAT-file. Going by Cell[-1], it can be deduced that these structs are ordered by `class_id`, with an first cell being empty. Each cell is in turn a `struct`. Its contents are unknown, but likely contains some kind of class related instantiations. This field is present since `FileWrapper__ v4`.

## Why do all regions of the subsystem start with zeros or empty arrays?

This is a tricky question to answer. If you've noticed, all of the offset region starts with a bunch of zeros. In fact, within the whole of the subsystem data, there is a general trend of repeating empty elements. Some speculations below:

- Maybe someone forgot to switch between 0 and 1 indexing (lol)
- They are using some kind of recursive method to write each object metadata to the file. The recursive loop ends when no more objects are available to write, resulting in a bunch of zeros.
- Its possibly used to identify and cache some kind of placeholder objects internally that can be re-written later
- MATLAB appears to keep weak references to objects. Maybe an ID of zero allows to load back a deleted weak reference?
