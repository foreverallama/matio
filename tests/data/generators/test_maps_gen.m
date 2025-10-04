%% containers.Map

map_empty = containers.Map();

keys = [1 2];
vals = {'a' 'b'};
map_numeric_keys = containers.Map(keys, vals);

keys = {'a','b'};
vals = [1,2];
map_char_keys = containers.Map(keys, vals);

keys = ["a", "b"];
vals = [1,2];
map_string_keys = containers.Map(keys, vals);

%% Data store

data = struct;
data.map_empty = map_empty;
data.map_numeric_keys = map_numeric_keys;
data.map_char_keys = map_char_keys;
data.map_string_keys = map_string_keys;

%% dictionary

dict_numeric_keys = dictionary([1, 2, 3], ["apple", "banana", "cherry"]);
dict_string_keys = dictionary(["x", "y", "z"], [10, 20, 30]);
dict_cell_vals = dictionary(["name", "age"], {"Alice", 25});
dict_cell_keys = dictionary({1, 2, 3}, ["one", "two", "three"]);
dict_empty = dictionary;
dict_val_scalar = dictionary([1, 2, 3], 'a');

%% Data store

data.dict_numeric_keys = dict_numeric_keys;
data.dict_string_keys = dict_string_keys;
data.dict_cell_vals = dict_cell_vals;
data.dict_cell_keys = dict_cell_keys;
data.dict_empty = dict_empty;
data.dict_val_scalar = dict_val_scalar;

%% Save

save('test_maps_v7.mat', '-struct', 'data');
save('test_maps_v73.mat', '-struct', 'data', '-v7.3')
