%% dictionary
% Saved with R2022b, which stores keys or values of most types other than
% double and string one per cell element (IsKeyCombined/IsValueCombined false)

dict_int_keys = dictionary(int8([1 2]), ["a" "b"]);
dict_datetime_vals = dictionary(["a" "b"], datetime(2024, 1, [1 2]));
dict_struct_vals = dictionary(["a" "b"], [struct('f', 1) struct('f', 2)]);
dict_cell_vals = dictionary(["a" "b"], {1, 'x'});
dict_configured_empty = dictionary(string.empty, double.empty);
dict_empty_datetime_keys = dictionary(datetime.empty(0, 1), zeros(0, 1));
dict_categorical_vals = dictionary(["a" "b"], categorical(["p" "q"]));

%% Data store

data = struct;
data.dict_int_keys = dict_int_keys;
data.dict_datetime_vals = dict_datetime_vals;
data.dict_struct_vals = dict_struct_vals;
data.dict_cell_vals = dict_cell_vals;
data.dict_configured_empty = dict_configured_empty;
data.dict_empty_datetime_keys = dict_empty_datetime_keys;
data.dict_categorical_vals = dict_categorical_vals;

%% Save

save('test_dictionary_v7.mat', '-struct', 'data');
save('test_dictionary_v73.mat', '-struct', 'data', '-v7.3')
