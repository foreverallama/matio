string_scalar = "Hello";
string_array = ["Apple", "Banana", "Cherry"; "Date", "Fig", "Grapes"];
string_empty = "";

data = struct;
data.string_scalar = string_scalar;
data.string_array = string_array;
data.string_empty = string_empty;

save('test_string_v7.mat', '-struct', 'data');
save('test_string_v73.mat', '-struct', 'data', '-v7.3');
