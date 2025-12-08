% Scalars and 2D arrays for all types
int_types = {
    'int8',    int8(42),      int8([1 2 3; 4 5 6])
    'uint8',   uint8(42),     uint8([1 2 3; 4 5 6])
    'int16',   int16(42),     int16([1 2 3; 4 5 6])
    'uint16',  uint16(42),    uint16([1 2 3; 4 5 6])
    'int32',   int32(42),     int32([1 2 3; 4 5 6])
    'uint32',  uint32(42),    uint32([1 2 3; 4 5 6])
    'int64',   int64(42),     int64([1 2 3; 4 5 6])
    'uint64',  uint64(42),    uint64([1 2 3; 4 5 6])
    'single',  single(3.14),  single([1.1 2.2 3.3; 4.4 5.5 6.6])
    'double',  3.14,          [1.1 2.2 3.3; 4.4 5.5 6.6]
};

complex_scalar = 1 + 2i;
complex_array = [1 + 2i; 2 + 4i; 4 + 8i];

char_scalar = 'Hello';
char_array = ['ab'; 'cd'; 'ef'];

logical_array = [true, false, true];
logical_scalar = true;

% Create a structure to store all data
data = struct();

for i = 1:size(int_types, 1)
    type = int_types{i,1};
    scalar = int_types{i,2};
    array2d = int_types{i,3};

    data.([type '_scalar']) = scalar;
    data.([type '_array']) = array2d;
end

data.complex_scalar = complex_scalar;
data.complex_array = complex_array;

data.char_scalar = char_scalar;
data.char_array = char_array;

data.logical_scalar = logical_scalar;
data.logical_array = logical_array;

data.numeric_empty = [];
data.char_empty = '';
data.logical_empty = logical([]);

%% Struct and Cell
% Basic cell array
data.cell_scalar = {'text'};
data.cell_array = {'A', [1 2; 3 4], {true, false}};
data.cell_empty = {};

% Deeply nested cell
nested_cell = {{'level1', {{'level2', {{'level3', 123}}}}}};
data.cell_nested = nested_cell;

% Basic struct
s1.name = 'test';
s1.value = 123;
s1.data = [1 2; 3 4];
data.struct_scalar = s1;

% Struct array
s2(1).id = 1;
s2(1).info = 'first';
s2(2).id = 2;
s2(2).info = 'second';
data.struct_array = s2;

% Deeply nested struct
deep_struct.level1.level2.level3.value = 42;
deep_struct.level1.level2.cell = {{'nested', struct('a', 1, 'b', 2)}};
data.struct_nested = deep_struct;

% Empty Structs
data.struct_no_fields = struct;
data.struct_empty = struct([]);

% Struct with large number of fields
struct_large = struct();
for ii = 1:526 % Total char >= 4096 writes as reference
    fieldName = sprintf('field%d', ii);
    struct_large.(fieldName) = 1;
end
data.struct_large = struct_large;

% Above is not necessarily > 64 KB for HDF5
struct_even_larger = struct();
for ii = 1:4093
    fieldName = sprintf('s%d', ii);
    struct_even_larger.(fieldName) = 2;
end
data.struct_even_larger = struct_even_larger;

%% Sparse Arrays

data.sparse_empty = sparse([]);
data.sparse_col = sparse([0; 1; 0; 3]);
data.sparse_row = sparse([0, 5, 0, 0]);
data.sparse_diag = sparse(diag(1:5));
data.sparse_rec_row = sparse([1 0; 0 2; 3 0; 0 4]);
data.sparse_rec_col = sparse([1 0 0 2; 0 3 0 0]);
data.sparse_symmetric = sparse([1 2 0; 2 3 4; 0 4 5]);
data.sparse_neg = sparse([0 -1 0; 2 0 0; 0 0 3]);
data.sparse_logical = sparse([true false false; false true false; false false true]);
data.sparse_complex = sparse([1+1i 0 0; 0 2-2i 0; 0 0 3+3i]);
data.sparse_nnz = sparse([1 2; 3 4]);
data.sparse_all_zeros = sparse([0 0;0 0]);

%% Save to v7.3 .mat file

save("test_basic_v73.mat", '-struct', 'data', '-v7.3');
save("test_basic_v7.mat", '-struct', 'data', '-v7');
