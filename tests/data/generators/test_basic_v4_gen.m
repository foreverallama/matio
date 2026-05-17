%% Numeric and Char

% Create a structure to store all data
data = struct();

double_scalar = 3.14;
double_array = [1.1 2.2 3.3; 4.4 5.5 6.6];
data.double_scalar = double_scalar;
data.double_array = double_array;
data.numeric_empty = [];

% From MATLAB Doc:
% The precision used by the save command depends on the size and
% type of each matrix.
% Matrices with any noninteger entries and matrices with 10,000 or
% fewer elements are saved in
% floating-point formats requiring 8 bytes per real element.
% Matrices with all integer entries and
% more than 10,000 elements are saved in the following formats,
% requiring fewer bytes per element.

% Expected MOPT values to change below
% However, MOPT = 0 for examples below

data.fp_small = ones(9999, 1);

N = 10002;
data.fp32 = 0.5 * ones(N,1);
data.fp64 = pi * ones(N,1);
data.i32  = 70000 * ones(N,1);
data.i16  = -1000 * ones(N,1);
data.u16  = 50000 * ones(N,1);
data.u8   = 200 * ones(N,1);
%% Complex

complex_scalar = 1 + 2i;
complex_array = [1 + 2i; 2 + 4i; 4 + 8i];

data.complex_scalar = complex_scalar;
data.complex_array = complex_array;

%% Char

char_scalar = 'Hello';
char_array = ['ab'; 'cd'; 'ef'];

data.char_scalar = char_scalar;
data.char_array = char_array;
data.char_empty = '';

%% Sparse Arrays

data.sparse_empty = sparse([]);
data.sparse_col = sparse([0; 1; 0; 3]);
data.sparse_row = sparse([0, 5, 0, 0]);
data.sparse_diag = sparse(diag(1:5));
data.sparse_rec_row = sparse([1 0; 0 2; 3 0; 0 4]);
data.sparse_rec_col = sparse([1 0 0 2; 0 3 0 0]);
data.sparse_symmetric = sparse([1 2 0; 2 3 4; 0 4 5]);
data.sparse_neg = sparse([0 -1 0; 2 0 0; 0 0 3]);
data.sparse_complex = sparse([1+1i 0 0; 0 2-2i 0; 0 0 3+3i]);
data.sparse_nnz = sparse([1 2; 3 4]);
data.sparse_all_zeros = sparse([0 0;0 0]);

%% Save to v4 .mat file

save("test_basic_v4.mat", '-struct', 'data', '-v4');
