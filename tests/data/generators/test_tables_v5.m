%% MATLAB 2025a bumps tables save version to v5

long_variable_name = string(repmat('A', 1, 100));
long_dimension_name = string(repmat('R', 1, 100));

table_v5_long_varname = table([1; 2; 3], ...
    'VariableNames', long_variable_name);

table_v5_long_dimname = table([1; 2; 3], ...
    'VariableNames', "Value", ...
    'DimensionNames', [long_dimension_name, "Variables"]);

%% Table v5 Datastore

data.table_v5_long_varname = table_v5_long_varname;
data.table_v5_long_dimname = table_v5_long_dimname;

save('test_tablev5_v7.mat', '-struct', 'data')
save('test_tablev5_v73.mat', '-struct', 'data', '-v7.3')