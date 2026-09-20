long_variable_name = string(repmat('A', 1, 100));
long_dimension_name = string(repmat('R', 1, 100));

%% MATLAB 2025a bumps tables save version to v5

table_v5_long_varname = table([1; 2; 3], 'VariableNames', long_variable_name);
table_v5_long_dimname = table([1; 2; 3], 'VariableNames', "Value", 'DimensionNames', [long_dimension_name, "Variables"]);

%% MATLAB 2025a bumps timetable save version to v7

data1 = [1;2;3];
time_date = datetime(2023,1,1) + days(0:2);
timetable_v7_long_varname = timetable(time_date', data1, 'VariableNames', long_variable_name);
timetable_v7_long_dimname = timetable(time_date', data1, 'VariableNames', "Value", 'DimensionNames', [long_dimension_name, "Variables"]);

%% Save

data.table_v5_long_varname = table_v5_long_varname;
data.table_v5_long_dimname = table_v5_long_dimname;
data.timetable_v7_long_varname = timetable_v7_long_varname;
data.timetable_v7_long_dimname = timetable_v7_long_dimname;

save('test_tablev5_v7.mat', '-struct', 'data')
save('test_tablev5_v73.mat', '-struct', 'data', '-v7.3')
