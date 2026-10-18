long_variable_name = string(repmat('A', 1, 100));

%% MATLAB 2026b bumps timetable save version to v8

data1 = [1;2;3];
time_date = datetime(2023,1,1) + days(0:2);
timetable_v8 = timetable(time_date', data1);
timetable_v8_long_varname = timetable(time_date', data1, 'VariableNames', long_variable_name);

%% Save

data.timetable_v8 = timetable_v8;
data.timetable_v8_long_varname = timetable_v8_long_varname;

save('test_timetablev8_v7.mat', '-struct', 'data')
save('test_timetablev8_v73.mat', '-struct', 'data', '-v7.3')
