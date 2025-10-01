%% Tables

table_numeric = table([1.1; 2.2; 3.3], [4.4; 5.5; 6.6]);
table_strings = table(["apple"; "banana"; "cherry"]);
table_empty = table;

Time = datetime(2020,1,1) + days(0:2)';
Duration = seconds([30; 60; 90]);
table_time = table(Time, Duration);

S = struct('field1', 123, 'field2', 'abc');
C = {S; S; S};
obj = TestClasses.BasicClass;
obj.a = 1;
table_with_objects = table(C, {obj; obj; obj});

table_from_cell = table({1; 'text'; datetime(2023,1,1)});

table_nan = table([1.1; NaN; 3.3], ["A"; ""; "C"]);

table_var_names = table([10; 20; 30], [100; 200; 300], 'VariableNames', {'X', 'Y'});

table_row_names = table([1; 2; 3], [4; 5; 6], ...
    'VariableNames', {'Col1', 'Col2'}, ...
    'RowNames', {'R1', 'R2', 'R3'});

table_with_attrs = table([1; 2], ["one"; "two"], 'VariableNames', {'ID', 'Label'});
table_with_attrs.Properties.Description = 'Test table with full metadata';
table_with_attrs.Properties.DimensionNames = {'RowId', 'Features'};
table_with_attrs.Properties.UserData = struct('CreatedBy', 'UnitTest', 'Version', 1.0);
table_with_attrs.Properties.VariableUnits = {'', 'category'};
table_with_attrs.Properties.VariableDescriptions = {'ID number', 'Category label'};
table_with_attrs.Properties.VariableContinuity = {'continuous', 'step'};

time = datetime(2023,1,1) + days(0:2);
multicoldata = [1,4;2,5;3,6];
table_multi_col_data = table(time', multicoldata);

%% Table Datastore

data.table_empty = table_empty;
data.table_numeric = table_numeric;
data.table_strings = table_strings;
data.table_time = table_time;
data.table_with_objects = table_with_objects;
data.table_from_cell = table_from_cell;
data.table_nan = table_nan;
data.table_var_names = table_var_names;
data.table_row_names = table_row_names;
data.table_with_attrs = table_with_attrs;
data.table_multi_col_data = table_multi_col_data;

%% Timetables

data1 = [1;2;3];
data2 = [1,4;2,5;3,6];

time_date = datetime(2023,1,1) + days(0:2);
timetable_datetime = timetable(time_date', data1);

time_duration = seconds([10,20,30]);
timetable_duration = timetable(time_duration', data1);

timetable_multi_col = timetable(time_duration', data2);

timetable_var_names = timetable(time_date', data1, 'VariableNames', {'Pressure'});
timetable_empty = timetable;

%%% Timetable with TimeSteps

calm = calmonths(3);

timetable_from_sample_rate = timetable(data1, 'SampleRate',10000);
timetable_from_duration = timetable(data1, 'TimeStep',seconds(1));
timetable_from_starttime_duration = timetable(data1, 'TimeStep',seconds(1), 'StartTime', seconds(10));
timetable_from_starttime_datetime = timetable(data1, 'TimeStep',seconds(1), 'StartTime', datetime(2020,1,1));
timetable_from_starttime_calendarDuration = timetable(data1, 'TimeStep', calm, 'StartTime', datetime(2020,1,1));

timetable_with_attrs = timetable(time_date', data1);

% Set units and continuity
timetable_with_attrs.Properties.DimensionNames = {'Date','Pressure'};
timetable_with_attrs.Properties.Description = "Random Description";
timetable_with_attrs.Properties.VariableDescriptions = {'myVar'};
timetable_with_attrs.Properties.VariableUnits = {'m/s'};
timetable_with_attrs.Properties.VariableContinuity = {'continuous'};

%% Timetable Datastore

data.timetable_datetime = timetable_datetime;
data.timetable_duration = timetable_duration;
data.timetable_multi_col = timetable_multi_col;
data.timetable_var_names = timetable_var_names;
data.timetable_from_sample_rate = timetable_from_sample_rate;
data.timetable_from_duration = timetable_from_duration;
data.timetable_from_starttime_duration = timetable_from_starttime_duration;
data.timetable_from_starttime_datetime = timetable_from_starttime_datetime;
data.timetable_from_starttime_calendarDuration = timetable_from_starttime_calendarDuration;
data.timetable_with_attrs = timetable_with_attrs;
data.timetable_empty = timetable;

%% Categorical

cat_scalar = categorical({'red', 'green', 'blue', 'red'});
cat_array = categorical({'low', 'medium'; 'high', 'low'});
cat_unordered = categorical({'cold', 'hot', 'warm'}, {'cold', 'warm', 'hot'});
cat_ordered = categorical({'small', 'medium', 'large'}, ...
                   {'small', 'medium', 'large'}, 'Ordinal', true);

cat_from_numeric = categorical([1, 2, 3, 2, 1], [1, 2, 3], {'low', 'medium', 'high'});
cat_empty = categorical({});
cat_missing = categorical({'cat', '', 'dog', 'mouse'}, ...
                   {'cat', 'dog', 'mouse'});

cat_string = categorical(["spring", "summer", "autumn", "winter"]);
cat_mixed_case = categorical({'On', 'off', 'OFF', 'ON', 'on'});
cat_3D = categorical(repmat(["yes", "no", "maybe"], [2, 1, 2]));

%% Categorical Datastore

data.cat_scalar = cat_scalar;
data.cat_array = cat_array;
data.cat_3D = cat_3D;
data.cat_unordered = cat_unordered;
data.cat_ordered = cat_ordered;
data.cat_from_numeric = cat_from_numeric;
data.cat_empty = cat_empty;
data.cat_missing = cat_missing;
data.cat_string = cat_string;
data.cat_mixed_case = cat_mixed_case;

%% Save

save('test_tables_v7.mat', '-struct', 'data')
save('test_tables_v73.mat', '-struct', 'data', '-v7.3')
