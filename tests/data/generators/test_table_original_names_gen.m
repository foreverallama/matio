%% Tables and timetables with names that are not valid identifiers

% Generated with R2022b. MATLAB saves valid identifiers in VariableNames
% and DimensionNames, and the names as given in VariableNamesOriginal and
% DimensionNamesOriginal.

table_original_names = table([1; 2], [3; 4], 'VariableNames', {'p [bar]', 'x(1)'});
table_original_names.Properties.DimensionNames = {'Row Id', 'Variables'};

timetable_original_names = timetable(seconds([1; 2]), [1; 2], 'VariableNames', {'p [bar]'});
timetable_original_names.Properties.DimensionNames = {'Time [s]', 'Variables'};

data.table_original_names = table_original_names;
data.timetable_original_names = timetable_original_names;

%% Save

save('test_table_original_names_v7.mat', '-struct', 'data')
save('test_table_original_names_v73.mat', '-struct', 'data', '-v7.3')
