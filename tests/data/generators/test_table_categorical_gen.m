%% Tables and timetables with categorical variables

% Generated with R2022b.

levels = categorical({'low'; 'high'; 'low'});

table_categorical = table(levels, [1; 2; 3], 'VariableNames', {'Level', 'Value'});
timetable_categorical = timetable(seconds([1; 2; 3]), levels, 'VariableNames', {'Level'});

data.table_categorical = table_categorical;
data.timetable_categorical = timetable_categorical;

%% Save

save('test_table_categorical_v7.mat', '-struct', 'data')
save('test_table_categorical_v73.mat', '-struct', 'data', '-v7.3')
