%% Table with a matrix variable whose split names clash with another variable

% Generated with R2022b.

table_name_clash = table([1 2; 3 4], [5; 6], 'VariableNames', {'m', 'm_1'});

data.table_name_clash = table_name_clash;

%% Save

save('test_table_name_clash_v7.mat', '-struct', 'data')
save('test_table_name_clash_v73.mat', '-struct', 'data', '-v7.3')
