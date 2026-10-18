%% Tables with struct array variables

% Generated with R2022b.

table_struct = table(struct('a', {1; 2}, 'b', {'x'; 'y'}), [3; 4], 'VariableNames', {'S', 'Value'});

data.table_struct = table_struct;

%% Save

save('test_table_struct_v7.mat', '-struct', 'data')
save('test_table_struct_v73.mat', '-struct', 'data', '-v7.3')
