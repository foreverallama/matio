%% Tables with char array columns

% Generated with R2024b. From R2025a on tables are saved with
% versionSavedFrom = 5 and mat_to_table does not convert them.

table_char = table(['ab ';'cde'], [1;2], 'VariableNames', {'Code', 'Value'});
table_char_single = table(['a';'b'], 'VariableNames', {'Code'});

data.table_char = table_char;
data.table_char_single = table_char_single;

%% Save

save('test_table_char_v7.mat', '-struct', 'data')
save('test_table_char_v73.mat', '-struct', 'data', '-v7.3')
