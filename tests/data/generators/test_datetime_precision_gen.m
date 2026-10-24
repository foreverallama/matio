%% Datetime with sub-millisecond precision (saved as complex data)

dt_subms = datetime(2024,1,1,12,0,0.1234567);

%% Save

data.dt_subms = dt_subms;

save('test_datetime_precision_v7.mat', '-struct', 'data')
save('test_datetime_precision_v73.mat', '-struct', 'data', '-v7.3')
