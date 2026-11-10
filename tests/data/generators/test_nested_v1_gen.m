% Requires Simulink. R2022b and earlier save CoderInfo as a nested FileWrapper stream
param = Simulink.Parameter(2);
param.CoderInfo.StorageClass = 'ExportedGlobal';

data = struct;
data.param = param;

save('test_nested_v1_v7.mat', '-struct', 'data');
save('test_nested_v1_v73.mat', '-struct', 'data', '-v7.3');
