jobj = java.lang.String('hello world'); % java
excel = actxserver('Excel.Application'); % handle

data = struct;
data.javatype = jobj;
data.handletype = excel;
save('type_systems_v7.mat', '-struct', 'data');
save('type_systems_v73.mat', '-struct', 'data', '-v7.3');
