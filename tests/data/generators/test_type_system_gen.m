jobj = java.lang.String('hello world'); % java
excel = actxserver('Excel.Application'); % handle

data = struct;
data.javatype = jobj;
data.handletype = excel;
save('type_systems_v7.mat', '-struct', 'data');
save('type_systems_v73.mat', '-struct', 'data', '-v7.3');

obj = TestClasses.BasicClass;
obj.a = jobj;
after = 42;
save('test_java_property_v7.mat', 'obj', 'after');
save('test_java_property_v73.mat', 'obj', 'after', '-v7.3');
