%% Basic Objects
obj_no_vals = TestClasses.BasicClass;

obj_with_vals = TestClasses.BasicClass;
obj_with_vals.a = 10;

obj_with_default_val = TestClasses.DefaultClass;

data.obj_no_vals = obj_no_vals;
data.obj_with_vals = obj_with_vals;
data.obj_with_default_val = obj_with_default_val;

%% Objects with Nested Properties

obj1 = TestClasses.BasicClass;
obj1.a = 1;
obj1.b = 'Obj1';
c{1,1} = obj1;

obj2 = TestClasses.BasicClass;
obj2.a = 2;
obj2.b = 'Obj2';
s.InnerProp = obj2;

obj_with_nested_props = TestClasses.BasicClass;
obj_with_nested_props.a = obj1;
obj_with_nested_props.b = c;
obj_with_nested_props.c = s;

data.obj_with_nested_props = obj_with_nested_props;

%% Object Array

obja1 = TestClasses.BasicClass;
obja2 = TestClasses.BasicClass;
objb1 = TestClasses.BasicClass;
objb2 = TestClasses.BasicClass;

obja1.a = 1;
obja2.a = 2;
objb1.a = 3;
objb2.a = 4;

obj_array = [obja1, obja2; objb1, objb2];
data.obj_array = obj_array;

%% Handle Class

obj_handle_1 = TestClasses.HandleClass;
obj_handle_2 = obj_handle_1;
obj_handle_2.a = 20;

data.obj_handle_1 = obj_handle_1;
data.obj_handle_2 = obj_handle_2;

%% Saving

save('test_user_defined_v7.mat', '-struct', 'data');
save('test_user_defined_v73.mat', '-struct', 'data', '-v7.3');
