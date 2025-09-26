%% Basic Enums

enum_scalar = TestClasses.EnumClass.enum1;
enum_uint32 = TestClasses.EnumClassWithBase.enum1;

enum_array = [TestClasses.EnumClass.enum1, TestClasses.EnumClass.enum3, TestClasses.EnumClass.enum5;
    TestClasses.EnumClass.enum2, TestClasses.EnumClass.enum4, TestClasses.EnumClass.enum6];

data.enum_scalar = enum_scalar;
data.enum_uint32 = enum_uint32;
data.enum_array = enum_array;

%% Enum Nested

e1 = TestClasses.EnumClass.enum1;
e2 = TestClasses.EnumClass.enum2;
e3 = TestClasses.EnumClass.enum3;
obj1 = TestClasses.BasicClass;

c{1, 1} = e2;
s.InnerProp = e3;

obj1.a = e1;
obj1.b = c;
obj1.c = s;

data.enum_nested = obj1;

%% Saving

save('test_enum_v7.mat', '-struct', 'data');
save('test_enum_v73.mat', '-struct', 'data', '-v7.3');
