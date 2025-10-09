obj = TestClasses.BasicDynamic('Example');

p = addprop(obj, 'DynamicData');
p.Description = 'Test Dyanmic Property';
obj.DynamicData = 42;

save('test_dynamic_v7.mat','obj');
save('test_dynamic_v73.mat','obj', '-v7.3');
