%% Object with saveobj returning a struct

obj_saveobj = TestClasses.SaveObjClass;
obj_saveobj.gain = 5;
obj_saveobj.name = 'motor';

data.obj_saveobj = obj_saveobj;

%% Saving

save('test_saveobj_v7.mat', '-struct', 'data');
save('test_saveobj_v73.mat', '-struct', 'data', '-v7.3');
