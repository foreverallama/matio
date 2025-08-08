clear
obj1 = Node('First');
obj2 = Node('Second');

obj1.Next = obj2;
obj2.Next = obj1;

save('test_handle_v7.mat', '-v7');
save('test_handle_v73.mat', '-v7.3');
