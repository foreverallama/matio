%% Heterogeneous object array with elements of two subclasses

het = [TestClasses.HetA, TestClasses.HetB];
het(1).a = 5;
het(2).b = 'two';

data.het = het;

%% Saving

save('test_heterogeneous_v7.mat', '-struct', 'data');
save('test_heterogeneous_v73.mat', '-struct', 'data', '-v7.3');
