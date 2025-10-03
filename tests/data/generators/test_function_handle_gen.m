builtin_fh = @sin;
custom_fh = @myfunc;
anonymous_fh = @(x) x.^2 + 1;

obj = testfunctions();
classmethod_fh = @obj.square;

nested_fh = make_nested();

%% Saving

data.builtin_fh = builtin_fh;
data.custom_fh = custom_fh;
data.anonymous_fh = anonymous_fh;
data.class_fh = classmethod_fh;
data.nested_fh = nested_fh;

save("test_function_handles_v7.mat", "-struct", "data")
save("test_function_handles_v73.mat", "-struct", "data", "-v7.3")

%% nested func

function f = make_nested()
    function y = inner(x)
        y = x + 100;
    end
    f = @inner;
end
