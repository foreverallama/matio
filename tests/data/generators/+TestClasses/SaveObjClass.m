classdef SaveObjClass
    properties
        gain = 2;
        name = '';
    end
    methods
        function s = saveobj(obj)
            s.gain = obj.gain;
            s.name = obj.name;
        end
    end
    methods (Static)
        function obj = loadobj(s)
            obj = TestClasses.SaveObjClass;
            obj.gain = s.gain;
            obj.name = s.name;
        end
    end
end
