classdef Node < handle
    properties
        Name
        Next
    end

    methods
        function obj = Node(name)
            if nargin > 0
                obj.Name = name;
            end
        end

        function setNext(obj, nextNode)
            obj.Next = nextNode;
        end
    end
end
