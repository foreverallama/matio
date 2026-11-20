classdef EnumEqualProps
   properties
       val;
   end
   enumeration
      Pending (1)
      Deferred (1)
      Complete (2)
   end
   methods
       function c = EnumEqualProps(num)
           c.val = num;
       end
   end
end
