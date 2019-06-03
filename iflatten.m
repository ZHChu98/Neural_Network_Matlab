classdef iflatten
    properties
        dim_x;
    end
    
    methods
        function [obj, y] = forward(obj, x)
            obj.dim_x = size(x);
            y = reshape(x, obj.dim_x(1)*obj.dim_x(2)*obj.dim_x(3), obj.dim_x(4))';
        end
        
        function [delta_x] = backward(obj, delta_y)
            delta_x = reshape(delta_y', obj.dim_x);
        end    
    end
end