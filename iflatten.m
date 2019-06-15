classdef iflatten < handle
    properties
        dim_x; % storing input size required during backpropagation
    end
    
    methods
        % forward propagation
        function y = forward(obj, x)
            obj.dim_x = size(x);
            y = reshape(x, [], obj.dim_x(4))';
        end
        
        % backpropagation
        function delta_x = backward(obj, delta_y)
            delta_x = reshape(delta_y', obj.dim_x);
        end    
    end
end