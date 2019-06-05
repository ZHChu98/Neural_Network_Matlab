classdef ifc
    properties
        weight;
        bias;
        input;
    end
    
    methods
        function obj = ifc(n_input, n_output)
            obj.weight = 0.1+0.01*randn(n_input, n_output);
            obj.bias = 0.1+0.01*randn(1, n_output);
        end
        
        function [obj, y] = forward(obj, x)
            obj.input = x;
            y = x * obj.weight + obj.bias;
        end
        
        function [obj, delta_x] = backward(obj, delta_y, learning_rate)
            delta_x = delta_y * obj.weight';
            delta_b = sum(delta_y, 1);
            obj.bias = obj.bias - delta_b * learning_rate;
            
            delta_W = obj.input' * delta_y;
            obj.weight = obj.weight - delta_W * learning_rate;
        end
    end
end

