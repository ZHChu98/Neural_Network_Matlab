classdef ifc
    properties
        weight;
        d_W;
        bias;
        input;
    end
    
    methods
        function obj = ifc(n_input, n_output)
            obj.weight = gpuArray.randn(n_input, n_output);
            obj.bias = gpuArray.randn(1, n_output);
            obj.d_W = [n_input, n_output];
        end
        
        function [y] = forward(obj, x)
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

