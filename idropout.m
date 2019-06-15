classdef idropout < handle
    properties
        keep_prob;
        dropout_mat;
    end
    
    methods
        % initialization
        function obj = idropout(p)
            obj.keep_prob = 1 - p;
        end
        
        % forward propagation
        function y = forward(obj, x)
            obj.dropout_mat = binornd(1, obj.keep_prob, size(x)); % Bernoulli distribution
            y = x .* obj.dropout_mat;
        end
        
        % backpropagation
        function delta_x = backward(obj, delta_y)
            delta_x = delta_y .* obj.dropout_mat;
        end
    end
end