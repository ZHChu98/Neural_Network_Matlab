classdef idropout 
    properties
        keep_prob;
        dropout_mat;
    end
    
    methods
        function obj = dropout(p)
            obj.keep_prob = 1 - p;
        end
        
        function [obj, y] = forward(obj, x)
            obj.dropout_mat = binornd(1, obj.keep_prob, size(x));
            y = x .* obj.dropout_mat;
        end
        
        function [delta_x] = backward(obj, delta_y)
            delta_x = delta_y .* obj.dropout_mat;
        end
    end
end