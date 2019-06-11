classdef iconv2d < handle
    properties
        weight;
        dim_W;
        bias;
        activation;
        input;
        dim_input;
        output;
    end
    
    methods
        function obj = iconv2d(w_1, w_2, w_3, w_4, activation)
            obj.weight = 0.01+0.01*randn(w_1, w_2, w_3, w_4);
            obj.dim_W = [w_1, w_2, w_3, w_4];
            obj.bias = 0.01+0.01*randn(1, w_4);
            obj.activation = activation;
        end
        
        function y = forward(obj, x)
            obj.input = x;
            obj.dim_input = size(x);
            
            o_h = obj.dim_input(1) - obj.dim_W(1) + 1;
            o_w = obj.dim_input(2) - obj.dim_W(2) + 1;
            y = zeros(o_h, o_w, obj.dim_W(4), obj.dim_input(4));
            for i = 1:obj.dim_input(4)
                for d = 1:obj.dim_W(4)
                    y_di = 0;
                    for j = 1:obj.dim_W(3)
                        W_jd = squeeze(obj.weight(:, :, j, d));
                        x_ji = squeeze(x(:, :, j, i));
                        y_di = y_di + conv2(x_ji, rot90(W_jd, 2), 'valid');
                    end
                    y_di = bsxfun(@plus, y_di, obj.bias(1, d));
                    y(:, :, d, i) = y_di;
                end
            end
            
            if strcmp(obj.activation, 'tanh') == 1
                y = tanh(y);
            else
                error('activation must be ''tanh'' at this time');
            end
            obj.output = y;
        end
        
        function delta_x = backward(obj, delta_y, learning_rate)
            if strcmp(obj.activation, 'tanh') == 1
                delta_u = (1 - obj.output .^ 2) .* delta_y;
            else
                error('activation must be ''tanh'' at this time');
            end
            delta_x = zeros(obj.dim_input);
            for i = 1:obj.dim_input(4)
                for d = 1:obj.dim_W(4)
                    for j = 1:obj.dim_W(3)
                        delta_x(:, :, j, i) = delta_x(:, :, j, i) + ...
                            conv2(delta_u(:, :, d, i), obj.weight(:, :, j, d), 'full');
                    end
                end
            end
            
            delta_W = zeros(obj.dim_W);
            delta_b = zeros(1, obj.dim_W(4));
            for d = 1:obj.dim_W(4)
                for j = 1:obj.dim_W(3)
                    for i = 1:obj.dim_input(4)
                        delta_W(:, :, j, d) = delta_W(:, :, j, d) + ...
                            conv2(obj.input(:, :, j, i), rot90(delta_u(:, :, d, i), 2), 'valid');
                    end
                end
                tmp = delta_u(:, :, d, :);
                delta_b(d) = sum(tmp(:));
            end
            obj.weight = obj.weight - delta_W .* learning_rate;
            obj.bias = obj.bias - delta_b .* learning_rate;
        end
    end
end

