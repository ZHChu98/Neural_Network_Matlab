classdef iconv2d < handle
    properties
        weight;
        dim_W;
        bias;
        activation;
        padding
        dim_P = [0, 0, 0, 0];
        input;
        dim_input;
        output;
        dim_output;
        relu_mat;
    end
    
    methods
        function obj = iconv2d(w_1, w_2, in_channel, out_channel, activation, padding)
            obj.weight = 0.01+0.01*randn(w_1, w_2, in_channel, out_channel);
            obj.dim_W = [w_1, w_2, in_channel, out_channel];
            obj.bias = 0.01+0.01*randn(1, out_channel);
            obj.activation = activation;
            obj.padding = padding;
        end
        
        function y = forward(obj, x)
            if strcmp(obj.padding, 'same') == true
                pad_up = floor((obj.dim_W(1)-1) / 2);
                pad_down = obj.dim_W(1) - pad_up - 1;
                pad_left = floor((obj.dim_W(2)-1) / 2);
                pad_right = obj.dim_W(2) - pad_left - 1;
                obj.dim_P = [pad_up, pad_down, pad_left, pad_right];
                x = padarray(x, [pad_up, pad_left], 0, 'pre');
                x = padarray(x, [pad_down, pad_right], 0, 'post');
            elseif strcmp(obj.padding, 'valid') == false
                error('wrong type for padding');
            end
            
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
            obj.dim_output = size(y);
            
            if strcmp(obj.activation, 'tanh') == true
                y = tanh(y);
            elseif strcmp(obj.activation, 'sigmoid') == true
                y = 1 ./ (1 + exp(-y));
            elseif strcmp(obj.activation, 'relu') == true
                tmp = reshape([y; zeros(obj.dim_output)], obj.dim_output(1), 2, ...
                    obj.dim_output(2), obj.dim_output(3), obj.dim_output(4));
                tmp = permute(tmp, [2, 1, 3, 4, 5]);
                [tmp1, tmp2] = max(tmp);
                y = squeeze(tmp1);
                obj.relu_mat = squeeze(tmp2) - 1;
            else
                error('wrong type for activation');
            end
            obj.output = y;
        end
        
        function delta_x = backward(obj, delta_y, learning_rate)
            if strcmp(obj.activation, 'tanh') == true
                delta_u = (1-obj.output.^2) .* delta_y;
            elseif strcmp(obj.activation, 'sigmoid') == true
                delta_u = (1-obj.output) .* obj.output .* delta_y;
            elseif strcmp(obj.activation, 'relu') == true
                delta_u = obj.relu_mat .* delta_y;
            else
                error('wrong type for activation');
            end
            delta_x_pad = zeros(obj.dim_input);
            for i = 1:obj.dim_input(4)
                for d = 1:obj.dim_W(4)
                    for j = 1:obj.dim_W(3)
                        delta_x_pad(:, :, j, i) = delta_x_pad(:, :, j, i) + ...
                            conv2(delta_u(:, :, d, i), obj.weight(:, :, j, d), 'full');
                    end
                end
            end
            delta_x = delta_x_pad(obj.dim_P(1)+1:end-obj.dim_P(2), obj.dim_P(3)+1:end-obj.dim_P(4), :, :);
            
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

