classdef iconv2d
    properties
        weight;
        dim_W;
        bias;
        activation;
        stride;
        padding;
        dim_pad;
        input;
        dim_input; % if padding = 'same', dim_input ~= size(input)
        output;
    end
    
    methods
        function obj = iconv2d(w_1, w_2, w_3, w_4, activation, stride, padding)
            obj.weight = 0.1+0.01*randn(w_1, w_2, w_3, w_4);
            obj.dim_W = [w_1, w_2, w_3, w_4];
            obj.bias = 0.1+0.01*randn(1, w_4);
            obj.activation = activation;
            obj.stride = stride;
            obj.padding = padding;
        end
        
        function [obj, y] = forward(obj, x)
            dim_x = size(x);
            obj.dim_input = dim_x;

            o_h = ceil(dim_x(1) / obj.stride);
            o_w = ceil(dim_x(2) / obj.stride);

            pad_h = max((o_h - 1) * obj.stride + obj.dim_W(1) - dim_x(1), 0);
            pad_top = floor(pad_h / 2);
            pad_bottom = pad_h - pad_top;
            pad_w = max((o_w - 1) * obj.stride + obj.dim_W(2) - dim_x(2), 0);
            pad_left = floor(pad_w / 2);
            pad_right = pad_w - pad_left;
            obj.dim_pad = [pad_top, pad_bottom, pad_left, pad_right];

            if strcmp(obj.padding, 'same')
                x = [zeros(dim_x(1), pad_left, dim_x(3), dim_x(4)), x, zeros(dim_x(1), pad_right, dim_x(3), dim_x(4))];
                x = [zeros(pad_top, dim_x(2)+pad_w, dim_x(3), dim_x(4)); x; zeros(pad_bottom, dim_x(2)+pad_w, dim_x(3), dim_x(4))];
            else
                error('padding must be ''same'' at this time');
            end
            
            obj.input = x;

            if strcmp(obj.activation, 'tanh') == 1
                tmp = zeros(o_h, o_w, obj.dim_W(4), dim_x(4));
                s = obj.stride; 
                d_W = obj.dim_W; 
                for i = 1:dim_x(4)
                    for i_h = 1:o_h
                        for i_w = 1:o_w
                            if dim_x(3) == 1
                                tmp(o_h, o_w, :, i) = sum(sum(...
                                    x((i_h-1)*s+1:(i_h-1)*s+d_W(1), (i_w-1)*s+1:(i_w-1)*s+d_W(2), :, i) .* obj.weight));
                            else
                                tmp(o_h, o_w, :, i) = sum(sum(sum(...
                                    x((i_h-1)*s+1:(i_h-1)*s+d_W(1), (i_w-1)*s+1:(i_w-1)*s+d_W(2), :, i) .* obj.weight)));
                            end
                        end
                    end
                end
                y = tanh(tmp + reshape(obj.bias, 1, 1, d_W(4)));
            else
                error('activation must be ''tanh'' at this time');
            end
            obj.output = y;
        end
        
        function [obj, delta_x] = backward(obj, delta_y, learning_rate)
            if strcmp(obj.activation, 'tanh') == 1
                delta_u = (1 - obj.output .^ 2) .* delta_y;
            else
                error('activation must be ''tanh'' at this time');
            end
            
            delta_b = reshape(sum(sum(sum(delta_u)), 4), 1, obj.dim_W(4));
            obj.bias = obj.bias - delta_b * learning_rate;
            [o_h, o_w, o_d, ~] = size(delta_y);

            d_window = [obj.dim_input(1)+obj.dim_pad(1)+obj.dim_pad(2), ...
                obj.dim_input(2)+obj.dim_pad(3)+obj.dim_pad(4)];
            s = obj.stride;
            d_W = obj.dim_W;
            delta_x_pad = zeros(d_window(1), d_window(2), obj.dim_input(3), obj.dim_input(4));
            for i = 1:obj.dim_input(4)
                delta_x_pad_i = zeros(d_window(1), d_window(2), obj.dim_input(3));
                for i_h = 1:o_h
                    for i_w = 1:o_w
                        tmp = 0;
                        for i_d = 1:o_d
                            tmp = tmp + delta_u(i_h, i_w, i_d, i) .* obj.weight(:, :, :, i_d);
                        end
                        delta_x_pad_i((i_h-1)*s+1:(i_h-1)*s+d_W(1), (i_w-1)*s+1:(i_w-1)*s+d_W(2), :) = ...
                            delta_x_pad_i((i_h-1)*s+1:(i_h-1)*s+d_W(1), (i_w-1)*s+1:(i_w-1)*s+d_W(2), :) + tmp;  
                    end
                end
                delta_x_pad(:, :, :, i) = delta_x_pad_i;
            end
            delta_x = delta_x_pad(obj.dim_pad(1)+1:end-obj.dim_pad(2), obj.dim_pad(3)+1:end-obj.dim_pad(4), :, :);
            
            delta_W = zeros(d_W);
            for i = 1:obj.dim_input(4)
                delta_W_i = zeros(d_W);
                for i_d = 1:o_d
                    delta_W_d_i = 0;
                    for i_h = 1:o_h
                        for i_w = 1:o_w
                             delta_W_d_i = delta_W_d_i + delta_u(i_h, i_w, i_d, i) * ...
                                 obj.input((i_h-1)*s+1:(i_h-1)*s+d_W(1), (i_w-1)*s+1:(i_w-1)*s+d_W(2), :, i);
                        end
                    end
                    delta_W_i(:, :, :, i_d) = delta_W_d_i;
                end
                delta_W = delta_W + delta_W_i;
            end
            obj.weight = obj.weight - delta_W * learning_rate;
        end
    end
end

