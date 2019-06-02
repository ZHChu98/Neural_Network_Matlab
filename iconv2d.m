classdef iconv2d
    properties
        weight;
        dim_w;
        bias;
        activation;
        stride;
        padding;
    end
    
    methods
        function obj = iconv2d(w_1, w_2, w_3, w_4, activation, stride, padding)
            obj.weight = gpuArray.randn(w_1, w_2, w_3, w_4);
            obj.dim_w = [w_1, w_2, w_3, w_4];
            obj.bias = gpuArray.randn(1, w_4);
            obj.activation = activation;
            obj.stride = stride;
            obj.padding = padding;
        end
        
        function [y] = forward(obj, x)
            dim_x = size(x);

            o_h = ceil(dim_x(1) / obj.stride);
            o_w = ceil(dim_x(2) / obj.stride);

            pad_h = max((o_h - 1) * obj.stride + obj.dim_w(1) - dim_x(1), 0);
            pad_top = floor(pad_h / 2);
            pad_bottom = pad_h - pad_top;
            pad_w = max((o_w - 1) * obj.stride + obj.dim_w(2) - dim_x(2), 0);
            pad_left = floor(pad_w / 2);
            pad_right = pad_w - pad_left;

            if strcmp(obj.padding, 'same')
                x = [zeros(dim_x(1), pad_left, dim_x(3), dim_x(4)), x, zeros(dim_x(1), pad_right, dim_x(3), dim_x(4))];
                x = [zeros(pad_top, dim_x(2)+pad_w, dim_x(3), dim_x(4)); x; zeros(pad_bottom, dim_x(2)+pad_w, dim_x(3), dim_x(4))];
            else
                error('padding must be ''same'' at this time');
            end

            if strcmp(obj.activation, 'tanh') == 1
                tmp = gpuArray(zeros(o_h, o_w, obj.dim_w(4), dim_x(4)));
                parfor i = 1:dim_x(4)
                    slice = x(:, :, :, i);
                    for i_h = 1:o_h
                        for i_w = 1:o_w
                            if dim_x(3) == 1
                                tmp(o_h, o_w, :, i) = sum(sum(...
                                    slice((i_h-1)*obj.stride+1:(i_h-1)*obj.stride+obj.dim_w(1), (i_w-1)*obj.stride+1:(i_w-1)*obj.stride+obj.dim_w(2), :) .* obj.weight));
                            else
                                tmp(o_h, o_w, :, i) = sum(sum(sum(...
                                    slice((i_h-1)*obj.stride+1:(i_h-1)*obj.stride+obj.dim_w(1), (i_w-1)*obj.stride+1:(i_w-1)*obj.stride+obj.dim_w(2), :) .* obj.weight)));
                            end
                        end
                    end
                end
                y = tanh(tmp + reshape(obj.bias, 1, 1, obj.dim_w(4)));
            else
                error('activation must be ''tanh'' at this time');
            end
        end
        
        function [] = backward()
            %backpropagation
        end
        
        function obj = update()
            %update parameter weight and bias
        end

    end
end

