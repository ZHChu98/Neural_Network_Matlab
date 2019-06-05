classdef ipooling
    properties
        dim_pool;
        dim_x
        pooling_mat;
    end
    
    methods
        function obj = ipooling(p_1, p_2)
            obj.dim_pool = [p_1, p_2];
        end
        
        function [obj, y] = forward(obj, x)
            d_x = size(x);
            obj.dim_x = d_x;
            d_pool = obj.dim_pool; % acceleration
            o_h = ceil(d_x(1) / d_pool(1));
            o_w = ceil(d_x(2) / d_pool(2));
            
            pooling_mat_tmp = zeros(o_h*d_pool(1), o_w*d_pool(2), d_x(3), d_x(4));
            y = zeros(o_h, o_w, d_x(3), d_x(4));
            
            % maxpooling
            for i = 1:d_x(4)
                for i_h = 1:o_h
                    for i_w = 1:o_w
                        [tmp1, tmp2] = max(x((i_h-1)*d_pool(1)+1:min(i_h*d_pool(2), end), ...
                            (i_w-1)*d_pool(2)+1:min(i_w*d_pool(2), end), :, i));
                        [tmp3, tmp4] = max(tmp1);
                        for i_d = 1:length(tmp3)
                            pooling_mat_tmp((i_h-1)*d_pool(1)+tmp2(1, tmp4(i_d), i_d), (i_w-1)*d_pool(2)+tmp4(i_d), i_d, i) = 1;
                            y(i_h, i_w, i_d, i) = tmp3(i_d);
                        end
                    end
                end
            end
            obj.pooling_mat = pooling_mat_tmp;
        end
        
        function [delta_x] = backward(obj, delta_y)
            d_pool = obj.dim_pool;
            d_x = obj.dim_x;
            dim_y = size(delta_y);
            tmp = ones(1, 1, 1, 1, d_pool(1), d_pool(2));
            delta_x_pad = reshape(permute(delta_y .* tmp, [5, 1, 6, 2, 3, 4]), ...
                dim_y(1)*d_pool(1), dim_y(2)*d_pool(2), dim_y(3), dim_y(4)) .* obj.pooling_mat;
            delta_x = delta_x_pad(1:d_x(1), 1:d_x(2), :, :);
        end
    end
end
