classdef softmax
    methods
        function [y] = forward(~, x)
            exp_x = exp(x);
            y_sum = sum(exp_x, 2);
            y = exp_x .* y_sum .^ (-1);
        end
        
        function [delt_x] = backward(~, y, labels, weight_decay) % attention, y ~= y_pred
            % we admit that we use cross entropy as loss function
            delt_x = (1 + weight_decay) * y;
            for i = 1:length(lables)
                delt_x(i, labels(i)) = delt_x(i, labels(i)) - 1;
            end
        end
    end
end

