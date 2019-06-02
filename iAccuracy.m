function [accuracy] = iAccuracy(y, labels)
    [~, y_pred] = max(y, [], 2);
    accuracy = sum(y_pred == labels) / length(labels);
end

