function [accuracy, f1_score, loss] = iEvaluation(y, labels, weight_decay)
    n_class = 10;
    loss = 0.0;
    for i = 1:length(labels)
        loss = loss - log(y(i, labels(i)));
    end
    loss = loss + weight_decay / 2 * sum(sum(y .^ 2));
    loss = loss / length(labels);

    [~, y_pred] = max(y, [], 2);
    confusion_matrix = zeros(n_class, n_class);
    for i = 1:length(labels)
        confusion_matrix(y_pred(i), labels(i)) = confusion_matrix(y_pred(i), labels(i)) + 1;
    end
    accuracy = trace(confusion_matrix) / length(labels);
    TP = diag(confusion_matrix) + 1e-5;
    FP = sum(confusion_matrix, 2) - TP;
    FN = sum(confusion_matrix, 1)' - TP;
    P = TP ./ (TP+FP);
    R = TP ./ (TP+FN);
    f1_score = mean(2 ./ (1./P+1./R));
end