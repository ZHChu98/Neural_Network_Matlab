function loss = iCrossEntropyLoss(y, labels, weight_decay) % L2 regularization
    loss = 0.0;
    for i = 1:length(labels)
        loss = loss - log(y(i, labels(i)));
    end
    loss = loss + weight_decay / 2 * sum(sum(y .^ 2));
    loss = loss / length(labels);
end

