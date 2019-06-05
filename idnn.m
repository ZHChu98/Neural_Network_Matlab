%% prework
mnist = data_reader('data');
p = parpool;

%% parameters
lr = 1e-4;
batch_size = 128;
train_step = 4000;
n_display = 100;
weight_decay = 0.01;

%% model 2fc
flatten_layer = iflatten();
fully_layer1 = ifc(28*28*1, 128);
fully_layer2 = ifc(128, 19);
softmax_layer = isoftmax();

test_images = gpuArray(mnist.test.images);
test_labels = gpuArray(mnist.test.labels);

for step = 1:train_step
    [train_images, train_labels] = mnist.train.next_batch(batch_size);
    train_images = gpuArray(train_images);
    train_labels = gpuArray(train_labels);

    [flatten_layer, y_flatten] = flatten_layer.forward(train_images);
    [fully_layer1, y_fc1] = fully_layer1.forward(y_flatten);
    [fully_layer2, y_fc2] = fully_layer2.forward(y_fc1);
    y = softmax_layer.forward(y_fc2);

    delta_y_fc2 = softmax_layer.backward(y, train_labels, weight_decay);
    [fully_layer2, delta_y_fc1] = fully_layer2.backward(delta_y_fc2, lr);
    [fully_layer1, ~] = fully_layer1.backward(delta_y_fc1, lr);
    
    if rem(step, n_display) == 0
        train_loss = iCrossEntropyLoss(y, train_labels, weight_decay);
        train_accuracy = iAccuracy(y, train_labels);
        fprintf('<train step: %5d> train_accuracy: %4g, train_loss: %4g\n', ...
            step, train_accuracy, train_loss);
        [flatten_layer, y_flatten] = flatten_layer.forward(test_images);
        [fully_layer1, y_fc1] = fully_layer1.forward(y_flatten);
        [fully_layer2, y_fc2] = fully_layer2.forward(y_fc1);
        y = softmax_layer.forward(y_fc2);
        test_loss = iCrossEntropyLoss(y, test_labels, weight_decay);
        test_accuracy = iAccuracy(y, test_labels);
        fprintf('                    test_accuray: %4g, test_loss: %4g\n', ...
            test_accuracy, test_loss);
    end
end

%% ending
delete(p);