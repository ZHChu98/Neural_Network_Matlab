%% prework
mnist = data_reader('data');

%% parameters
lr = 1e-4;
batch_size = 128;
train_step = 6000;
n_display = 500;
weight_decay = 0.01;

%% model 2fc
%    
%       y_fc2   # # # # # # #      
%              /             \  fully connected layer
%  y_fc1_drop  # # # # # # # #     
%              |             |  dropout layer drop_prob=0.3
%       y_fc1  # # # # # # # #     
%             /               \  fully connected layer
%          x  # # # # # # # # #    
flatten_layer = iflatten();
fully_layer1 = ifc(28*28*1, 128);
dropout_layer = idropout(0.3);
fully_layer2 = ifc(128, 10);
softmax_layer = isoftmax();

%% train
test_images = mnist.test.images;
test_labels = mnist.test.labels;

for step = 1:train_step
    [train_images, train_labels] = mnist.train.next_batch(batch_size);

    y_flatten = flatten_layer.forward(train_images);
    y_fc1 = fully_layer1.forward(y_flatten);
    y_drop = dropout_layer.forward(y_fc1);
    y_fc2 = fully_layer2.forward(y_drop);
    y = softmax_layer.forward(y_fc2);

    delta_y_fc2 = softmax_layer.backward(y, train_labels, weight_decay);
    delta_y_fc1 = fully_layer2.backward(delta_y_fc2, lr);
    delta_y_drop = dropout_layer.backward(delta_y_fc1);
    [~] = fully_layer1.backward(delta_y_drop, lr);
    
    if rem(step, n_display) == 0
        train_loss = iCrossEntropyLoss(y, train_labels, weight_decay);
        train_accuracy = iAccuracy(y, train_labels);
        fprintf('<train step: %5d> train_accuracy: %.6f, train_loss: %.6f\n', ...
            step, train_accuracy, train_loss);
        y_flatten = flatten_layer.forward(test_images);
        y_fc1 = fully_layer1.forward(y_flatten);
        y_fc2 = fully_layer2.forward(y_fc1);
        y = softmax_layer.forward(y_fc2);
        test_loss = iCrossEntropyLoss(y, test_labels, weight_decay);
        test_accuracy = iAccuracy(y, test_labels);
        fprintf('                    test_accuracy:  %.6f, test_loss:  %.6f\n', ...
            test_accuracy, test_loss);
    end
end
