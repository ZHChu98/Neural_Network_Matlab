%%
mnist = data_reader('data');
[images, labels] = mnist.train.next_batch(64);

%% introduction to each layer function
%% convolution layer
conv_layer = iconv2d(5, 5, 1, 6, 'tanh', 'same');
y = conv_layer.forward(x);
delta_x = conv_layer.backward(delta_y, learning_rate);

%% flatten layer
flatten_layer = iflatten();
y = flatten_layer.forward(x);
delta_x = flatten_layer.backward(delta_y)

%% fully connected layer
fully_layer = ifc(128, 10);
y = fully_layer.forward(x);
delta_x = fully_layer.backward(delta_y, learning_rate);

%% max-pooling layer
pooling_layer = ipooling(2, 2);
y = pooling_layer.forward(x)
delta_x = pooling_layer.backward(delta_y);

%% dropout layer
dropout_layer = idropout(0.5);
y = dropout_layer.forward(x);
delta_x = dropout_layer.backward(delta_y);

%% softmax layer
softmax_layer = isoftmax();
y = softmax_layer.forward(x);
delta_x = softmax_layer.backward(y, labels, 0.01);

%% evaluation
[accuracy, f1_score, loss] = iEvaluation(y, labels, 0.01);

% evalin('base', 'who')