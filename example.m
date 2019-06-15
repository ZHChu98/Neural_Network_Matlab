%% reading data
mnist = data_reader('data');
[images, labels] = mnist.train.next_batch(64);

%% convolution layer
% constructor with kernel size = [5, 5]
conv_layer = iconv2d(5, 5, 6, 12, 'tanh', 'same');
% forward propagation
x = randn(10, 10, 6, 64); 
y = conv_layer.forward(x);
% backpropagation & learning rate = 0.01
delta_y = randn(10, 10, 12, 64); 
delta_x = conv_layer.backward(y, 0.01);

%% flatten layer
flatten_layer = iflatten();
% forward propagation
x = randn(4, 4, 5, 64);
y = flatten_layer.forward(x);
% backpropagation
delta_y = randn(64, 4*4*5);
delta_x = flatten_layer.backward(delta_y);

%% fully connected layer
% constructor
fc_layer = ifc(16, 10);
% forward propagation
x = randn(64, 16);
y = fc_layer.forward(x);
% backpropagation with learning_rate = 0.01
delta_y = randn(64, 10);
delta_x = fc_layer.backward(delta_y, 0.01);

%% max-pooling layer
% constructor
pooling_layer = ipooling(2, 2);
% forward propagation
x = randn(12, 12, 5, 24);
y = pooling_layer.forward(x);
% backpropagation
delta_y = randn(6, 6, 5, 24);
delta_x = pooling_layer.backward(delta_y);

%% dropout layer
drop_layer = idropout(0.5);
% forward propagation
x = randn(5, 5, 4, 16);
y = drop_layer.forward(x);
% backpropagation
delta_y = randn(5, 5, 4, 16);
delta_x = drop_layer.backward(delta_y);

%% softmax layer
% constructor
softmax_layer = isoftmax();
% forward propagation
x = randn(64, 10);
y = softmax_layer.forward(x);
% backpropagation with weight_decay = 0.01
delta_y = randn(64, 10);
labels = unidrnd(10, 64, 1);
delta_x = softmax_layer.backward(delta_y, labels, 0.01);

%% evaluation
% batch_size = 64, n_class = 6
output = rand(64, 6);
labels = unidrnd(6, 64, 1);
% weight_decay = 0.01
[accuracy, f1, loss] = iEvaluation(output, labels, 0.01);

% evalin('base', 'who')