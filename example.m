mnist = data_reader('data');
p = parpool;

pos = 1;
[images, labels, pos] = mnist.train.next_batch(64, pos);
images = gpuArray(images);
labels = gpuArray(labels);

% convolution layer
conv1 = iconv2d(5, 5, 1, 6, 'tanh', 1, 'same');
y = conv1.forward(images);

% dropout layer
dropout_layer = dropout(0.5);
[dropout_layer, y] = dropout_layer.forward(x);
delta_x = dropout_layer.backward(delta_y);

% softmax layer
softmax_layer = softmax();
y = softmax_layer.forward(x);
delta_x = softmax_layer.backward(y, labels, 0.01);

