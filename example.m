mnist = data_reader('data');
p = parpool;

pos = 1;
[images, labels, pos] = mnist.train.next_batch(64, pos);
images = gpuArray(images);

% convolution layer
conv1 = iconv2d(5, 5, 1, 6, 'tanh', 1, 'same');
y = conv1.forward(images);

% dropout layer
drop1 = dropout(0.5);

[drop1, y] = drop1.forward(x);
delta_x = drop1.backward(delta_y);

disp(size(y));
