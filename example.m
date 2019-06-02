mnist = data_reader('data');
p = parpool;
[images, labels] = mnist.train.next_batch(64);
images = gpuArray(images);
conv1 = iconv2d(5, 5, 1, 6, 'tanh', 1, 'same');
y = conv1.forward(images);
disp(size(y));
