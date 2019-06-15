# optimization_project
2018-2019 2nd semester

## INTRODUCTION  
There are already two models available for test. You can run this command in the console to test:<br>
```
icnn  
icnn2
```
Parameters are be changed efficiently by editing icnn.m or icnn2.m.<br>

Generally, each main code is composed of following five parts.<br>
* Part 1. Reading datasets and setting hyperparameters.<br>
* Part 2. Designing model and initializing each layer.<br>
* Part 3. Feeding model in forward direction.<br>
* Part 4. Backpropagating in backward direction.<br>
* Part 5. Evaluating model's performance.<br>

### Part 1
To read data with all datasets are stored in the folder 'data', you can run this command:<br>
`mnist = data_reader('data');`<br>
To obtain all the train images, you can run this command:<br>
`train_images = mnist.train.images;`<br>
To obtain a batch of train datasets, you can run this command:<br>
`[train_images, train_labels] = mnist.train.next_batch(batch_size);`<br>

hyperparameters:<br>
* lr (float) - learning rate  
* batch_size (int) - input batch size  
* train_step (int) - total number of train batch  
* n_display (int) - period for evaluating performance on test datasets  
* weight_decay (float) - parameter for L2 regularization  

### Part 2
Initializing all layers which we need with parameters. Details about each kind of neural network layer are available in the documentation below.<br>

Note:<br>
Please pay attention to the size of each layer's input and output and check whether they correspond to the previous or following layers.<br>

### Part 3  
Generally, for each training step, we read a batch of images and labels. Then, we feed the model in the forward direction by calling forward function of each layer class, which is explained in detail in the documentation below.<br>

Note:<br>
Please pay attention to the name of each layer's input and output as well as layer itself. One useful trick is clear workspace each time you finish training.<br>

### Part 4
Backpropagating should correspond to the model created by Part 3. Details about backpropagation are exposed in the documentation below.<br>

Note:<br>
Please pay attention again to the name of each layer's input and output as well as layer itself, because it is sometimes hard to distinguish the variable we want among dozens of variable names.<br>

### Part 5  
This part is usually used when the training step is multiple times of n_display. Given the output of the model and correct labels, iEvaluation function will calculate automatically the accuracy, loss and F1 score.<br>


****
## Documentation
### iconv2d 
<b>CLASS</b> iconv2d(kernel_height, kernel_width, in_channels, out_channels, activation, padding)<br>
Applies a 2D convolution and an activation function over an input signal composed of several input planes.<br>

In the simplest case, the input with size (H_in, W_in, C_in, N) correspond to the output with size (H_out, W_out, C_out, N), where N is a batch size, C denotes the number of channels, H is a height of planes in pixels, and W is width in pixels.<br>

Parameters<br>
* kernel_height (int) - Height of the convolution kernel  
* kernel_width (int) - Width of the convolution kernel  
* in_channels (int) - Number of channels in the input image  
* out_channels (int) - Number of channels produced by the convolution  
* activation (string) - mode of activation function. Possible options for activation: 'sigmoid', 'tanh', 'relu'  
* padding (string) - mode of zero-paddings. Possible options for padding: 'same', 'valid', where 'same' denotes input size and output size are the same and 'valid' denotes no zero-paddings  

Member Function<br>
* forward(input)->output
* backward(delta_output, learning_rate)->delta_input

Examples
```
>> % constructor with kernel size = [5, 5]
>> conv_layer = iconv2d(5, 5, 6, 12, 'tanh', 'same');
>> % forward propagation
>> x = randn(10, 10, 6, 64); 
>> y = conv_layer.forward(x);
>> % backpropagation & learning rate = 0.01
>> delta_y = randn(10, 10, 12, 64); 
>> delta_x = conv_layer.backward(y, 0.01);
```

### idropout
<b>CLASS</b> idropout(drop_prob)<br>

During training, randomly zeros some of the elements of the input with the probability drop_prob using samples from a Bernoulli distribution. Each channel will be zeroed out independently on every forward call. This has proven to be an effective technique for regularization and preventing the coadaptation of neurons.<br>

Parameters
* drop_prob (float) - the probability of an element to be zeroed.

Member Function
* forward(input)->output
* backward(delta_output)->delta_input

Examples
```
>> drop_layer = idropout(0.5);
>> % forward propagation
>> x = randn(5, 5, 4, 16);
>> y = drop_layer.forward(x);
>> % backpropagation
>> delta_y = randn(5, 5, 4, 16);
>> delta_x = drop_layer.backward(delta_y);
```

### iEvaluation
<b>FUNCTION</b> iEvaluation(output, labels, weight_decay)->[accuracy, f1, loss]<br>

Applies cross entropy function with L2 regularization to get average loss. Generates a confusion matrix to calculate accuracy and f1 macro score. This function does not have any change on model's parameters.<br>

Parameters
* output (2D float array) - Output of the model with size [batch_size, n_class]
* labels (int array) - Correct labels attached to the inputs.
* weight_decay (float) - Parameter for L2 regularization.

Examples
```
>> % batch_size = 64, n_class = 6
>> output = rand(64, 6);
>> labels = unidrnd(6, 64, 1);
>> % weight_decay = 0.01
>> [accuracy, f1, loss] = iEvaluation(output, labels, 0.01);
```

### ifc
<b>CLASS</b> ifc(in_features, out_features)<br>
Applies a linear transformation to the incoming data<br>

Parameters
* in_features (int) - size of each input sample
* out_features (int) - size of each output sample

Member Function
* forward(input)->output
* backward(delta_output, learning_rate)->delta_input

Examples
```
>> % constructor
>> fc_layer = ifc(16, 10);
>> % forward propagation
>> x = randn(64, 16);
>> y = fc_layer.forward(x);
>> % backpropagation with learning_rate = 0.01
>> delta_y = randn(64, 10);
>> delta_x = fc_layer.backward(delta_y, 0.01);
```

### iflatten
<b>CLASS</b> iflatten()<br>
Flatten the 4D array to 2D array with size [batch_size, n_features].<br>

Member Function
* forward(input)->output
* backward(delta_output)->delta_input

Examples
```
>> flatten_layer = iflatten();
>> % forward propagation
>> x = randn(4, 4, 5, 64);
>> y = flatten_layer.forward(x);
>> % backpropagation
>> delta_y = randn(64, 4*4*5);
>> delta_x = flatten_layer.backward(delta_y);
```

### ipooling
<b>CLASS</b> ipooling(kernel_height, kernel_width)<br>
Applies a 2D max pooling over an input signal composed of input planes.<br>

In the simplest case, the input with size (H_in, W_in, C, N) correspond to the output with size (H_out, W_out, C, N), where N is a batch size, C denotes the number of channels, H is a height of planes in pixels, and W is width in pixels. The relation between input size and the output size is that<br>
H_out = ceil(H_in / kernel_height)<br>
W_out = ceil(W_in / kernel_width)<br>

Note (hyperparameters):<br>
stride is set to [kernel_height, kernel_width] as default and no zero-paddings. Mode is set to max pooling as default.<br>

Parameters
* kernel_height (int) - Height of the max pooling kernel
* kernel_width (int) - Width of the max pooling kernel

Member Function
* forward(input)->output
* backward(delta_output)->delta_input

Examples
```
>> % constructor
>> pooling_layer = ipooling(2, 2);
>> % forward propagation
>> x = randn(12, 12, 5, 24);
>> y = pooling_layer.forward(x);
>> % backpropagation
>> delta_y = randn(6, 6, 5, 24);
>> delta_x = pooling_layer.backward(delta_y);
```

### isoftmax
<b>CLASS</b> isoftmax()<br>
Applies the Softmax function to an 2D float array input rescaling them so that the elements of the output lie in the range [0, 1] and sum to 1.<br>

Note:<br>
The backward function calculates errors including softmax layer and cross entropy layer with L2 regularization. So, softmax layer's backward function is the first function called in backpropagation.<br>

Member Function
* forward(input)->output
* backward(delta_output, labels, weight_decay)->delta_input

Examples
```
>> % constructor
>> softmax_layer = isoftmax();
>> % forward propagation
>> x = randn(64, 10);
>> y = softmax_layer.forward(x);
>> % backpropagation with weight_decay = 0.01
>> delta_y = randn(64, 10);
>> labels = unidrnd(10, 64, 1);
>> delta_x = softmax_layer.backward(delta_y, labels, 0.01);
```
