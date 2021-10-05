# **NEURAL NETWORK IN MATLAB**

Project for Optimization

## **Brief Description**

In this project, our goal is to implement mathematically in MATLAB the forward pass and backpropagation for each layer in the neural network, such as linear layer, convolution layer, softmax layer, etc.

## **Usage**

There are two models available for test. You can run in the MATLAB console to test: `icnn` or `icnn2`

Parameters could be changed easily by editing icnn.m or icnn2.m.

## **Project Structure**

Generally, training a model comsists of five steps as following:

1. [Read the datasets and initialize hyperparameters](#step-1)
2. [Build the model one layer after another and initialize each layer weights](#step-2)
3. [Forward the data through the model by calling forward method of each layer and passing the result to the next](#step-3)
4. [Backpropagating the gradient through the model by calling backward method of each layer](#step-4)
5. [Evaluate model performance and tune the model](#step-5)

### **Step 1**

To read the datasets stored in the folder `data`, you can run:

`mnist = data_reader('data');`

To obtain all the train images, you can run:

`train_images = mnist.train.images;`

To obtain a batch of train datasets, you can run:

`[train_images, train_labels] = mnist.train.next_batch(batch_size);`

hyperparameters are decribed as:

- lr (float): learning rate  
- batch_size (int): input batch size  
- train_step (int): total number of train batch  
- n_display (int): period for evaluating performance on test datasets  
- weight_decay (float): parameter for L2 regularization  

### **Step 2**

Initializing all layers which we need with parameters. Details about each kind of neural network layer are available in the [documentation](#documentation).

Note: please pay attention to the size of each layer's input and output and check whether they correspond to the previous or following layers.

### **Step 3**

Generally, for each training step, we read a batch of images and labels. Then, we feed the model in the forward direction by calling `forward` method of each layer class, which is explained in detail in the [documentation](#documentation).

Note: please pay attention to the name of each layer's input and output as well as layer itself. One useful trick is clear workspace each time you finish training.

### **Step 4**

Backpropagating should correspond to the model created by Part 3. Details about backpropagation are shown in the [documentation](#documentation).

Note: please pay attention again to the name of each layer's input and output as well as layer itself, because it is sometimes hard to distinguish the variable that we want among dozens of variable names.

### **Step 5**

When the `train_step` could be divided by `n_display`, we show the model performance on test dataset. Given the output of the model and gold labels, `iEvaluation` function will calculate automatically the accuracy, loss and F1 score.

****

## **Documentation**

### **iconv2d**

**CLASS** `iconv2d(kernel_height, kernel_width, in_channels, out_channels, activation, padding)`

Applies a 2D convolution and an activation function over an input signal composed of several input planes.

In the simplest case, the input with size (H_in, W_in, C_in, N) correspond to the output with size (H_out, W_out, C_out, N), where N is a batch size, C denotes the number of channels, H is a height of planes in pixels, and W is width in pixels.

Parameters

- `kernel_height` (int): height of the convolution kernel  
- `kernel_width` (int): width of the convolution kernel
- `in_channels` (int): number of channels in the input image
- `out_channels` (int): number of channels produced by the convolution
- `activation` (string): mode of activation function. Possible options for activation: 'sigmoid', 'tanh', 'relu'  
- `padding` (string): mode of zero-paddings, possible options for padding: 'same', 'valid', where 'same' denotes input size and output size are the same and 'valid' denotes no zero-paddings  

Methods

- `forward` (input)$\rightarrow$output
- `backward` (delta_output, learning_rate)$\rightarrow$delta_input

Examples

```MATLAB
>> % constructor with kernel size = [5, 5]
>> conv_layer = iconv2d(5, 5, 6, 12, 'tanh', 'same');
>> % forward propagation
>> x = randn(10, 10, 6, 64); 
>> y = conv_layer.forward(x);
>> % backpropagation & learning rate = 0.01
>> delta_y = randn(10, 10, 12, 64); 
>> delta_x = conv_layer.backward(y, 0.01);
```

### **idropout**

**CLASS** `idropout(drop_prob)`

During training, randomly zeros some of the elements of the input with the probability `drop_prob` using samples from a Bernoulli distribution. Each channel will be zeroed out independently on every forward call. This has proven to be an effective technique for regularization and preventing the coadaptation of neurons.

Parameters

- `drop_prob` (float): the probability of an element to be zeroed.

Methods

- `forward` (input)$\rightarrow$output
- `backward` (delta_output)$\rightarrow$delta_input

Examples

```MATLAB
>> drop_layer = idropout(0.5);
>> % forward propagation
>> x = randn(5, 5, 4, 16);
>> y = drop_layer.forward(x);
>> % backpropagation
>> delta_y = randn(5, 5, 4, 16);
>> delta_x = drop_layer.backward(delta_y);
```

### **iEvaluation**

**FUNCTION** `iEvaluation` (output, labels, weight_decay)$\rightarrow$[accuracy, f1, loss]

Applies Cross Entropy Function with L2 regularization to get average loss. Generates a confusion matrix to calculate accuracy and f1 macro score. This function does not have any change on model's weights.

Parameters

- `output` (2D float array): output of the model with size [batch_size, n_class]
- `labels` (int array): correct labels attached to the inputs
- `weight_decay` (float): parameter for L2 regularization

Examples

```MATLAB
>> % batch_size = 64, n_class = 6
>> output = rand(64, 6);
>> labels = unidrnd(6, 64, 1);
>> % weight_decay = 0.01
>> [accuracy, f1, loss] = iEvaluation(output, labels, 0.01);
```

### **ifc**

**CLASS** `ifc(in_features, out_features)`

Applies a linear transformation to the incoming data

Parameters

- `in_features` (int): size of each input sample
- `out_features` (int): size of each output sample

Methods:

- `forward` (input)$\rightarrow$output
- `backward` (delta_output, learning_rate)$\rightarrow$delta_input

Examples

```MATLAB
>> % constructor
>> fc_layer = ifc(16, 10);
>> % forward propagation
>> x = randn(64, 16);
>> y = fc_layer.forward(x);
>> % backpropagation with learning_rate = 0.01
>> delta_y = randn(64, 10);
>> delta_x = fc_layer.backward(delta_y, 0.01);
```

### **iflatten**

**CLASS** `iflatten()`

Flatten the 4D array to 2D array with size [batch_size, n_features].

Methods

- `forward` (input)$\rightarrow$output
- `backward` (delta_output)$\rightarrow$delta_input

Examples

```MATLAB
>> flatten_layer = iflatten();
>> % forward propagation
>> x = randn(4, 4, 5, 64);
>> y = flatten_layer.forward(x);
>> % backpropagation
>> delta_y = randn(64, 4*4*5);
>> delta_x = flatten_layer.backward(delta_y);
```

### **ipooling**

**CLASS** `ipooling(kernel_height, kernel_width)`

Applies a 2D max pooling over an input signal composed of input planes.

In the simplest case, the input with size (H_in, W_in, C, N) correspond to the output with size (H_out, W_out, C, N), where N is a batch size, C denotes the number of channels, H is a height of planes in pixels, and W is width in pixels. The relation between input size and the output size should be:

$$H_{out} = ceil(H_{in} / kernel\_height)$$

$$W_{out} = ceil(W_{in} / kernel\_width)$$

Note (hyperparameters):
stride is set to [kernel_height, kernel_width] as default and no zero-paddings. Mode is set to max pooling as default.

Parameters

- `kernel_height` (int): height of the max pooling kernel
- `kernel_width` (int): width of the max pooling kernel

Methods

- `forward` (input)$\rightarrow$output
- `backward` (delta_output)$\rightarrow$delta_input

Examples

```MATLAB
>> % constructor
>> pooling_layer = ipooling(2, 2);
>> % forward propagation
>> x = randn(12, 12, 5, 24);
>> y = pooling_layer.forward(x);
>> % backpropagation
>> delta_y = randn(6, 6, 5, 24);
>> delta_x = pooling_layer.backward(delta_y);
```

### **isoftmax**

**CLASS** `isoftmax()`

Applies the Softmax function to an 2D float array input rescaling them so that the elements of the output lie in the range [0, 1] and sum up to 1.

Note: the `backward` function calculates errors including softmax layer and Cross Entropy layer with L2 regularization. So, softmax layer's backward function is the first function called in backpropagation.

Methods

- `forward` (input)$\rightarrow$output
- `backward` (delta_output, labels, weight_decay)$\rightarrow$delta_input

Examples

```MATLAB
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
