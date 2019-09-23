The iris data is the most commonly used data set for testing machine learning algorithms.
The data contains four features — sepal length, sepal width, petal length,
and petal width for the different species (versicolor, virginica and setosa) of the flower, iris.
Also, for each species there are 50 instances (rows of data).

We will build a classification model on this data using neural network.
For simplicity we'll use only ‘Versicolor’ and ‘Virginica’ species and ‘petal length’ and ‘petal width’ as the features.

### Neuron

The basic unit of a neural network is a neuron.
A neuron takes inputs, does some math with them, and produces one output.

![Neuron scheme](neuron-scheme.png)

First, neuron adds up the value of every input. On the figure, there are $n$ inputs ($x^1, x^2, \dots x^n$ ) coming to the neuron.

This value is multiplied, before being added, by another variable called “weight” ($w_1, w_2, \dots w_n$).
Weights are the only values that will be modified during the learning process. A bias value $b$ may be added to the total value calculated.

After all those summations, the neuron finally applies a function called “activation function” $\sigma$ to the obtained value.
Activation function usually serves to turn the total value calculated before to a number between `0` and `1`.

A commonly used activation function is the [sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function) function.

The mathematical formula for the neuron output is as follows:

$$a = \sigma(\sum\limits_{j=1}^n w_j x^j + b)$$

### Neural network

A neural network is nothing more than a bunch of neurons connected together.

Neural Networks consist of the following components:
- An input layer
- An arbitrary amount of hidden layers
- An output layer
- A set of weights and biases between each layer, $W$
- A choice of activation function for each hidden layer $\sigma$

Here is 2-layer Neural Network. The input layer is excluded when counting the number of layers in a Neural Network.
![Neuralnet](neuralnet.png)


### Training the Neural Network
The output $\hat{y}$ of a simple 2-layer Neural Network is:
$$\hat{y} = \sigma(W_2 \sigma(W_1x + b_1) + b_2$$

Weights $W$ and the biases $b$ are the only variables that affects the output \hat{y}.
The right values for the weights and biases determines the strength of the predictions.
The process of fine-tuning the weights and biases from the input data is known as training the Neural Network.

Each iteration of the training process consists of the following steps:
- Calculating the predicted output, known as feedforward
- Updating the weights and biases, known as backpropagation

### Feedforward

Feedforward is just simple calculus and for a basic 2-layer neural network,
the output of the Neural Network is:

$$\hat{y} = \sigma(W_2 \sigma(W_1x + b_1) + b_2$$

### Задание

Implement `feedforward` function. Let's assume the biases to be 0 for simplicity.

