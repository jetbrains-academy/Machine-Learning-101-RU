Since we have a random set of weights, we need to alter them to make our inputs equal to the corresponding outputs from our data set.
This is done through a method called backpropagation.

Backpropagation works by using a loss function to calculate how far the network was from the target output.

### Loss

One way of representing the loss function is by using the sum squared loss function:

$$Loss(y, \hat{y}) = \sum\limits_{i=1}^{n} (y_i - \hat{y}_i) ^ 2$$

where $\hat{y}$ - predicted output

$y$ - actual output

Sum-of-squares error is simply the sum of the difference between each predicted value and the actual value.
The difference is squared so that we measure the absolute value of the difference.

Our goal in training is to find the best set of weights and biases that minimizes the loss function.

In order to know the appropriate amount to adjust the weights and biases by, we need to know the derivative of the loss
function with respect to the weights and biases.

Recall from the previous lesson (Gradient Descent) that the derivative of a function is simply the slope of the function.
If we have the derivative, we can simply update the weights and biases by increasing/reducing with it.

However, we can’t directly calculate the derivative of the loss function with respect to the weights and biases because
the equation of the loss function does not contain the weights and biases. Therefore, we need the chain rule to help us calculate it.

$$\frac {\partial Loss(y, \hat{y})}{\partial W} =  \frac { \partial Loss(y, \hat{y} ) } {\partial \hat{y}}
\frac { \partial \hat{y} } {\partial z} \frac { \partial z } {\partial W} $$

$$= 2 (y - \hat{y} ) * z (1- z) * x$$

where $z = Wx + b$

Here’s how we will calculate the incremental change to our weights:

- Find the margin of error of the output layer by taking the difference of the predicted output and the actual output

- Apply the derivative of our sigmoid activation function to the output layer error. Let's call this result the delta output sum.

- Use the delta output sum of the output layer error to figure out how much our hidden layer contributed to the output error by
performing a dot product with our second weight matrix. We can call this the layer1 error.

- Calculate the delta output sum for the layer1 by applying the derivative of our sigmoid activation function.

- Adjust the weights for the first layer by performing a dot product of the input layer with the hidden delta output sum.
For the second layer, perform a dot product of the hidden layer and the output delta output sum.

### Задание

Implement backward propagation function that does everything specified in the four steps above.