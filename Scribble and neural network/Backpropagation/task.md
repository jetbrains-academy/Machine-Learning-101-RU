### Loss

Before we train our network, we first need a way to quantify how “good” it’s doing so that it can try to do “better”. That’s what the loss is.
We’ll use the mean squared error (MSE) loss:

$$MSE = \sum\limits_{i = 1}^n (y_{true} - y_{predicted})^2$$

- $y$ represents the variable being predicted
- $y_{true​}$ is the true value of the variable (the “correct answer”)
- $y_{pred}$​ is the predicted value of the variable. It’s whatever our network outputs.
- $n$ is the number of samples

Now that we have the loss function, our goal is to get it as close as we can to `0`.
As we are training our network, all we are doing is minimizing the loss.

From the previous lesson we know that to figure out which direction to alter our weights, we need to find the rate of change
of our loss with respect to our weights.
In other words, we need to use the derivative of the loss function to understand how the weights affect the input.

Here’s how we will calculate the incremental change to our weights:

Find the margin of error of the output layer (o) by taking the difference of the predicted output and the actual output (y)

Apply the derivative of our sigmoid activation function to the output layer error. We call this result the delta output sum.

Use the delta output sum of the output layer error to figure out how much our z2 (hidden) layer contributed to the output error by performing a dot product with our second weight matrix. We can call this the z2 error.

Calculate the delta output sum for the z2 layer by applying the derivative of our sigmoid activation function (just like step 2).

Adjust the weights for the first layer by performing a dot product of the input layer with the hidden (z2) delta output sum. For the second layer, perform a dot product of the hidden(z2) layer and the output (o) delta output sum.

### Задание

Реализуйте обучение нейронной сети с помощью метода обратного распространения ошибки (Backpropagation).
Функция `backpropagation` должна возвращать производные, необходимые для обновления массива весов.

При реализации метода `train` может быть полезно обратиться к реализации метода стохастического градиента из предыдущего урока.