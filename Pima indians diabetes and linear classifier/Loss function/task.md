A linear classifier makes a classification decision based on a linear
predictor function combining a set of weights with the feature vector.

If the input feature vector to the classifier is a real vector $\vec {x}$, then the output score is

$$y=f(\vec {w} \cdot \vec {x}) = f ( \sum_{j} w_{j} x_{j})$$

where $\vec {w}$ is a real vector of weights and $f$ is a function that converts the dot product of the two vectors
into the desired output.  The weight vector $\vec {w}$ is learned from a set of labeled training samples.
Often f is a threshold function, which maps all values of $ \vec {w} \cdot \vec {x}$ above a certain threshold to the first class and all other values to the second class.

For a two-class classification problem, one can visualize the operation of a linear classifier as splitting a high-dimensional
input space with a hyperplane: all points on one side of the hyperplane are classified as "yes", while the others are classified as "no".

The loss is the error in our predicted value of $\vec{w}$. Our goal is to minimize this error to obtain the most accurate value.

There are several options to calculate the loss. We'll use `log_loss` function and `sigmoid_loss` function.

Log loss function is defined as follows:

$$L(M) = \log_2(1 + e^{-M})$$

Sigmoid loss function is defined as follows:

$$L(M) = 2(1 + e^{M})^{-1}$$

### Задание

Реализуйте логарифмическую (`log_loss`) и сигмоидную (`sigmoid_loss`) функции потерь. Функция потерь должна принимать на
вход вектор и возвращать пару из вектора значений функции потерь и вектора её производных. Например, если бы
мы решили использовать степенную функцию потерь:

    def power_loss(M, n=5):
        return M ** n, n * (M ** (n - 1))
