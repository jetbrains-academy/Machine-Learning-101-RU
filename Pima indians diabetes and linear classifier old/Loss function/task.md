Linear regression is a linear approach to modelling the relationship between a dependent variable
and one or more independent variables.
Let `X` be the independent variable and `Y` be the dependent variable. We will define a linear relationship between these
two variables as follows:
$$Y = a X + b$$

This is the equation for a line where `a` is the slope of the line and `b` is the `y` intercept.

The training step of the linear classifier is to determine the value of `a` and `b` such that the line
corresponding to those values is the best fitting line or gives the minimum error.

The loss is the error in predicted value of `a` and `b`. The goal is to minimize this error to obtain the most accurate
value of `a` and `b`.

Linear Classifier use the Mean Squared Error function to calculate the loss.

$$ E = \frac{1}{n} \sum \limits_{i=0}^{n} (y_i - \hat{y}_i)^2 $$
Here $y_i$ is the actual value and $\hat{y}_i$ is the predicted value.

### Задание

Реализуйте логарифмическую (log_loss) и сигмоидную (sigmoid_loss) функции потерь. Функция потерь должна принимать на
вход вектор и возвращать пару из вектора значений функции потерь и вектора её производных. Например, если бы
мы решили использовать степенную функцию потерь:
    
    def power_loss(M, n=5):
        return M ** (-n), -n * (M ** (-n - 1))
