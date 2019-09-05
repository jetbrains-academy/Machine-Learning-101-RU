Stochastic gradient descent (SGD) computes the gradient for each update using a single training data point $x_i$ (chosen at random).
The idea is that the gradient calculated this way is a stochastic approximation to the gradient calculated using the entire training data.
Each update is now much faster to calculate than in batch gradient descent, and over many updates,
we will head in the same general direction.

In Stochastic mini-batch gradient descent, we calculate the gradient for each small mini-batch of training data.
That is, we first divide the training data into small batches (say k samples per batch).
We perform one update per mini-batch. $k$ is usually in the range 30–500, depending on the problem.

### Задание

Реализуйте метод стохастического градиентного спуска для обучения линейного классификатора.
﻿

Как и в случае с `GradientDescent` метод `fit` должен
возвращать значения $Q$ на каждой итерации.
Чтобы оценить $Q$ на каждой итерации воспользуйтесь формулой
$$Q = (1 − \eta)Q + \eta L_i$$

Значение $\eta \in [0, 1]$,
используемое для вычисления оценки $Q$, можно выбрать любое, например,
`1 / len(X)`. Так как величина $Q$ не стабильна, использовать
её для определения сходимости не следует.

Вместо этого предлагается использовать "стратегию оптимиста": сделать ровно `n_iter`
итераций и надеяться, что за это время стохастический градиентный спуск
сойдётся.

Параметр `k` определяет размер случайной подвыборки из
`X`, используемой для вычисления градиента.