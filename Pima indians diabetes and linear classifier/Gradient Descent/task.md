Gradient Descent is an optimization algorithm that helps machine learning models converge at a minimum value through repeated steps.

Loss function measures how bad our model performs compared to actual occurrences.
Hence, it only makes sense that we should reduce this loss.

The gradient (or derivative) tells us the incline or slope of the loss function.
Hence, to minimize the loss function, we move in the direction opposite to the gradient.

A simple gradient Descent Algorithm is as follows:
- Initialize the weights $\vec{w}$ randomly.

- Calculate the cost function $$Q(\vec{w}) = \sum\limits_{i \in \text{training set}} L(M_i(\vec{w}))$$
where $M_i(\vec{w}) = \langle \vec{w}, \vec{x}_i\rangle y_i$

- Calculate the gradients $\bigtriangledown Q(\vec{w})$ of cost function.
$\bigtriangledown Q(\vec{w}) = (\frac{\partial Q(\vec{w})}{\partial w_j} )_{j=0}^n
= \sum \mathcal{L}'(\langle \vec{w}, \vec{x}_i \rangle y_i) \vec{x}_i y_i$

- Update the weights by an amount proportional to G, i.e.
$$\vec{w} = \vec{w} - \alpha \bigtriangledown Q(\vec{w})$$
- Repeat until the cost $Q(w)$ stops reducing, or some other pre-defined termination criteria is met.

In step 3, $\alpha$ is the learning rate which determines the size of the steps we take to reach a minimum.
We need to be very careful about this parameter. High values of $\alpha$ may overshoot the minimum, and very low values will
reach the minimum very slowly.


### Задание

Реализуйте метод градиентного спуска для обучения линейного классификатора в виде класса `GradientDescent`.

Метод fit должен вернуть список значений функционала качества $Q(w)$ на каждой итерации градиентного спуска.

В качестве критерия остановки следует использовать отсечку `threshold` на расстояние между векторами весов на текущей и
предыдущей итерациях. Расстояние можно выбрать любое, например, Евклидово или `l1`.

