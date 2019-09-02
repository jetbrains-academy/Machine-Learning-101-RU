
The problem with the naive approach is following:

Если какое-то слово не встречалось в тренировочной выборке класса Spam, то его вероятность $P(word | Spam) = 0$.


It is problematic when a frequency-based probability is zero, because it will wipe out all the information in the other probabilities.

A solution would be <a href="https://en.wikipedia.org/wiki/Laplace_smoothing">Laplace smoothing</a>,
which is a technique for smoothing categorical data.
A small-sample correction, or pseudo-count, will be incorporated in every probability estimate.

We add 1 to every count so it’s never zero. To balance this, we add the number of possible words to the divisor,
so the division will never be greater than 1.

### Задание

Обновите свою имплементацию метода `fit` так, чтобы использовать Laplace Smoothing.