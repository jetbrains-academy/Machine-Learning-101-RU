####Information Gain

The information gain is based on the decrease in entropy after a dataset is split on an attribute.
Constructing a decision tree is all about finding attribute that returns the highest information gain.

$$IGain = H(parent) - H(children) $$

We simply subtract the entropy of `Y` given `X` from the entropy of just `Y` to calculate the reduction of uncertainty about
`Y` given an additional piece of information `X` about `Y`.

Constructing a decision tree is all about finding attribute that returns the highest information gain
(i.e., the most homogeneous branches).

### Задание

Реализуйте метод `information_gain`, который принимает выборку, разделяет ее на 2 независимые подвыборки и
подсчитывает information gain.
