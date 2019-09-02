Now we need to transform the probability we want to calculate into something that can be calculated using word frequencies.
For this, we will use some basic properties of probabilities, and <a href="https://en.wikipedia.org/wiki/Bayes%27_theorem">Bayes’ Theorem</a>.

Bayes’ Theorem is useful when working with conditional probabilities (like we are doing here), because it provides us with a way to reverse them:

$$P(A|B) = \frac{P(B|A) \times P(A)}{P(B)}$$

In our case, we have $P(spam | sentence)$, so using this theorem we can reverse the conditional probability:

$$P(spam|sentence) = \frac{P(sentence|spam) \times P(spam)}{P(sentence)}$$

For our classifier we’re just trying to find out which tag has a bigger probability,
we can discard the divisor — which is the same for both tags — and just compare:

$$P(sentence|spam) \times P(spam)$$

and

$$P(sentence|ham) \times P(ham)$$

### Naive

Naive Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature.
Hence it is called **naive**.

В нашем случае это означает, что вероятность встретить некоторое слово в сообщении не зависит от наличия других слов в этом сообщении.

$$P(\text{Who let the dogs out}) = P(\text{Who}) \times P(\text{let}) \times P(\text{the}) \times P(\text{dogs}) \times P(\text{out})$$

Calculating a probability is just counting in our training data.


### Задание

Реализуйте метод `fit`, который по переданной выборке вычисляет и сохраняет следующие параметры, которые понадобятся на этапе классификации:

  -  `classes_prior` -- оценка априорной вероятности классов в виде numpy вектора длины 2 (количество классов)

  $$P (\text{spam}) = \frac{\text{Num documents that have been classified as spam}}{\text{Num documents}}$$

  - `likelihood` -- относительные частоты слов для каждого класса в виде вектора numpy размерности (`2 x размер словаря`)

  - `classes_words_count` -- суммарное количество слов для сообщений каждого класса в виде вектора длины 2
