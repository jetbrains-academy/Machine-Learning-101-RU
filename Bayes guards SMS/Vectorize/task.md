In this lesson we're going to implement Naive Bayes Classifier to detect spam in SMS.

A Naive Bayes classifier is a probabilistic machine learning model that’s used for classification task.
This classifier is based on the <a href="https://en.wikipedia.org/wiki/Bayes%27_theorem">Bayes theorem</a>:

$$P(A|B) = \frac{P(B|A) \times P(A)}{P(B)}$$

Naive Bayes is a probabilistic model, which means that it calculates the probability of each tag for a given text,
and then output the tag with the highest one.

In our task we want to calculate the probability that the sentence is "Spam" and the probability that it’s "Ham".
Then, take the largest one.

$P(Spam | sentence)$ — the probability that the tag of a sentence is "Spam" given the particular sentence.

### Задание

В файле "spam.txt" находится датасет, содержащий размеченные сообщения. Первое слово строки это идентификатор класса spam или ham,
далее через табуляцию следует сообщение.

Прежде чем строить классификатор нужно привести данные в удобный для классификации формат.
Для этого воспользуемся стандартной моделью для текстов под названием
[Bag of words](https://en.wikipedia.org/wiki/Bag-of-words_model).

We need numerical features as input for our classifier. So an intuitive choice would be word frequencies, i.e.,
counting the occurrence of every word in the document ignoring word order and sentence construction.

Реализуйте функцию `vectorize`, принимающую на вход вектор
строк длины N и возвращающую пару из словаря (построенного на основе входных данных) и матрицы размера ($N$, $M$),
где $M$ -- размер словаря.
В качестве словаря будем использовать все слова, которые встречаются в переданном массиве.
В каждой строке матрицы на $j$-й позиции находится число $x$, которое означает, что $j$-е слово встретилось в сообщении $x$ раз.