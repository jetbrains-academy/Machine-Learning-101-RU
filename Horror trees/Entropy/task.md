A decision tree is built top-down from a root node and involves partitioning the data into subsets that contain instances
with similar values (homogeneous).

Algorithm uses **entropy** to calculate the homogeneity of a sample. Entropy is measured between 0 and 1.
If the sample is completely homogeneous the entropy is zero and if the sample is equally divided then it has entropy of one.

The mathematical formula for Entropy is as follows:

$$H = - \sum\limits_{i=1}^{C} p_i \log_2 p_i$$

where $p_i$ -- the frequentist probability of an element/class $i$ in our data


### Задание

Реализуйте функцию `entropy`, вычисляющую энтропию для некоторого подмножества объектов. На вход функция получает массив
меток объектов.
