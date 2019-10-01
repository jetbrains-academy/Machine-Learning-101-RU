### ID3

The core algorithm for building decision trees is called ID3.
The ID3 algorithm is run recursively on the non-leaf branches, until all data is classified.

1. Calculate entropy of the target.
2. The dataset is then split on the different attributes. The entropy for each branch is calculated.
Then it is added proportionally, to get total entropy for the split. The resulting entropy is subtracted from the entropy
before the split. The result is the Information Gain, or decrease in entropy.
3. Choose attribute with the largest information gain as the decision node, divide the dataset by its branches and
repeat the same process on every branch.
4. - A branch with entropy of 0 is a leaf node.
   - A branch with entropy more than 0 needs further splitting.


### Задание

Реализуйте рекурсивный алгоритм построения дерева решения в методе `build` класса `DecisionTree`.
Для этого:

1. Постройте все возможные предикаты для конкретного признака.
 Для этого нужно определить уникальные значения данного признака.
2. Оцените information gain всех возможных предикатов для всех признаков на основании entropy.
3. Выберите предикат, который обеспечивает наилучшее с точки зрения информативности разбиение.
4. Рекурсивно постройте правое и левое поддеревья.
5. Метод должен возвращать `self`
