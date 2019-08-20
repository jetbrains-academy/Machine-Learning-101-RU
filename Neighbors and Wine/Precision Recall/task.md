
Самое время оценить качество получившегося классификатора в
терминах точности (*precision*) и полноты (*recall*). Формально эти
метрики определяются следующим образом:
$$
\operatorname{Precision} = \frac{TP}{TP + FP} \\\\
\operatorname{Recall} = \frac{TP}{TP + FN}
$$

Здесь
- $TP$ --- это количество элементов, которые классификатор **верно** отнёс к классу $c$,
- $FP$ --- количество элементов, которые классификатор **неверно** отнёс к классу $c$,
- $FN$ --- количество элементов, которые классификатор **неверно** отнёс к классу, **отличному** от $c$.


### Задание

Реализуйте функцию, вычисляющую для каждого класса точность и полноту по полученным от `knn` предсказаниям.
Функция должна возвращать список, состоящий из троек (class, precision, recall).

Функция может выглядеть следующим образом:

    # Обозначим за y_pred результат работы k-ближайших соседей на тестовой выборке X_test.
    
    y_pred = knn(X_train, y_train, X_test, k, dist)

    def precision_recall(y_pred, y_test):
        class_precision_recall = []
        for c in range(n_classes):
            # ...
            
        return class_precision_recall

<div class="hint">
Значение <code>n_classes</code> можно вычислить по <code>y_test</code>
<pre>
<code>
    n_classes = len(set(y_test))
</code>
</pre>
или
<pre>
<code>
    import numpy as np
    n_classes = len(np.unique(y_test))
</code>
</pre>
</div>