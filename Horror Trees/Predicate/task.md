### Predicate


Для построения дерева решений нам потребуется на каждом шаге разделять выборку, 
находящуюся в узле дерева, на 2 независимых подвыборки и подсчитывать энтропию 
каждой. Для удобства мы создадим отдельный класс `Predicate` в файле `divide.py`, хранящий в себе значение 
признака и порогового значения, по которому происходит разделение.



### Задание

Реализуйте метод `divide`, позволяющий разбить выборку на 2 независимых 
подвыборки по какому-либо признаку. Функция должна принимать на вход сам 
датасет (`X`) и метки класса (`y`). Обратите внимание, что в датасете присутствуют 
признаки двух типов – номинальные и количественные. Сначала метод должен проверить, 
является ли признак количественным, и в таком случае пороговое условие должно 
представлять собой неравенство. Для номинальных признаков количество предикатов 
равно количеству уникальных значений признака, поэтому пороговое условие, по 
которому происходит разделение выборки, является проверкой признака на равенство. 
Метод должен возвращать разделенные по данному признаку датасет и метки класса.

<div class="hint">

Создайте массив-фильтр `mask`, сравнивая элементы в нужной колонке с 
пороговым условием, и используйте его для того, чтобы разделить выборку. </div>

<div class="hint">

Чтобы получить вторую часть выборки, которая не прошла по 
пороговому условию, можно применить [побитовое отрицание](https://numpy.org/doc/stable/reference/generated/numpy.invert.html) к созданному 
массиву для фильтрации данных. Оператор `~` может быть использован вместо np.invert для краткости при работе с ndarrays.</div>

На этом этапе не обращайте внимание на другой метод класса, его нужно будет
реализовать в следующем задании.

Для того чтобы посмотреть на результаты работы кода, вы можете добавить
следующие строки в `task.py` и запустить его:
1. Необходимые импорты:
 ```python
        import numpy as np
        from divide import Predicate
```
2. Игрушечный датасет для проверки работы метода `divide` и вывод результата добавьте в блок `main`.
```python
        predicate = Predicate(3, 'clear')           
        X = np.array([[1, 1, 1, 'clear'],
                    [2, 2, 2, 'clear'],
                    [3, 3, 3, 'green'],
                    [1, 2, 3, 'black']])
        y = np.array([1, 2, 3, 4])

        X1, y1, X2, y2 = predicate.divide(X, y)
        print(f'Division result: '
            f'\nFirst group labels: {y1} '
            f'\nFirst group objects: {X1} '
            f'\nSecond group labels: {y2} '
            f'\nSecond group objects: {X2}\n')
```