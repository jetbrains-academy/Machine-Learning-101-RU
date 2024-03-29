####Information Gain
Количество получаемой информации ([Information Gain](https://en.wikipedia.org/wiki/Information_gain_in_decision_trees)) в теории информации &mdash; 
это количество информации, полученной о случайной величине при наблюдении за другой 
случайной величиной. Оно основано на уменьшении энтропии после разбиения выборки по 
тому или иному признаку и позволяет оценить качество разбиения. Другими словами, 
ожидаемый information gain &mdash; это изменение информационной энтропии H от предыдущего 
состояния к состоянию, которое принимает некоторую информацию (условие) как заданную. 
Построение решающих деревьев заключается в нахождении признака (то есть гомогенных ветвей), 
дающего наибольшее количество информации.

$$IGain = H(parent_y) - H(children_{y|x}) $$

Мы вычитаем энтропию `Y` при условии `X` из энтропии `Y` для вычисления уменьшения неопределенности
`Y`, при условии наличия дополнительного знания `X` про `Y`.



### Задание

В том же файле `divide.py`, в методе `information_gain` класса `Predicate`: удалите оператор `pass`, 
раскомментируйте все строки, содержащие `# TODO`, а также строку с `return`. 

Реализуйте метод `information_gain`, который 
принимает выборку, разделяет ее на 2 независимые подвыборки и подсчитывает information gain.
Для разбиения выборки этот метод должен использовать метод divide, написанный на предыдущем 
шаге. 




<div class="hint">

Для подсчета information gain можно использовать приведенную выше формулу в таком виде:

`gain = entropy(y) - p * entropy(y1) - (1 - p) * entropy(y2)`

Где:
- `y` – это все метки объектов (parent);
- `y1` – один из двух получившихся после разбиения сетов меток (child);
- `y2` – второй из двух получившихся после разбиения сетов меток (child);
- `p` – доля объектов из первого дочернего сета среди всех объектов полного датасета (следовательно, `1 - p` – доля второго).
</div>

Для того чтобы посмотреть на результаты работы кода, вы можете добавить
следующую строку в блок `main` в `task.py` и запустить его:

```python
    print(f'Information Gain: {predicate.information_gain(X, y)}\n')     
```
Переменные, необходимые для корректной работы этого кода, вводились на предыдущих шагах; 
если вы до сих пор не работали с `task.py`, то обратите на них внимание.