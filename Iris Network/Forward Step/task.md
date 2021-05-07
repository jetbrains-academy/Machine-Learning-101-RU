Ирисы Фишера (**ирисы Андерсона**, **the iris data**) - наиболее распространенный датасет для тестирования алгоритмов машинного обучения. Данные содержат 4 признака:
- длину наружной доли околоцветника (чашелистика)
- ширину наружной доли околоцветника (чашелистика)
- длину внутренней доли околоцветника (лепестка)
- ширину внутренней доли околоцветника (лепестка)
для различных видов цветков ириса - щетинистого, вергинского и разноцветного. Для каждого вида представлено по 50 экземпляров.

Мы построим классификационную модель на основе этих данных, используя нейронную сеть. Для простоты мы будем использовать лишь длину и ширину лепестков разноцветного и вергинского видов ириса.

### Neuron

Структурной единицей нейронной сети является нейрон.
Нейрон получает входные данные, производит над ними некие математические преобразования и выдает единый результат.

![Neuron scheme](neuron-scheme.png)

Сначала нейрон складывает значения всех входных данных. На изображении представлены $n$ входных сигналов ($x^1, x^2, \dots x^n$ ).

Эти значения перед сложением умножаются на весовые коэффициенты (**weight**) ($w_1, w_2, \dots w_n$).
Веса - единственные значения, которые будут изменены в процессе обучения. Смещение $b$ может быть добавлено к полученному выходному значению для корректировки.

После сложения входных данных, нейрон применяет к полученному значению функцию активации (**activation function**) $\sigma$.
Функция активации обычно предназначена для того, чтобы нормировать вычисленное ранее значение между `0` и `1`.

Зачастую в качестве функции активации используется [сигмоида](https://ru.wikipedia.org/wiki/%D0%A1%D0%B8%D0%B3%D0%BC%D0%BE%D0%B8%D0%B4%D0%B0) (**sigmoid function**).

Формула для выходных данных нейрона:

$$a = \sigma(\sum\limits_{j=1}^n w_j x^j + b)$$

### Нейронная сеть

Нейронная сеть - всего лишь совокупность нейронов, соединенных вместе.

Нейронные сети состоят из следующих частей:
- Слой входных нейронов (**input layer**)
- Произвольное количество скрытых нейронов (**hidden layer**)
- Слой выходных нейронов (**output layer**)
- Набор весов и смещений между слоями, $W$
- Выбора функции активации для каждого из скрытых слоев $\sigma$

Ниже представлена 2-слойная нейронная сеть. При подсчете количества слоев слой входных нейронов не учитывается.
![Neuralnet](neuralnet.png)


### Обучение нейронной сети
Выходные данные $\hat{y}$ простой 2-слойной нейронной сети:
$$\hat{y} = \sigma(W_2 \sigma(W_1x + b_1) + b_2$$

Веса $W$ и смещения $b$ - единственные параметры, влияющие на выходные данные $\hat{y}$.
Правильные значения весов и смещений определяют корректность прогнозов.
Процесс настройки весов и смещений известен как обучение нейронной сети.

Каждая итерация обучающего процесса состоит из следующих шагов:
- Вычисление предсказания для выходных данных, известного как "упреждение" или же "прямая связь" (**feedforward**)
- Обновление весов и смещений, известное как "обратное распространение ошибки" (**backpropagation**)

### Прямая связь

Прямая связь - обычные вычисления, и для простой 2-слойной нейронной сети выходные данные будут выглядеть следующим образом:

$$\hat{y} = \sigma(W_2 \sigma(W_1x + b_1) + b_2$$

### Задание

Реализуйте функцию `feedforward`. Для простоты предположим смещения равными 0.
