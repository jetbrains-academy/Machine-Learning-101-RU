Индейцы племени пима проживают в центральной и южной части штата Аризона. По неизвестным на данный момент причинам
индейцы пима имеют критический риск заболевания сахарным диабетом (2 типа). В этом задании предлагается применить линейный
классификатор для диагностики и изучения сахарного диабета (2 типа) у представителей племени пима.
В файле `pima-indians-diabetes.csv` находятся медицинские данные представителей племени пима.
Последняя колонка каждой строки — индикатор наличия сахарного диабета (2 типа). Значения остальных колонок указаны в заголовке
файла.

Реализуйте функцию `read_data`, которая принимает путь к файлу с данными и возвращает пару из двух массивов NumPy.

- Первый элемент пары — матрица признаков `X`, в первой колонке которой находится константный признак `-1`, а в остальных
признаки из файла с данными.
- Второй элемент пары — вектор `y`, в котором `-1` означает наличие диабета, а `1` — его отсутствие.

![Pima](pima.png)