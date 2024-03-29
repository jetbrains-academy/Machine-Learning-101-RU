Скорость работы созданного нами алгоритма достаточно сильно зависит от количества кластеров. Попробуйте запустить программу для 4-х цветов, а затем — для 16-и, и вы увидите, насколько сильно изменится время выполнения.

Другая особенность метода — чувствительность к первоначальному выбору центров кластеров: не всегда случайный выбор центроидов дает удовлетворительный результат кластеризации. Если центроиды были расположены неудачно, то вы можете наблюдать артефакты. В нашем случае, если вы несколько раз запустите алгоритм с `k = 4`, то рано или поздно получите картинку, части которой будут раскрашены в неожиданные цвета. Такого рода недостаток устраняется последовательной доработкой метода первоначального выбора кластеров, а в конце вы сможете выбрать наиболее подходящий метод.


Описанный нами алгоритм реализован в библиотеке [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html). При использовании метода из scikit-learn необходимо задать различные параметры, которые мы изучили в этом уроке: количество кластеров, количество пробных запусков с различным первоначальным разбиением на кластеры, допустимая ошибка при схождении точек, максимальное количество итераций.

Развитием идеи алгоритма **k-средних** является алгоритм [k-means++](https://ru.wikipedia.org/wiki/K-means%2B%2B). Отличается он как раз реализацией стадии выбора первоначальных кластеров.

Еще одной модификацией является алгоритм [X-means, англ.](https://en.wikipedia.org/wiki/Determining_the_number_of_clusters_in_a_data_set#:~:text=In%20statistics%20and%20data%20mining,criterion%20(BIC)%20is%20reached). Метод принимает на вход диапазон возможного количества кластеров и, запустив алгоритм для каждого из этих значений, определяет наилучший результат согласно выбранной метрике качества.

Алгоритм **k-средних** также используется в более сложных задачах машинного обучения: к примеру, в [нейронной сети Кохонена](http://www.machinelearning.ru/wiki/index.php?title=%D0%9D%D0%B5%D0%B9%D1%80%D0%BE%D0%BD%D0%BD%D0%B0%D1%8F_%D1%81%D0%B5%D1%82%D1%8C_%D0%9A%D0%BE%D1%85%D0%BE%D0%BD%D0%B5%D0%BD%D0%B0).

