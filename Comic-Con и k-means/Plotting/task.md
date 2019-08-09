Для оценки результата работы реализуйте две функции.
Первая функция `centroid_histogram(labels)` строит гистограмму на основе количества пикселей, приписанных каждому кластеру и возвращает ее в виде вектора.
Вторая функция `plot_colors(hist, centroids)` принимает построенную гистограмму и список центров кластеров и строит bar chart, показывающий относительную частоту каждого цвета. 

Пример: 
<img src="barchart.png"/>

Функция может выглядеть так:

    def plot_colors(hist, centroids):
        # инициализировать переменные bar и start_x
    
        for (percent, color) in zip(hist, centroids):
            # вычислить end_x
            cv2.rectangle(bar, (int(start_x), 0), (int(end_x), 50),
            color.astype("uint8").tolist(), -1)
            # обновить значение start_x
    
        return bar
	
