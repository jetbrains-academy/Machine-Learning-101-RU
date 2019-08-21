### K-means

K-means clustering is a type of unsupervised learning, which is used when the resulting groups in the data are unknown.
The goal of this algorithm is to find groups in the data, with the number of groups represented by the variable K.

The algorithm works iteratively to assign each data point to one of K groups based on the features that are provided.
Data points are clustered based on feature similarity.


The Κ-means clustering algorithm uses iterative refinement to produce a final result.
The algorithm inputs are the number of clusters Κ and the data set. The data set is a collection of features for each data point.
The algorithms starts with initial estimates for the Κ centroids, which can either be randomly generated or randomly selected from the data set.
The algorithm then iterates between two steps:

1. Data assignment step. Each centroid defines one of the clusters. In this step, each data point is assigned to its nearest centroid.
More formally,

$$c_i = \underset{{c \in 1\dots k}}{\arg\min}  \rho(x_i, \mu_c)$$

where
- $c_i$ -- cluster center assigned to the $x_i$ data point
- $\rho(x_i, \mu_c)$ -- distance between $x_i$ data point and $\mu_c$ cluster center
- $\mu_{c}$ -- cluster center

2. Centroid update step. In this step, the centroids are recomputed by taking the mean of all data points assigned to that centroid's cluster.

$$ {\mu_{c} = \frac{\sum\limits_{j=1,\dots, n} [c_i = c] x_i^j}{\sum\limits_{c_i = c} 1} } $$


The algorithm iterates between steps one and two until a stopping criteria is met
(i.e., no data points change clusters, the sum of the distances is minimized, or some maximum number of iterations is reached).
This algorithm is guaranteed to converge to a result.


### Задание

Реализуйте функцию, `k_means(X, n_clusters, distance_metric)`, которая принимает матрицу $X$ размерности
`(n_samples, n_features)`, количество кластеров, на которые мы хотим разбить изображение и метрику. 

Результатом работы функции является пара из вектора размера `n_samples`, где в $i$-й ячейке содержится кластер,
соответствующий $i$-му пикселю, и вектор размера `(n_clusters)` с центрами кластеров.

