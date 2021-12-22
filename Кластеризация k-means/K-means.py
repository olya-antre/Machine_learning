import pandas as pd
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

def generate_data(sample_size, classes_num, random_seed):
    x, y = datasets.make_blobs(n_samples=sample_size,
                               centers=classes_num,
                               n_features=2,
                               random_state=random_seed)
    return x, y


def plot_data(x, y, classes_num):
    plt.figure(figsize=(15, 10))
    plt.tick_params(axis="both", which="major", labelsize=20)
    plt.grid(True)
    cmap = plt.cm.get_cmap('hsv', classes_num+1)
    for i in range(classes_num):
        plt.scatter(x[:, 0][y == i],
                    x[:, 1][y == i],
                    marker='o',
                    color=cmap(i))
    plt.savefig('Исходное разделение на кластеры', dpi=300)
    plt.show()


def neighbors(x):
    neigh = NearestNeighbors(n_neighbors=3)
    nbrs = neigh.fit(x)
    distances, indices = nbrs.kneighbors(x)
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]
    plt.plot(distances)
    plt.show()
    return distances


def cluster(x):
    m = DBSCAN(eps=0.162)
    m.fit(x)
    clusters = m.labels_
    print(clusters)
    return clusters


def plot_result(clusters):
    classes_num = len(set(clusters))
    plt.figure(figsize=(15, 10))
    plt.tick_params(axis="both", which="major", labelsize=20)
    plt.grid(True)
    cmap = plt.cm.get_cmap('hsv', classes_num + 1)
    for i in range(classes_num):
        plt.scatter(x[:, 0][y == i],
                    x[:, 1][y == i],
                    marker='o',
                    color=cmap(i))
    plt.savefig('Итоговое разделение на кластеры', dpi=300)
    plt.show()


x, y = generate_data(10000, 5, 6)
plot_data(x, y, 5)
distances = neighbors(x)
clusters = cluster(x)
plot_result(clusters)

# реализовать расчет ближайших соседей вручную
# реализовать алгоритм dbscan вручную (ну или просто упрощенную версию)
# найти метрику для оценки работы алгоритма
