import pandas as pd
import math
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
    #plt.plot(distances)
    #plt.show()
    plt.hist(distances)
    plt.show()
    #plt.boxplot(distances)
    #plt.show()
    answer = max(distances)
    return distances, answer


def cluster(x, answer):
    m = DBSCAN(eps=answer)
    m.fit(x)
    clusters = m.labels_
    print('Количество классов: {}'.format(set(clusters)))
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
    #plt.savefig('Итоговое разделение на кластеры', dpi=300)
    plt.show()


def cleaning_distance(distances):
    percent = int(len(distances)*0.99)
    answer = distances[:percent]
    return max(answer)


x, y = generate_data(10000, 8, 2)
plot_data(x, y, 8)
distances, answer = neighbors(x)
answer_1 = float(input())
print(cleaning_distance(distances))
clusters = cluster(x, answer_1)
plot_result(clusters)

# реализовать расчет ближайших соседей вручную
# реализовать алгоритм dbscan вручную (ну или просто упрощенную версию)
# найти метрику для оценки работы алгоритма
