import math
import random
import time
from datetime import timedelta

import matplotlib.pyplot as plt

from sklearn import datasets


def generate_data(classes_num, sample_size, random_seed):
    x, y = datasets.make_blobs(n_samples=sample_size,
                               centers=classes_num,
                               n_features=2,
                               cluster_std=1.2,
                               random_state=random_seed)
    data = []
    for i in range(len(x)):
        data.append([[x[i][0], x[i][1]], y[i]])
    return data, x, y


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
    plt.savefig('KNN', dpi=300)
    plt.show()


def data_split(data, test_percent):
    train_data, test_data = [], []
    for point in data:
        if random.random() < test_percent:
            test_data.append(point)
        else:
            train_data.append(point)
    return train_data, test_data


def distance(point_1, point_2):
    answer = math.sqrt((point_1[0]-point_2[0])**2 + (point_1[1]-point_2[1])**2)
    return answer


def knn(train_data, test_data, k, classes_num):
    test_labels = []
    for point in test_data:
        test_distance = [
            [distance(point, train_data[i][0]), train_data[i][1]] for i in range(len(train_data))
        ]
        neighbors = [0 for i in range(classes_num)]
        for d in sorted(test_distance)[0:k]:
            neighbors[d[1]] += 1
        test_labels.append(sorted(zip(neighbors, range(classes_num)), reverse=True)[0][1])
    return test_labels


def calculate_accuracy(classes_num, k, test_split_data, train_split_data):
    test_data = [test_split_data[i][0] for i in range(len(test_split_data))]
    test_data_labels = knn(train_split_data, test_data, k, classes_num)
    sum_correct_answer = sum( [int(test_data_labels[i] == test_split_data[i][1]) for i in range(len(test_split_data))] )
    print('Точность: ', sum_correct_answer / float(len(test_split_data)))


classes_num = 10
size_sample = 8000
k = 3
test_percent = 0.8
random_seed = 100

start_time = time.monotonic()

data, x, y = generate_data(classes_num, size_sample, random_seed)
train_split_data, test_split_data = data_split(data, test_percent)

#plot_data(x, y, classes_num)
calculate_accuracy(classes_num, k, test_split_data, train_split_data)

end_time = time.monotonic()
print('Duration: {}'.format(timedelta(seconds=end_time-start_time)))
