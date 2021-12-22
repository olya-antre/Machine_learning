import time
from datetime import timedelta
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

def generate_data(sample_size, classes_num, random_seed):
    x, y = datasets.make_blobs(n_samples=sample_size,
                               centers=classes_num,
                               n_features=2,
                               cluster_std=1.2,
                               random_state=random_seed)
    df = pd.DataFrame(dict(x=x[:, 0], y=x[:, 1], label=y))
    return df, x, y


def plot_data(df, classes_num):
    cmap = plt.cm.get_cmap('hsv', classes_num + 1)
    fig, ax = plt.subplots()
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax,
                   kind='scatter',
                   x='x',
                   y='y',
                   color=cmap(key))
    plt.savefig('SVM', dpi=300)
    plt.show()


def split_data(df, test_percent):
    return train_test_split(df.drop('label', axis=1),
                            df['label'],
                            test_size=test_percent)


def svc(x_train, y_train, x_test):
    clf = OneVsRestClassifier(SVC(kernel='poly'))
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    return y_pred


def calculate_accuracy(y_pred, y_test):
    y_test = y_test.to_numpy()
    sum_correct_answer = sum([int(y_pred[i] == y_test[i]) for i in range(len(y_test))])
    print('Accuracy: ',
          sum_correct_answer / float(len(y_test)))


classes_num = 10
size_sample = 8000
test_percent = 0.8
random_seed = 100

start_time = time.monotonic()

df, x, y = generate_data(size_sample, classes_num, random_seed)
x_train, x_test, y_train, y_test = split_data(df, test_percent)
y_pred = svc(x_train, y_train, x_test)

#plot_data(df, classes_num)
calculate_accuracy(y_pred, y_test)

end_time = time.monotonic()
print('Duration: {}'.format(timedelta(seconds=end_time-start_time)))
