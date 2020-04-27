import os
import numpy as np
from sklearn.model_selection import KFold


def load_data(path):
    files = os.listdir(path)
    files.sort()

    train =[]
    test = []
    for i in np.arange(start=0,stop=len(files), step=2):

        train_path, test_path = os.path.join(path, files[i]), os.path.join(path,files[i+1])
        print(train_path)
        print(test_path)
        train_data = np.loadtxt(train_path, delimiter='\t')
        test_data =  np.loadtxt(test_path, delimiter='\t')
        x_train, y_train = train_data[:,:-1], train_data[:, -1]
        x_test, y_test = test_data[:,:-1], test_data[:, -1]

        train.append((x_train, y_train))
        test.append((x_test, y_test))

    return train, test

def load_data1(path):
    data = np.loadtxt(path, delimiter=',')
    X, y = data[:, :-1], data[:, -1]

    train = []
    test = []
    kf = KFold(n_splits=5, shuffle=True)
    for train_idx, test_idx in kf.split(X, y):
        x_train, y_train = X[train_idx], y[train_idx]
        x_test, y_test = X[test_idx], y[test_idx]
        train.append((x_train, y_train))
        test.append((x_test, y_test))

    return train, test


if __name__ == '__main__':
    path = "/Users/warm/PycharmProjects/pythonCode/毕业实验/improve_gan/datasets/Liver1.csv"
    train, test = load_data1(path)
    print(train[0][0].shape)
    print(train[0][1].shape)