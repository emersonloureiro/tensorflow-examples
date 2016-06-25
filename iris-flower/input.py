import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize


class Input:
    def __init__(self, file_path, number_features):
        dataset = self.load_dataset(file_path, number_features)
        xs = dataset[:, 0:number_features + 1]
        ys = dataset[:, number_features + 1]
        self.xs, self.xs_test, ys, ys_test = train_test_split(xs, ys, train_size=0.6)
        self.ys = np.transpose(label_binarize(ys, classes=[0, 1, 2]))
        self.ys_test = np.transpose(label_binarize(ys_test, classes=[0, 1, 2]))
        self.m = self.xs.shape[0]
        self.test_set_size = self.xs_test.shape[0]

    def load_dataset(self, file_path, number_features):
        dataset = np.genfromtxt(file_path, delimiter=',')
        xs = dataset[:, 0:number_features]
        xs_mean = np.mean(xs, axis=0)
        xs_std = np.std(xs, axis=0)
        m = dataset.shape[0]
        bias = np.ones((m, 1))
        dataset = np.append(bias, dataset, axis=1)
        for i in range(m):
            for j in range(1, number_features):
                normalized_x = (dataset[i][j] - xs_mean[j - 1]) / xs_std[j - 1]
                dataset[i][j] = normalized_x
        return dataset
