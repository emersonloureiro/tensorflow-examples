import csv
import numpy as np


class Input:
    def __init__(self, file_path, number_features, number_classes):
        self.current = 0
        xs_temp = []
        self.ys = []
        with open(file_path) as inputFile:
            csvReader = csv.reader(inputFile, delimiter=',')
            for line in csvReader:
                x = []
                for i in range(0, number_features):
                    x.append(float(line[i]))
                xs_temp.append(x)
                y = int(line[4])
                ys_temp = []
                for c in range(0, number_classes):
                    if y == c:
                        ys_temp.append(1)
                    else:
                        ys_temp.append(0)
                self.ys.append(ys_temp)

        self.xs = []
        xs_temp = np.array(xs_temp)
        xs_mean = np.mean(xs_temp, axis=0)
        xs_std = np.std(xs_temp, axis=0)
        for i in range(len(xs_temp)):
            normalized_xs = []
            # Bias
            normalized_xs.append(1)
            for j in range(len(xs_temp[i])):
                normalized_x = (xs_temp[i][j] - xs_mean[j]) / xs_std[j]
                normalized_xs.append(normalized_x)
            self.xs.append(normalized_xs)
        self.m = len(self.xs)
        self.ys = np.transpose(self.ys)

    def next_batch(self, batch_size):
        x_temp = []
        y_temp = []
        for i in range(self.current, self.current + batch_size):
            if i > self.m:
                break
            x_temp.append(self.xs[i])
            y_temp.append(self.ys[i])
        self.current = self.current + batch_size
        return x_temp, y_temp
