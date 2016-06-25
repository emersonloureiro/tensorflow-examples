import tensorflow as tf
import csv
import numpy as np
import matplotlib.pyplot as plt


class IrisNN:
    def __init__(self, number_hidden_units, number_features, number_classes):
        self.number_hidden_units = number_hidden_units
        self.number_features = number_features
        self.number_classes = number_classes


class Input:
    def __init__(self, file_path, iris_nn):
        self.current = 0
        xs_temp = []
        self.ys = []
        with open(file_path) as inputFile:
            csvReader = csv.reader(inputFile, delimiter=',')
            for line in csvReader:
                x = []
                for i in range(0, iris_nn.number_features):
                    x.append(float(line[i]))
                xs_temp.append(x)
                y = int(line[4])
                ys_temp = []
                for c in range(0, iris_nn.number_classes):
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


# The NN architecture info
iris_nn = IrisNN(3, 4, 3)
# Read the input file
input_data = Input('sample-datasets/iris-flower-edited.csv', iris_nn)

# Weights matrix
W1 = tf.Variable(tf.random_normal([iris_nn.number_hidden_units, iris_nn.number_features + 1]), name='W1')

# Input layer
batch_size = input_data.m
X = tf.placeholder(tf.float32, shape=[batch_size, iris_nn.number_features + 1], name='X')

# Hidden layer 1
A1 = tf.sigmoid(tf.matmul(W1, X, transpose_b=True), name='A1')

# Output layer
W2 = tf.Variable(tf.random_normal([iris_nn.number_classes, iris_nn.number_hidden_units]), name='W2')
H = tf.transpose(tf.nn.softmax(tf.transpose(tf.matmul(W2, A1))), name='H')

# Cost function
Y = tf.placeholder(tf.float32, shape=[iris_nn.number_classes, batch_size], name='Y')
j = -tf.reduce_sum(Y * tf.log(H), name='j')

# Gradient descent optimizer
optimizer = tf.train.GradientDescentOptimizer(0.005).minimize(j)

# Training
session = tf.Session()
session.run(tf.initialize_all_variables())
epochs = 2000
cost = []
iteration = []
for epoch in range(epochs):
    cost.append(session.run(j, feed_dict={X: input_data.xs, Y: input_data.ys}))
    iteration.append(epoch)
    session.run(optimizer, feed_dict={X: input_data.xs, Y: input_data.ys})

plt.ion()
fig, ax = plt.subplots(1, 1)
plt.plot(iteration, cost)
fig.show()
plt.draw()
plt.waitforbuttonpress()
