import tensorflow as tf
import csv
import numpy as np


class IrisNN:
    def __init__(self, number_hidden_units, number_features, number_classes):
        self.number_hidden_units = number_hidden_units
        self.number_features = number_features
        self.number_classes = number_classes


class Input:
    def __init__(self, file_path, iris_nn):
        self.current = 0
        self.xs = []
        self.ys = []
        with open(file_path) as inputFile:
            csvReader = csv.reader(inputFile, delimiter=',')
            for line in csvReader:
                x = []
                # Bias
                x.append(1)
                for i in range(0, iris_nn.number_features):
                    x.append(float(line[i]))
                self.xs.append(x)
                y = int(line[4])
                ys_temp = []
                for c in range(0, iris_nn.number_classes):
                    if y == c:
                        ys_temp.append(1)
                    else:
                        ys_temp.append(0)
                self.ys.append(ys_temp)
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
W = tf.Variable(tf.random_normal([iris_nn.number_hidden_units, iris_nn.number_features + 1]), name='W')

# Input layer
batch_size = input_data.m
X = tf.placeholder(tf.float32, shape=[batch_size, iris_nn.number_features + 1], name='X')

# Hidden layer 1
A1 = tf.matmul(W, X, transpose_b=True, name='A1')

# Output layer
Y = tf.placeholder(tf.float32, shape=[iris_nn.number_classes, batch_size], name='Y')
H = tf.nn.softmax(A1, name='H')

# Cost function
j = tf.reduce_sum(np.power((H - Y), 2)) + (2/input_data.m)
# j = -tf.reduce_sum(Y * tf.log(H), name='j')

# Gradient descent optimizer
optimizer = tf.train.GradientDescentOptimizer(5.0).minimize(j)

# Training
session = tf.Session()
session.run(tf.initialize_all_variables())

# Evaluation
correct_prediction = tf.equal(tf.argmax(H, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

epochs = 55
for epoch in range(epochs):
    # print('Cost %.5f' % session.run(j, feed_dict={X: input_data.xs, Y: input_data.ys}))
    # print('Accuracy %.5f' % session.run(accuracy, feed_dict={X: input_data.xs, Y: input_data.ys}))
    print(session.run(H, feed_dict={X: input_data.xs}))
    # print(session.run(A1, feed_dict={X: input_data.xs}))
    # print(W.eval(session))
    print('=====================================================================')
    session.run(optimizer, feed_dict={X: input_data.xs, Y: input_data.ys})
