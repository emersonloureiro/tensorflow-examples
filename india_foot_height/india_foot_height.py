import tensorflow as tf
import csv
import matplotlib.pyplot as plt
import numpy as np

plt.ion()
xs_temp = []
ys = []
# Bias
bs = []
with open('india_foot_height.csv') as inputFile:
    csvReader = csv.reader(inputFile, delimiter=',')
    for line in csvReader:
        xs_temp.append(float(line[0]))
        ys.append(float(line[1]))
        bs.append(1.0)

xs_mean = np.mean(xs_temp)
xs_std = np.std(xs_temp)
xs = []
for i in range(len(xs_temp)):
    x = (xs_temp[i] - xs_mean) / xs_std
    xs.append(x)

X0 = tf.placeholder(tf.float32)
X1 = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
m = len(xs)

# Weights
W0 = tf.Variable(tf.random_normal([1]), name="theta0")
W1 = tf.Variable(tf.random_normal([1]), name="theta1")
# Hypothesis
H = tf.add(tf.mul(W0, X0), tf.mul(W1, X1))
# Cost function
j = tf.reduce_sum(tf.pow(H - Y, 2)) / (2 * m)

# Gradient descent
learning_rate = 0.5
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(j)

prev_training_cost = 0.0
session = tf.Session()
session.run(tf.initialize_all_variables())

while 1:
    for (b, x, y) in zip(bs, xs, ys):
        session.run(optimizer, feed_dict={X0: b, X1: x, Y: y})
    training_cost = session.run(j, feed_dict={X0: bs, X1: xs, Y: ys})
    print(training_cost)
    if np.abs(prev_training_cost - training_cost) < 0.000001:
        break
    prev_training_cost = training_cost

# Plotting
fig, ax = plt.subplots(1, 1)
# Plotting input data
ax.scatter(xs, ys)

# Plotting the fitted function
min_x = np.min(xs)
max_x = np.max(xs)
min_y = session.run(H, feed_dict={X0: 1.0, X1: min_x})
max_y = session.run(H, feed_dict={X0: 1.0, X1: max_x})

plt.plot([min_x, max_x], [min_y, max_y])
fig.show()
plt.draw()
plt.waitforbuttonpress()
