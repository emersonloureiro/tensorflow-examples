import tensorflow as tf
import matplotlib.pyplot as plt
from input import Input
from iris_nn import IrisNN

# Read the input file
input_data = Input('../sample-datasets/iris-flower-edited.csv', 4, 3)
# The NN architecture & info
iris_nn = IrisNN(3, 4, 3, input_data.m)

# Gradient descent optimizer
optimizer = tf.train.GradientDescentOptimizer(0.005).minimize(iris_nn.j)

# Training
session = tf.Session()
session.run(tf.initialize_all_variables())
epochs = 2000
cost = []
iteration = []
for epoch in range(epochs):
    cost.append(session.run(iris_nn.j, feed_dict={iris_nn.X: input_data.xs, iris_nn.Y: input_data.ys}))
    iteration.append(epoch)
    session.run(optimizer, feed_dict={iris_nn.X: input_data.xs, iris_nn.Y: input_data.ys})

plt.ion()
fig, ax = plt.subplots(1, 1)
plt.plot(iteration, cost)
fig.show()
plt.draw()
plt.waitforbuttonpress()
