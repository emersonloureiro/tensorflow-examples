import tensorflow as tf


class IrisNN:
    def __init__(self, number_hidden_units, number_features, number_classes, m):
        self.number_hidden_units = number_hidden_units
        self.number_features = number_features
        self.number_classes = number_classes

        # Input layer
        self.X = tf.placeholder(tf.float32, shape=[m, self.number_features + 1], name='X')

        # Hidden layer 1
        W1 = tf.Variable(tf.random_normal([self.number_hidden_units, self.number_features + 1]), name='W1')
        A1 = tf.sigmoid(tf.matmul(W1, self.X, transpose_b=True), name='A1')

        # Output layer
        W2 = tf.Variable(tf.random_normal([self.number_classes, self.number_hidden_units]), name='W2')
        H = tf.transpose(tf.nn.softmax(tf.transpose(tf.matmul(W2, A1))), name='H')

        # Cost function
        self.Y = tf.placeholder(tf.float32, shape=[self.number_classes, m], name='Y')
        self.j = -tf.reduce_sum(self.Y * tf.log(H), name='j')
