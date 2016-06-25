import tensorflow as tf


class IrisNN:
    def __init__(self, number_hidden_units, number_features, number_classes, train_set_size, test_set_size):
        self.number_hidden_units = number_hidden_units
        self.number_features = number_features
        self.number_classes = number_classes

        # Training network
        # Input layer
        self.X = tf.placeholder(tf.float32, shape=[train_set_size, self.number_features + 1], name='X')
        # Hidden layer 1
        W1 = tf.Variable(tf.random_normal([self.number_hidden_units, self.number_features + 1]), name='W1')
        A1 = tf.sigmoid(tf.matmul(W1, self.X, transpose_b=True), name='A1')
        # Output layer
        W2 = tf.Variable(tf.random_normal([self.number_classes, self.number_hidden_units]), name='W2')
        self.H = tf.transpose(tf.nn.softmax(tf.transpose(tf.matmul(W2, A1))), name='H')
        # Cost function
        self.Y = tf.placeholder(tf.float32, shape=[self.number_classes, train_set_size], name='Y')
        self.j = -tf.reduce_sum(self.Y * tf.log(self.H), name='j')

        # Cross validation network
        # Input layer
        self.X_test_set = tf.placeholder(tf.float32, shape=[test_set_size, self.number_features + 1], name='X_test_set')
        # Hidden layer 1
        A1_test_set = tf.sigmoid(tf.matmul(W1, self.X_test_set, transpose_b=True), name='A1')
        # Output layer
        self.H_test_set = tf.transpose(tf.nn.softmax(tf.transpose(tf.matmul(W2, A1_test_set))), name='H')
