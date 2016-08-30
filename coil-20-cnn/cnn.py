import tensorflow as tf


class ConvolutionalLayer:
    def __init__(self, input, filter_size, stride, feature_maps, index):
        input_shape = input.get_shape()
        filter_shape = [filter_size, filter_size, input_shape[3].value, feature_maps]
        self.filter = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.5), name='W' + index)
        self.bias = tf.Variable(tf.constant(0.1, shape=[feature_maps], name='B' + index))
        convolution = tf.nn.conv2d(input, self.filter, [1, stride, stride, 1], 'VALID', name='Conv' + index) + self.bias
        self.output = tf.nn.relu(convolution, name='ReLU' + index)
        self.feature_maps = feature_maps


class MaxPoolingLayer:
    def __init__(self, convolutional_layer, filter_size, stride, index):
        filter_shape = [1, filter_size, filter_size, 1]
        stride = [1, stride, stride, 1]
        self.output = tf.nn.max_pool(convolutional_layer.output, filter_shape, stride, 'VALID', name='Max_' + index)
        self.feature_maps = convolutional_layer.feature_maps


class FullyConnectedLayer:
    def __init__(self, pooling_layer, hidden_units):
        max_pool_layer_dimensions = pooling_layer.output.get_shape()
        fc_input_size = pooling_layer.feature_maps * max_pool_layer_dimensions[1].value * max_pool_layer_dimensions[2].value
        self.weights = tf.Variable(tf.truncated_normal([hidden_units, fc_input_size], stddev=0.5), name='W_fc')
        pool_layer_flat = tf.reshape(pooling_layer.output, [fc_input_size, -1])
        self.bias = tf.Variable(tf.constant(0.1, shape=[hidden_units, 1], name='B_fc'))
        self.output = tf.nn.sigmoid(tf.matmul(self.weights, pool_layer_flat) + self.bias, name='FC')


class DropoutLayer:
    def __init__(self, input_layer):
        self.keep_prob = tf.placeholder(tf.float32, name='Keep_prob')
        self.output = tf.nn.dropout(input_layer.output, self.keep_prob, name='Drop')


class OutputLayer:
    def __init__(self, input_layer, hidden_units, number_of_classes):
        weights = tf.Variable(tf.truncated_normal([number_of_classes, hidden_units]), name='W_o')
        self.output = tf.transpose(tf.nn.softmax(tf.transpose(tf.matmul(weights, input_layer.output))), name='H')
