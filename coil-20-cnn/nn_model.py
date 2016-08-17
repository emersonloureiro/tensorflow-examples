from cnn import *


class Model:
    def __init__(self, number_of_classes):
        # Input images
        self.input_channels = 1
        self.X = tf.placeholder(tf.float32, [None, 128, 128, self.input_channels], 'X')

        # Convolutional layers stacked up
        self.convolutional_layer_1 = ConvolutionalLayer(self.X, 5, 1, 50, '1')
        max_pooling_layer_1 = MaxPoolingLayer(self.convolutional_layer_1, 2, 2, '1')

        self.convolutional_layer_2 = ConvolutionalLayer(max_pooling_layer_1.output, 5, 1, 50, '2')
        max_pooling_layer_2 = MaxPoolingLayer(self.convolutional_layer_2, 2, 2, '2')

        # Fully connected layer
        fully_connected_layer_hidden_units = 2000
        self.fully_connected_layer = FullyConnectedLayer(max_pooling_layer_2, fully_connected_layer_hidden_units)

        # Dropout layer
        self.dropout_layer = DropoutLayer(self.fully_connected_layer)

        # Output layer
        self.H = OutputLayer(self.dropout_layer, fully_connected_layer_hidden_units, number_of_classes)

        # Cost function
        self.Y = tf.placeholder(tf.float32, shape=[number_of_classes, None], name='Y')
        self.E = -tf.reduce_sum(self.Y * tf.log(tf.clip_by_value(self.H.output, 0.1e-10, 1.0)), 0, name='E')
        self.j = tf.reduce_mean(self.E, name='j')
