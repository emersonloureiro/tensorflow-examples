import tensorflow as tf
import numpy as np
import os
from os.path import isfile, join
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize

np.set_printoptions(threshold=np.inf, precision=3, linewidth=1000, suppress=True)


class ImageReader:
    def __init__(self, base_folder, classes, number_of_classes):
        self.number_of_classes = number_of_classes
        class_index = 0
        image_classes = []
        decoded_images = []
        for class_name in classes:
            path = '{}/{}'.format(base_folder, class_name)
            with tf.Session() as sess:
                images = ['{}/{}'.format(path, f) for f in os.listdir(path) if isfile(join(path, f)) & f.endswith('.png')]
                filename_queue = tf.train.string_input_producer(images)
                key, value = tf.WholeFileReader().read(filename_queue)
                decoded_image = tf.image.decode_png(value)
                sess.run(tf.initialize_all_variables())
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord)

                print('Loading {} images for class \'{}\'...'.format(len(images), class_name))
                for i in range(len(images)):
                    decoded_images.append(np.copy(decoded_image.eval()))
                    image_classes.append(class_index)
                print('Finished loading images for class \'{}\''.format(class_name))
                coord.request_stop()
                coord.join(threads)
            class_index = class_index + 1

        print('Splitting dataset...')
        xs, xs_test, ys, ys_test = train_test_split(decoded_images, image_classes, train_size=0.7)
        self.xs = np.asarray(xs)
        self.xs_test = np.asarray(xs_test)
        self.ys = np.transpose(label_binarize(ys, classes=range(len(classes))))
        self.ys_test = np.transpose(label_binarize(ys_test, classes=range(len(classes))))
        print('Dataset split')
