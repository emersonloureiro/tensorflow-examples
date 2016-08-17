import image_reader
from nn_model import *
import numpy as np
from sklearn.metrics import f1_score

# Training
classes = ['mug', 'pigbank', 'rubber-duck']
image_width = 250
image_height = 350
image_reader = image_reader.ImageReader('images', classes, len(classes))
batch_size = 20
model = Model(len(classes))
train_step = tf.train.AdamOptimizer(0.05).minimize(model.j)
saver = tf.train.Saver()
correct_prediction = tf.cast(tf.equal(tf.argmax(tf.transpose(model.H.output), 1), tf.argmax(tf.transpose(model.Y), 1)), tf.int32)

# Visualization
convolutional_layer_1 = model.convolutional_layer_1
convolutional_layer_2 = model.convolutional_layer_2
tf.scalar_summary('loss', model.j)
filter_1_dimension = convolutional_layer_1.filter.get_shape()
filter_1_reshaped = tf.reshape(convolutional_layer_1.filter, [filter_1_dimension[3].value * filter_1_dimension[2].value, filter_1_dimension[0].value, filter_1_dimension[1].value, 1])
tf.image_summary(convolutional_layer_1.filter.name, filter_1_reshaped, max_images=1000000)

merged = tf.merge_all_summaries()

base_dir = '/Users/emerson/Development/learning/tensorflow-examples/coil-20-cnn'

with tf.Session() as session:
    summary_writer = tf.train.SummaryWriter('{}/events'.format(base_dir))
    print('Initializing variables...')
    session.run(tf.initialize_all_variables())
    print('Starting training...')
    k = 0
    stop = False
    number_of_chunks = np.ceil(len(image_reader.xs) / batch_size)

    while not stop:
        print('Iteration {}...'.format(k + 1))
        xs_batches = np.array_split(image_reader.xs, number_of_chunks)
        ys_batches = np.array_split(image_reader.ys, number_of_chunks, axis=1)

        for i in range(len(xs_batches)):
            x = xs_batches[i]
            y = ys_batches[i]

            # Training
            print('Running optimizer...')
            session.run(train_step, feed_dict={model.X: x, model.Y: y, model.dropout_layer.keep_prob: 1.0})

            # Logging metrics
            print('Logging metrics...')
            summary, loss, _ = session.run([merged, model.j, filter_1_reshaped], feed_dict={model.X: x, model.Y: y, model.dropout_layer.keep_prob: 1.0})
            summary_writer.add_summary(summary, float('{}.{}'.format(k, i)))
            print('Loss: {}'.format(loss))

            train_predictions = session.run(correct_prediction, feed_dict={model.X: x, model.Y: y, model.dropout_layer.keep_prob: 1.0})
            print('Train preds: {}'.format(train_predictions))

            validation_predictions = session.run(correct_prediction, feed_dict={model.X: image_reader.xs_test, model.Y: image_reader.ys_test, model.dropout_layer.keep_prob: 1.0})
            print('Valid preds: {}'.format(validation_predictions))

            if loss < 0.001:
                saver.save(session, '{}/model.ckpt'.format(base_dir))
                stop = True
        k = k + 1
        print('--------------------------------')
