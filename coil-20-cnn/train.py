import image_reader
from nn_model import *
import numpy as np
from sklearn.metrics import f1_score

# Training
classes = ['mug', 'pigbank', 'rubber-duck']
image_width = 250
image_height = 350
image_reader = image_reader.ImageReader('images', classes, len(classes))
batch_size = 5
model = Model(len(classes))
train_step = tf.train.AdamOptimizer(0.05).minimize(model.j)
saver = tf.train.Saver()
correct_prediction = tf.cast(tf.equal(tf.argmax(tf.transpose(model.H.output), 1), tf.argmax(tf.transpose(model.Y), 1)), tf.int32)

# Visualization
convolutional_layer_1 = model.convolutional_layer_1
convolutional_layer_2 = model.convolutional_layer_2
tf.scalar_summary('loss', model.j)

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
        print('--------------------------------')
        print('Iteration {}...'.format(k + 1))
        xs_batches = np.array_split(image_reader.xs, number_of_chunks)
        ys_batches = np.array_split(image_reader.ys, number_of_chunks, axis=1)

        print('Running optimizer for batches...')
        for i in range(len(xs_batches)):
            x = xs_batches[i]
            y = ys_batches[i]

            # Training
            session.run(train_step, feed_dict={model.X: x, model.Y: y, model.dropout_layer.keep_prob: 0.6})

        # Logging visualization data
        print('Calculating metrics...')
        train_pred, summary, loss, = session.run([correct_prediction, merged, model.j], feed_dict={model.X: image_reader.xs, model.Y: image_reader.ys, model.dropout_layer.keep_prob: 1.0})
        print('Logging visualization data...')
        summary_writer.add_summary(summary, float('{}'.format(k)))
        print('Loss: {}'.format(loss))
        print('Train accuracy: {}'.format(np.mean(train_pred)))

        # Accuracy
        validation_pred, h = session.run([correct_prediction, model.H.output], feed_dict={model.X: image_reader.xs_test, model.Y: image_reader.ys_test, model.dropout_layer.keep_prob: 1.0})
        validation_accuracy = np.mean(validation_pred)
        # F1
        y_pred = np.transpose(h).argmax(1)
        y_true = np.transpose(image_reader.ys_test).argmax(1)
        f1 = f1_score(y_true, y_pred, average=None)

        print('Valid accuracy: {}'.format(validation_accuracy))
        print('F1 score {}'.format(f1))

        if np.mean(f1) > 0.99:
            saver.save(session, '{}/model.ckpt'.format(base_dir))
            stop = True

        k = k + 1
