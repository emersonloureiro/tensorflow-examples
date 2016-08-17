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

# Visualization
convolutional_layer_1 = model.convolutional_layers[0]
convolutional_layer_2 = model.convolutional_layers[1]
convolutional_layer_3 = model.convolutional_layers[2]
tf.scalar_summary('loss', model.j)
filter_1_dimension = convolutional_layer_1.filter.get_shape()
filter_1_reshaped = tf.reshape(convolutional_layer_1.filter, [filter_1_dimension[3].value * filter_1_dimension[2].value, filter_1_dimension[0].value, filter_1_dimension[1].value, 1])
tf.image_summary(convolutional_layer_1.filter.name, filter_1_reshaped, max_images=1000000)

merged = tf.merge_all_summaries()

with tf.Session() as session:
    summary_writer = tf.train.SummaryWriter('/Users/emerson/Development/learning/coil-20-cnn/events')
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
            session.run(train_step, feed_dict={model.X: x, model.Y: y, model.dropout_layer.keep_prob: 0.5})

            # Logging metrics
            print('Logging metrics...')
            summary, loss, _ = session.run([merged, model.j, filter_1_reshaped], feed_dict={model.X: x, model.Y: y, model.dropout_layer.keep_prob: 1.0})
            summary_writer.add_summary(summary, float('{}.{}'.format(k, i)))

            y_pred = np.transpose(session.run(model.H.output, feed_dict={model.X: image_reader.xs_test, model.Y: image_reader.ys_test, model.dropout_layer.keep_prob: 1.0})).argmax(1)
            y_true = np.transpose(image_reader.ys_test).argmax(1)
            f1 = f1_score(y_true, y_pred, average=None)
            print('F1 score {}'.format(f1))
            print('True: {}'.format(y_true))
            print('Pred: {}'.format(y_pred))

            if loss < 0.001:
                saver.save(session, '/Users/emerson/Development/learning/coil-20-cnn/model.ckpt')
                stop = True
        k = k + 1
        print('--------------------------------')
