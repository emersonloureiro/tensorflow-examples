# location to where to save the TFRecord data.
OUTPUT_DIRECTORY=$HOME/Development/learning/tensorflow-examples/coil-20-cnn
TRAIN_DIR=$HOME/Development/learning/tensorflow-examples/coil-20-cnn/images/training
VALIDATION_DIR=$HOME/Development/learning/tensorflow-examples/coil-20-cnn/images/validation
LABELS_FILE=$HOME/Development/learning/tensorflow-examples/coil-20-cnn/labels.txt

# build the preprocessing script.
bazel build $HOME/Development/osProjects/tensorflow-models/inception/build_image_data

# convert the data.
bazel-bin/inception/build_image_data --train_directory="${TRAIN_DIR}" --validation_directory="${VALIDATION_DIR}" --output_directory="${OUTPUT_DIRECTORY}" --labels_file="${LABELS_FILE}" --train_shards=128 --validation_shards=24 --num_threads=8
