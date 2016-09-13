#!/bin/sh

PREPROCESSED_IMAGES_DIR=$BASE_DIR/images/inception-images
TRAIN_DIR=$BASE_DIR/images/training
VALIDATION_DIR=$BASE_DIR/images/validation
LABELS_FILE=$BASE_DIR/labels.txt

# Build the preprocessing scripts
bazel build inception/build_image_data

# convert the data.
bazel-bin/inception/build_image_data --train_directory="${TRAIN_DIR}" --validation_directory="${VALIDATION_DIR}" --output_directory="${PREPROCESSED_IMAGES_DIR}" --labels_file="${LABELS_FILE}" --train_shards=128 --validation_shards=24 --num_threads=8

echo "Finished converting images"
