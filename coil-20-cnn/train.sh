#!/bin/sh

PRETRAINED_MODEL_PATH="$1"

if [ -z "$PRETRAINED_MODEL_PATH" ]
then
    echo "Missing parameters"
	echo "Usage: ./train.sh PATH_TO_PRETRAINED_MODEL"
	exit 1
fi

BASE_DIR=$(pwd)
PREPROCESSED_IMAGES_DIR=$BASE_DIR/images/inception-images

echo "Building for retraining..."

bazel build coil_20_train

# Directory where to save the checkpoint and events files.
RETRAINED_MODEL_DIR=$BASE_DIR/inception-retrained

echo "Retraining model..."

bazel-bin/coil_20_train \
    --num_gpus=10 \
    --batch_size=10 \
    --train_dir="${RETRAINED_MODEL_DIR}" \
    --data_dir="${PREPROCESSED_IMAGES_DIR}" \
    --pretrained_model_checkpoint_path="${PRETRAINED_MODEL_PATH}" \
    --fine_tune=True \
    --initial_learning_rate=0.001 \
    --input_queue_memory_factor=1
