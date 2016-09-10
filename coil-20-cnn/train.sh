#!/bin/sh

PRETRAINED_MODEL_PATH="$1"

if [ -z "$PRETRAINED_MODEL_PATH" ]
then
    echo "Missing parameters"
	echo "Usage: ./train.sh PATH_TO_PRETRAINED_MODEL"
	exit 1
fi

PREPROCESSED_IMAGES_DIR=$BASE_DIR/images/inception-images

cd $TENSORFLOW_INCEPTION

echo "Building model for retraining..."

# Build the model. Note that we need to make sure the TensorFlow is ready to
# use before this as this command will not build TensorFlow.
bazel build inception/coil_20_train

# Directory where to save the checkpoint and events files.
RETRAINED_MODEL_DIR=$BASE_DIR/inception-retrained

echo "Retraining model..."

bazel-bin/inception/coil_20_train \
  --train_dir="${RETRAINED_MODEL_DIR}" \
  --data_dir="${PREPROCESSED_IMAGES_DIR}" \
  --pretrained_model_checkpoint_path="${PRETRAINED_MODEL_PATH}" \
  --fine_tune=True \
  --initial_learning_rate=0.001 \
  --input_queue_memory_factor=1