#!/bin/sh

## Build the model for evaluating
echo "Building for evaluating..."

bazel build coil_20_eval

# Directory where the validation images are
BASE_DIR=$(pwd)
PREPROCESSED_IMAGES_DIR=$BASE_DIR/images/inception-images

# Directory where save the evaluation results
EVAL_DIR=$BASE_DIR/inception-retrained/eval

# Directory where the retrained model is
RETRAINED_MODEL_DIR=$BASE_DIR/inception-retrained

echo "Running evaluation..."

bazel-bin/coil_20_eval \
    --num_gpus=1 \
    --batch_size=1 \
    --eval_dir="${EVAL_DIR}" \
    --data_dir="${PREPROCESSED_IMAGES_DIR}" \
    --subset=validation \
    --num_examples= \
    --checkpoint_dir="${RETRAINED_MODEL_DIR}" \
    --input_queue_memory_factor=1 \
    --run_once
