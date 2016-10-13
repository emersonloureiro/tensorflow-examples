# tensorflow-examples

A set of examples using tensorflow for machine learning classification

## Regression

### Left Footprint Length and Height in Indian Adult Male Tamils

Dataset originally taken from [here](http://www.stat.ufl.edu/~winner/datasets.html). Full dataset available [here](http://www.stat.ufl.edu/~winner/data/india_foot_height.dat).

To run, `cd` into `india_foot_height` and run `python india_foot_height.py`. It'll print out the value of the cost function on each iteration of training and in the end plot the fit function on the dataset.

## Classification

### Iris Flower

Description of the dataset available [here](https://archive.ics.uci.edu/ml/datasets/Iris), and the actual dataset is available [here](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data).

To run, `cd` into `iris-flower` and run `python iris-flower.py`. It'll split the total dataset into training and test sets, train a neural network and then plot the value cost function over time.

## Visual Learning

### COIL-20

A convolutional neural network for image classification, using a subset of the [COIL-20](http://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php) image dataset - three classes to be more precise, rubber-duck, mug, pigbank. It uses Google's pretrained Inception V3 model. To run this, you'll need:

1. cd into `coil-20`
2. `mkdir images/inception-images` - this is where the pre-processed images for re-training will be kept
3. `./preprocess.sh` - this will pre-process all images under images/training, to a format that the tensorflow inception model accepts
4. `./train.sh PATH_TO_INCEPTION_V3_MODEL` - you'll need to download the Inception V3 model and use the path you saved in place of `PATH_TO_INCEPTION_V3_MODEL`. You can tune the training parameters (e.g., max iterations) by modifying the `train.sh` script
5. After training, run `./eval` to see the level of accuracy you achieved
