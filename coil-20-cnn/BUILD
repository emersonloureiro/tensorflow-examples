# Description:
# Example TensorFlow models for ImageNet.

package(default_visibility = [":internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_binary(
    name = "coil_20_train",
    srcs = [
        "coil_20_train.py",
    ],
    deps = [
        ":coil_20_data",
        ":inception_train",
    ],
)

py_library(
    name = "coil_20_data",
    srcs = [
        "coil_20_data.py",
    ],
    deps = [
        ":dataset",
    ],
)

py_library(
    name = "dataset",
    srcs = [
        "dataset.py",
    ],
)

py_library(
    name = "inception",
    srcs = [
        "inception_model.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":dataset",
        "//inception/slim",
    ],
)
