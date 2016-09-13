# Description:
# Example TensorFlow models for ImageNet.

package(default_visibility = [":internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package_group(
    name = "internal",
    packages = ["//inception/..."],
)

py_binary(
    name = "coil_20_train",
    srcs = [
        "coil_20_train.py",
    ],
    deps = [
        ":coil_20_data",
        "//inception:inception_train",
    ],
)

py_library(
    name = "coil_20_data",
    srcs = [
        "coil_20_data.py",
    ],
    deps = [
        "//inception:dataset",
    ],
)
