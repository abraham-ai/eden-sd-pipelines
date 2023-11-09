import sys
print(sys.version)

import importlib.util
import operator as op
import os
import sys
from collections import OrderedDict
from typing import Union

from huggingface_hub.utils import is_jinja_available  # noqa: F401
from packaging import version
from packaging.version import Version, parse

# The package importlib_metadata is in a different place, depending on the python version.
if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata

try:
    _torch_version = importlib_metadata.version("torch")
    print(f"PyTorch version {_torch_version} available.")

    # check if cuda is available:
    _torch_cuda_available = importlib.util.find_spec("torch.cuda") is not None
    if _torch_cuda_available:
        print("CUDA is available!")

    import torch
    print("PyTorch version:")
    print(torch.__version__)
    print("Cuda is_available: ", torch.cuda.is_available())
    print("Cuda version:")
    print(torch.version.cuda)
except importlib_metadata.PackageNotFoundError:
    _torch_available = False

_xformers_available = importlib.util.find_spec("xformers") is not None
try:
    _xformers_version = importlib_metadata.version("xformers")
    print(f"Successfully imported xformers version {_xformers_version}")
except importlib_metadata.PackageNotFoundError:
    _xformers_available = False
    print("xformers is not installed")

print("#####################################################################################")
print("#####################################################################################")
print("#####################################################################################")
print("#####################################################################################")

# Test tensorflow:
import tensorflow as tf
is_gpu_available = tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)

print("Successfully imported tensorflow version ", tf.__version__)
print("TF is_gpu_available:", str(is_gpu_available))

# Checks and prints out detailed info about GPUs
gpus = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(gpus))
for gpu in gpus:
    print("Name:", gpu.name, "  Type:", gpu.device_type)


if tf.config.list_physical_devices('GPU'):
    print("TensorFlow is running on GPU.")
else:
    print("TensorFlow is NOT running on GPU.")

# Create random tensors
a = tf.random.normal([1000, 1000])
b = tf.random.normal([1000, 1000])

# Perform matrix multiplication
with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
    c = tf.matmul(a, b)

# Verify device
print("Tensor multiplication performed on:", c.device)