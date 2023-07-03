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


"""

conda activate diffusers
cd /home/xander/Projects/cog/diffusers/eden/templates
python test.py


"""

ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
ENV_VARS_TRUE_AND_AUTO_VALUES = ENV_VARS_TRUE_VALUES.union({"AUTO"})

USE_TF = os.environ.get("USE_TF", "AUTO").upper()
USE_TORCH = os.environ.get("USE_TORCH", "AUTO").upper()
USE_JAX = os.environ.get("USE_FLAX", "AUTO").upper()
USE_SAFETENSORS = os.environ.get("USE_SAFETENSORS", "AUTO").upper()

STR_OPERATION_TO_FUNC = {">": op.gt, ">=": op.ge, "==": op.eq, "!=": op.ne, "<=": op.le, "<": op.lt}

_torch_version = "N/A"
if USE_TORCH in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TF not in ENV_VARS_TRUE_VALUES:
    _torch_available = importlib.util.find_spec("torch") is not None
    if _torch_available:
        try:
            _torch_version = importlib_metadata.version("torch")
            print(f"PyTorch version {_torch_version} available.")

            # check if cuda is available:
            _torch_cuda_available = importlib.util.find_spec("torch.cuda") is not None
            if _torch_cuda_available:
                print("CUDA available")

            import torch
            print("PyTorch version:")
            print(torch.__version__)
            print("Cuda is_available: ", torch.cuda.is_available())
            print("Cuda version:")
            print(torch.version.cuda)
        except importlib_metadata.PackageNotFoundError:
            _torch_available = False
else:
    print("Disabling PyTorch because USE_TORCH is set")
    _torch_available = False

# The package importlib_metadata is in a different place, depending on the python version.
if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata

_xformers_available = importlib.util.find_spec("xformers") is not None
try:
    _xformers_version = importlib_metadata.version("xformers")
    if _torch_available:
        import torch

        if version.Version(torch.__version__) < version.Version("1.12"):
            raise ValueError("PyTorch should be >= 1.12")
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
