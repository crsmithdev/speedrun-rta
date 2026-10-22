"""The image for the airbench recipes in modal.json."""

import pathlib

import modal

HERE = pathlib.Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.11.0", "torchvision==0.26.0", index_url="https://download.pytorch.org/whl/cu128")
    .workdir("/root/airbench")
    .add_local_dir(HERE, "/root/airbench", ignore=["cifar10", "logs", "results", "runs", "__pycache__"])
    # The CIFAR-10 server is slow enough to time out a run, so ship the local tensors.
    # Run any airbench script locally once to create them.
    .add_local_file(HERE / "cifar10" / "train.pt", "/root/airbench/cifar10/train.pt")
    .add_local_file(HERE / "cifar10" / "test.pt", "/root/airbench/cifar10/test.pt")
)
