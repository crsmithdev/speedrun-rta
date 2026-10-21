"""Run an airbench script unchanged on a Modal GPU and save its output.

    uv run modal run airbench/modal_bench.py --gpu A100-40GB
    uv run modal run airbench/modal_bench.py --gpu H100 --script airbench94_muon.py
"""

import pathlib

import modal

HERE = pathlib.Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.11.0",
        "torchvision==0.26.0",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .add_local_dir(HERE, "/root/airbench", ignore=["cifar10", "logs", "results", "__pycache__"])
    # The CIFAR-10 server is slow enough to time out a run, so ship the local tensors.
    # Run any airbench script locally once to create them.
    .add_local_file(HERE / "cifar10" / "train.pt", "/root/airbench/cifar10/train.pt")
    .add_local_file(HERE / "cifar10" / "test.pt", "/root/airbench/cifar10/test.pt")
)

app = modal.App("airbench", image=image)


@app.function(timeout=3600)
def bench(script: str) -> str:
    import subprocess

    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,power.limit", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    run = subprocess.run(
        ["python", script], cwd="/root/airbench",
        capture_output=True, text=True,
    )
    return f"# gpu: {gpu}\n# script: {script}\n# exit: {run.returncode}\n{run.stdout}{run.stderr}"


@app.local_entrypoint()
def main(gpu: str = "A100-40GB", script: str = "airbench94_muon.py"):
    out = bench.with_options(gpu=gpu).remote(script)
    path = HERE / "results" / f"{pathlib.Path(script).stem}_{gpu}.log"
    path.parent.mkdir(exist_ok=True)
    path.write_text(out)
    print(out.splitlines()[0])
    print(f"saved {path}")
