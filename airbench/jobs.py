"""Run an airbench script unchanged on a Modal GPU, from the recipes in recipes.json.

    uv run mlib deploy
    uv run mlib run muon-a100 --wait
    uv run mlib run ablate --set env.ABLATE=whiten
    uv run mlib pull <id>

Each run saves the script's output as log.txt in its folder, and its mean
accuracy and train time in record.json.
"""

import modal_lib as ml  # first: its import time is the container start

import pathlib

import modal

HERE = pathlib.Path(__file__).parent
APP = "airbench-jobs"

image = (
    ml.torch_image()
    .add_local_dir(HERE, "/root/airbench", ignore=["cifar10", "logs", "results", "__pycache__"])
    # The CIFAR-10 server is slow enough to time out a run, so ship the local tensors.
    # Run any airbench script locally once to create them.
    .add_local_file(HERE / "cifar10" / "train.pt", "/root/airbench/cifar10/train.pt")
    .add_local_file(HERE / "cifar10" / "test.pt", "/root/airbench/cifar10/test.pt")
)

app = modal.App(APP, image=image)


@app.function(timeout=3600, single_use_containers=True)
def bench(recipe: dict, run_id: str) -> dict:
    import statistics
    import subprocess
    import sys

    with ml.Run(APP, run_id, recipe) as run:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,power.limit", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        out = subprocess.run(["python", recipe["script"]], cwd="/root/airbench",
                             capture_output=True, text=True)
        run.clock.mark("script")
        log = f"# gpu: {gpu}\n# script: {recipe['script']}\n# exit: {out.returncode}\n{out.stdout}{out.stderr}"
        (run.dir / "log.txt").write_text(log)

        sys.path.insert(0, "/root/airbench")
        from summarize import runs
        rs = runs(log)
        if out.returncode or not rs:
            raise RuntimeError(f"{recipe['script']} exited {out.returncode}; see log.txt")
        run.result = dict(runs=len(rs), acc=statistics.mean(a for a, _ in rs),
                          train=statistics.mean(t for _, t in rs))
    return run.record
