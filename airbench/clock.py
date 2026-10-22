"""The honest clock: time and cost to 94% on CIFAR-10 from a cold start, per Modal GPU.

Each call runs in a fresh container, and rta.py records wall-clock time
per phase: call to container start, import torch, CUDA init, the compile
warmup, and each training run of airbench94_muon.py (unchanged). The cached
variant keeps the Inductor and Triton caches on a Volume per GPU type, as one
tar archive.

    uv run modal run airbench/clock.py
    uv run modal run airbench/clock.py --gpus L4,H100 --repeats 1
"""

import rta as ml  # first: its import time is the container start

import contextlib
import io
import os
import pathlib
import sys
import time

import modal

HERE = pathlib.Path(__file__).parent
APP = "airbench"
# The T4 is left out: the unchanged script runs out of its 16 GB (results/clock_T4_oom.txt).
GPUS = ["L4", "A10", "L40S", "A100-40GB", "A100-80GB", "RTX-PRO-6000", "H100", "H200", "B200"]

image = (
    ml.torch_image()
    .add_local_dir(HERE, "/root/airbench", ignore=["cifar10", "logs", "results", "__pycache__"])
    .add_local_file(HERE / "cifar10" / "train.pt", "/root/airbench/cifar10/train.pt")
    .add_local_file(HERE / "cifar10" / "test.pt", "/root/airbench/cifar10/test.pt")
)

app = modal.App("airbench-clock", image=image)


@app.function(timeout=3600, single_use_containers=True)
def clock(gpu: str, cached: bool, runs: int, fill: bool = False) -> dict:
    c = ml.Clock()
    c.mark("enter")
    cache = ml.compile_cache(fill=ml.cache_volume(APP, gpu) if fill else None)
    with cache if cached else contextlib.nullcontext():
        c.mark("copy")

        import torch
        c.mark("torch")
        torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
        c.mark("cuda")

        os.chdir("/root/airbench")
        sys.path.insert(0, "/root/airbench")
        sys.argv = ["airbench94_muon.py"]
        import airbench94_muon as ab

        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            model = ab.CifarNet().cuda().to(memory_format=torch.channels_last)
            model.compile(mode="max-autotune")
            ab.main("warmup", model)
            torch.cuda.synchronize()
            c.mark("warm")
            results = []
            for run in range(runs):
                t = time.time()
                acc = ab.main(run, model)
                torch.cuda.synchronize()
                results.append(dict(acc=acc, wall=time.time() - t))

        # The script's own CUDA-event training time, from each eval row except the warmup.
        timed = [float(line.strip().strip("|").split("|")[-1])
                 for line in log.getvalue().splitlines() if "|   eval  |" in line][1:]
        for r, t in zip(results, timed):
            r["train"] = t

        from torch._dynamo.utils import counters
        return c.done(runs=results, inductor={k: v for k, v in counters["inductor"].items() if "cache" in k})


@app.local_entrypoint()
def main(gpus: str = ",".join(GPUS), repeats: int = 3, runs: int = 5, concurrency: int = 8):
    gpus = gpus.split(",")

    def fn(gpu, cached):
        volumes = {ml.CACHE_MOUNT: ml.cache_volume(APP, gpu)} if cached else {}
        return clock.with_options(gpu=gpu, volumes=volumes)

    # Fill each GPU's cache once. These calls are not measured.
    empty = [g for g in gpus if not ml.cache_filled(ml.cache_volume(APP, g))]
    fills = [(g, fn(g, True).spawn(g, True, 1, fill=True)) for g in empty]
    for g, call in fills:
        try:
            call.get()
            print(f"filled cache {g}")
        except Exception as e:
            print(f"fill failed {g}: {e!r}")

    # The account's GPU limit is 10, so 8 calls at a time do not wait in a queue.
    calls = [(dict(gpu=g, cached=cached), lambda g=g, cached=cached: fn(g, cached).remote(g, cached, runs))
             for _ in range(repeats) for cached in (False, True) for g in gpus]
    for row in ml.measure(calls, HERE / "results" / "clock.jsonl", concurrency):
        print(row["gpu"], "warm" if row["cached"] else "cold",
              row.get("error") or f"{row['t_end'] - row['t_start']:.1f}s in container")
