"""The honest clock: time and cost to 94% on CIFAR-10 from a cold start, per Modal GPU.

Each call runs in a fresh container and records wall-clock time per phase:
call to container start, import torch, CUDA init, the compile warmup, and each
training run of airbench94_muon.py (unchanged). The cached variant keeps the
Inductor and Triton caches on a Volume per GPU type, as one tar archive: the
cache is about 5,000 small files, and the Volume takes about 0.4 s per file.

    uv run modal run airbench/clock.py
    uv run modal run airbench/clock.py --gpus L4,H100 --repeats 1
"""

import time

T_START = time.time()  # the container imports this module when it starts

import concurrent.futures
import contextlib
import io
import json
import os
import pathlib
import subprocess
import sys

import modal

HERE = pathlib.Path(__file__).parent
# The T4 is left out: the unchanged script runs out of its 16 GB (results/clock_T4_oom.txt).
GPUS = ["L4", "A10", "L40S", "A100-40GB", "A100-80GB", "RTX-PRO-6000", "H100", "H200", "B200"]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "torch==2.11.0",
        "torchvision==0.26.0",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .add_local_dir(HERE, "/root/airbench", ignore=["cifar10", "logs", "results", "__pycache__"])
    .add_local_file(HERE / "cifar10" / "train.pt", "/root/airbench/cifar10/train.pt")
    .add_local_file(HERE / "cifar10" / "test.pt", "/root/airbench/cifar10/test.pt")
)

app = modal.App("airbench-clock", image=image)


def cache_volume(gpu: str) -> modal.Volume:
    return modal.Volume.from_name(f"airbench-cache-{gpu.lower()}", create_if_missing=True)


@app.function(timeout=3600, single_use_containers=True)
def clock(gpu: str, cached: bool, runs: int, fill: bool = False) -> dict:
    import shutil

    t_enter = time.time()
    if cached:
        if os.path.exists("/cache/cache.tar"):
            shutil.unpack_archive("/cache/cache.tar", "/tmp/cache")
        # Inductor and Triton read these when torch is imported.
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/cache/inductor"
        os.environ["TRITON_CACHE_DIR"] = "/tmp/cache/triton"
    t_copy = time.time()

    import torch
    t_torch = time.time()
    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    t_cuda = time.time()

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
        t_warm = time.time()
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

    if fill:
        shutil.make_archive("/tmp/cache", "tar", "/tmp/cache")
        shutil.copy("/tmp/cache.tar", "/cache/cache.tar")
        cache_volume(gpu).commit()
    from torch._dynamo.utils import counters
    gpu_name = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    return dict(
        gpu_name=gpu_name, t_start=T_START, t_enter=t_enter, t_copy=t_copy, t_torch=t_torch, t_cuda=t_cuda,
        t_warm=t_warm, t_end=time.time(), runs=results,
        inductor={k: v for k, v in counters["inductor"].items() if "cache" in k},
    )


@app.local_entrypoint()
def main(gpus: str = ",".join(GPUS), repeats: int = 3, runs: int = 5, concurrency: int = 8):
    gpus = gpus.split(",")
    out = HERE / "results" / "clock.jsonl"
    out.parent.mkdir(exist_ok=True)

    def fn(gpu, cached):
        volumes = {"/cache": cache_volume(gpu)} if cached else {}
        return clock.with_options(gpu=gpu, volumes=volumes)

    # Fill each GPU's cache once. These calls are not measured.
    empty = [g for g in gpus if "cache.tar" not in [e.path for e in cache_volume(g).listdir("/")]]
    fills = [(g, fn(g, True).spawn(g, True, 1, fill=True)) for g in empty]
    for g, call in fills:
        try:
            call.get()
            print(f"filled cache {g}")
        except Exception as e:
            print(f"fill failed {g}: {e!r}")

    # Stay under the account's GPU limit (10), so that no call waits in a queue
    # and the call-to-start time is the cold start only.
    def measure(g, cached):
        t_call = time.time()
        try:
            return dict(gpu=g, cached=cached, t_call=t_call, **fn(g, cached).remote(g, cached, runs))
        except Exception as e:
            return dict(gpu=g, cached=cached, t_call=t_call, error=repr(e))

    jobs = [(g, cached) for _ in range(repeats) for cached in (False, True) for g in gpus]
    with out.open("a") as f, concurrent.futures.ThreadPoolExecutor(concurrency) as pool:
        for row in pool.map(lambda job: measure(*job), jobs):
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(row["gpu"], "warm" if row["cached"] else "cold",
                  row.get("error") or f"{row['t_end'] - row['t_start']:.1f}s in container")
