# airbench: CIFAR-10 to 94%

The target is 94% mean test accuracy on CIFAR-10 in the shortest training
time. The training script is `airbench94_muon.py` from
[KellerJordan/cifar10-airbench](https://github.com/KellerJordan/cifar10-airbench)
(MIT, see `LICENSE`), unchanged. Upstream reports 94.01% in 2.59 s on a
400 W A100 with torch 2.4.1.

All runs here use torch 2.11.0 with CUDA 12.8. Accuracy is with test-time
augmentation (TTA). Time is the training time that the script measures
with CUDA events, mean over the runs; it excludes `torch.compile`.

## Reproduction on three GPUs

| GPU | Runs | Mean acc | Std (points) | P(acc ≥ 94%) | Time (s) |
| - | - | - | - | - | - |
| H100 80GB, 700 W (Modal) | 200 | 93.99% | 0.12 | 0.44 | 1.342 |
| A100 40GB (Modal) | 200 | 94.01% | 0.12 | 0.55 | 2.650 |
| RTX 5070 12GB, 250 W (local) | 200 | 94.00% | 0.13 | 0.53 | 6.912 |

The A100 time is 2.3% above the upstream figure.

## Real time: cold start to 94% on Modal

In speedrunning, in-game time leaves out the loading screens and real time
does not. The training time above is in-game time. `clock.py` measures real
time: each call starts a fresh Modal container and records wall-clock time
for each phase until the result. `rta.py` has the clock, the compile cache
and the GPU prices.

- **Compile:** the warmup run of `torch.compile(mode="max-autotune")`.
- **Train:** the script's own training time for one run.
- **First result:** container start to the first 94% result. Modal bills
  this span.
- **Dollars:** the first result, and 100 runs in one container, at the GPU
  prices on modal.com/pricing on 2026-10-21. CPU and memory are left out.
- **Cold:** an empty compile cache.
- **Warm:** the Inductor and Triton caches from an earlier call on that GPU
  type, unpacked from a Modal Volume.

Medians of 3 calls, 5 runs per call:

| GPU | Cache | Compile (s) | Train (s) | First result (s) | $ first | $ 100 runs |
| - | - | - | - | - | - | - |
| L4 | cold | 214.2 | 10.99 | 230.0 | 0.051 | 0.313 |
| L4 | warm | 60.8 | 10.16 | 78.2 | 0.017 | 0.257 |
| A10 | warm | 40.8 | 7.11 | 52.4 | 0.016 | 0.247 |
| L40S | cold | 165.0 | 3.00 | 170.2 | 0.092 | 0.266 |
| L40S | warm | 36.2 | 2.99 | 46.1 | 0.025 | 0.200 |
| A100-SXM4-40GB | cold | 165.9 | 2.76 | 171.2 | 0.100 | 0.270 |
| RTX PRO 6000 | cold | 108.7 | 1.74 | 112.0 | 0.094 | 0.250 |
| RTX PRO 6000 | warm | 31.6 | 1.75 | 36.3 | 0.031 | 0.188 |
| H100 | cold | 127.1 | 1.33 | 132.1 | 0.145 | 0.308 |
| H100 | warm | 31.1 | 1.33 | 38.4 | 0.042 | 0.205 |
| H200 | cold | 125.8 | 1.25 | 129.6 | 0.163 | 0.341 |
| H200 | warm | 27.9 | 1.24 | 34.4 | 0.043 | 0.218 |
| B200 | cold | 76.6 | 0.93 | 79.1 | 0.137 | 0.312 |
| B200 | warm | 22.1 | 0.93 | 31.7 | 0.055 | 0.237 |

Every cell has a mean accuracy between 93.92% and 94.19%.
`results/clock.md` has every cell with all phases.

What the table shows:

- **Compile is most of the real time.** From cold, the compile takes 77 to
  214 s and the training takes 0.9 to 11 s. The training time that
  speedruns report is 1% to 5% of the real time to the first result.
- **The compile cache saves more than a faster GPU.** A warm cache cuts the
  compile by 3.4 to 4.6 times. A warm L40S gets its first result in 46 s,
  faster than a cold B200 at 79 s, for 18% of the cost.
- **The slowest GPUs are the cheapest for one result.** A warm A10 or L4
  costs under $0.02. A cold H200 costs $0.16 for the same result.
- **The cost of 100 runs is nearly flat across GPUs.** It is $0.19 to
  $0.34 for every type. A faster GPU costs more per second, and it finishes
  the training in fewer seconds by about the same ratio. The warm RTX PRO
  6000 is the cheapest at $0.19.

### What you ask for is not always what you get

Modal does not always give the device that the call asks for. Calls for
`A100-40GB` got an A100-SXM4-80GB in 2 of 6 calls. Calls for `A100-80GB`
got an SXM4 card in 3 of 6 calls and a PCIe card in the rest. Calls for
`A10` got an A10G in 2 of 6 calls.

The compile cache is specific to the device. Each warm call hit all 10 FX
graphs or missed all 10. Every miss was on a different device from the one
that most calls of that type got. The sweep did not record which device
filled each cache, so this is the pattern, not a direct check:

| Asked for | Got | Cache | FX graph hits | Compile (s) | First result (s) |
| - | - | - | - | - | - |
| A100-40GB | A100-SXM4-40GB | warm | 10 | 48.5 | 59.1 |
| A100-40GB | A100-SXM4-80GB | warm (median of 2) | 0 | 140.9 | 151.8 |
| A100-80GB | A100 80GB PCIe | warm (median of 2) | 10 | 28.7 | 36.1 |
| A100-80GB | A100-SXM4-80GB | warm | 0 | 110.4 | 119.4 |

To keep a warm cache on these types, key the cache by the device name, not
by the type that the call asks for.

### Two problems on the way

- **A cache on the Volume, file by file, was slower than no cache.** The
  cache is 5,165 files (170 MB). The Volume read them at about 0.39 s per
  file, so the copy took 2,009 s on an L4. The cache now goes to the Volume
  as one tar file, and the unpack takes 1.3 to 4.7 s.
- **The T4 does not run the unchanged script.** It runs out of its 16 GB
  during the compile warmup, with 12.5 GB in CUDA graph memory pools
  (`results/clock_T4_oom.txt`).

### What this does not show

- **Few calls:** each cell has 3 calls or fewer, and the device split
  leaves some cells with 1.
- **Start time:** the time from the call to the container start was
  either 2 to 6 s or 17 to 90 s. I did not find the cause. The account
  limit is 10 GPUs and the sweep ran 8 calls at a time; containers that are
  still stopping can count against the limit. This span is not billed and
  is not in the table.
- **Prices change:** the dollar columns use prices from one date.
- **Container size:** the CPU and memory size of the container can change
  the compile time, and the sweep used Modal's defaults.

## Ablations on the RTX 5070

`airbench94_ablate.py` is the same script with switches. Each row turns one
trick off and keeps the rest. Fifty runs per row. The standard error of a
difference between two rows is about 0.025 points.

| Switch | What changes | Mean acc | Δ (points) |
| - | - | - | - |
| none | baseline | 94.02% | |
| `whiten` | the first conv keeps its random init, not the patch-whitening init | 93.22% | −0.80 |
| `altflip` | random flips each epoch, not flips that alternate per epoch | 93.86% | −0.16 |
| `dirac` | conv layers do not start as identity | 93.89% | −0.13 |
| `smoothing` | no label smoothing (0.2 in the baseline) | 93.91% | −0.11 |
| `bnmom` | BatchNorm running-stat momentum 0.1, not 0.4 | 93.98% | −0.04 |
| `muon` | normalized SGD at Muon's learning rate | 80.81% | −13.21 |
| TTA off | evaluate without flip and translate views (200 runs) | 93.18% | −0.82 |

The TTA row comes from the `val_acc` column of the baseline log.

The `muon` row does not measure what Muon is worth. The learning rate and
momentum are tuned for Muon, and the swap gets no tuning. The fair
comparison is upstream's earlier record with tuned SGD
(`airbench94_compiled.py`, unchanged, 9.9 epochs, not 8):

| Optimizer | Runs | Mean acc | Time (s) |
| - | - | - | - |
| Muon (`airbench94_muon.py`) | 200 | 94.00% | 6.912 |
| tuned SGD (`airbench94_compiled.py`) | 25 | 93.99% | 7.914 |

At equal accuracy Muon is 13% faster on the 5070. Upstream reports 16% on
an A100.

## What this does not show

- Local time is noisy between batches. The baseline row in the ablation
  set took 7.400 s, against 6.912 s for the same code in the 200-run set.
  Time differences below about 7% on the 5070 are not meaningful, and the
  ablation table compares accuracy only.
- Each ablation turns one trick off from the full recipe. It does not show
  what each trick adds to a weaker recipe, and the switches can interact.
- No row was retuned. A retuned recipe can recover part of each loss, so
  each delta is an upper bound on what the trick is worth.

## Run it

On the local GPU:

```sh
cd airbench
uv run python airbench94_muon.py                    # 200 runs
ABLATE=whiten TRIALS=50 uv run python airbench94_ablate.py
uv run python summarize.py results/*.log
```

On Modal, from the repo root, with [modal-lib](https://github.com/crsmithdev/modal-lib)
and the recipes in `modal.json` (smoke, muon-a100, muon-h100, ablate). The
runs need `cifar10/*.pt` from a local run first.

```sh
uv run mlib run muon-h100 --wait
uv run mlib run ablate --set env.ABLATE=whiten
uv run mlib ps                                       # status, times and dollars
uv run mlib pull <id>                                # to runs/<id>: log.txt, record.json, gpu.csv
uv run python airbench/summarize.py runs/<id>/log.txt
uv run modal run airbench/clock.py                   # the real-time sweep
```
