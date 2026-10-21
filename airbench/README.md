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

```sh
cd airbench
uv run python airbench94_muon.py                    # 200 runs on the local GPU
ABLATE=whiten TRIALS=50 uv run python airbench94_ablate.py
uv run modal run modal_bench.py --gpu H100          # needs cifar10/*.pt from a local run
uv run python summarize.py results/*.log
```
