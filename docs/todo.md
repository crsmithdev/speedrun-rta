# To do

Things to build or look at in speedrun-rta. `aleph todo` owns this file.
Item numbers never change: commits, the spec and the vault quote them.

## 1. GPT-2 speedrun on Modal: reproduce modded-nanogpt to 3.28 val loss, then real time and cost
---
id: 1
status: open
created: 2026-10-21
updated: 2026-10-21
priority: high
labels: [project, modal]
---

Next project; Chris agreed on 2026-10-21. Same format as airbench:
- Reproduce unchanged on 8x H100.
- Run several seeds and report P(hit 3.28).
- Measure real time: cold start and compile.
- Compare dollars to target on 8x H100, 8x A100, and 1x H100 with gradient accumulation.
Give it its own pyproject.toml if it needs a different torch.

## 2. GRPO on Qwen2.5-1.5B on Modal: two reward designs at one budget
---
id: 2
status: open
created: 2026-10-21
updated: 2026-10-21
priority: medium
labels: [project, modal]
---

Compare binary and partial-credit rewards on GSM8K or Countdown at the same budget.
Report reward curves, output length, and before and after accuracy.
Log each case of reward hacking.

## 3. DDP to FSDP2 on 2 to 4 Modal GPUs, with a rank killed mid-run
---
id: 3
status: open
created: 2026-10-21
updated: 2026-10-21
priority: medium
labels: [project, modal]
---

Measure MFU, peak memory per GPU, exposed communication time, and loss parity against 1 GPU.
Then kill one rank. Measure recovery time, and lost work against checkpoint overhead as the checkpoint interval changes.

## 4. Find why Modal call-to-start is 17 to 90 s in some airbench clock calls
---
id: 4
status: open
created: 2026-10-21
updated: 2026-10-21
priority: medium
labels: [airbench, modal]
---

Call-to-start was either 2 to 6 s or 17 to 90 s, with 8 concurrent calls against a 10-GPU account limit.
One possible cause: containers that are still stopping count against the limit.
Test with concurrency 4 or with spaced calls. Data: airbench/results/clock.jsonl.

## 5. Key the airbench compile cache by nvidia-smi device name
---
id: 5
status: open
created: 2026-10-21
updated: 2026-10-22
priority: medium
labels: [airbench, modal]
---

Modal gives other A100 and A10 variants in some calls, and the cache misses on each of them (airbench README, "What you ask for is not always what you get").
Also record which device fills each cache.

### Notes

- 2026-10-22 03:08: The compile cache now lives in modal-lib (modal_lib/cache.py, cache_volume(app, gpu)); key it by device there.

## 6. airbench: split the 0.13-point run-to-run spread by source
---
id: 6
status: open
created: 2026-10-21
updated: 2026-10-21
priority: medium
labels: [airbench]
---

Fix the seed for the init, then for the data order, then turn on deterministic kernels.
Measure the spread left at each step, and check whether one seed gives bitwise-identical runs. Runs locally on the 5070.

## 7. airbench: sweep the SGD learning rate at 8 epochs for a fair Muon ablation
---
id: 7
status: open
created: 2026-10-21
updated: 2026-10-21
priority: medium
labels: [airbench]
---

The muon ablation row (80.81%) has no tuning. A small learning-rate and momentum sweep would turn it into a real comparison at equal epochs.

## 8. modal-lib: one [tool.modal-lib] per repo; the GPT-2 project needs a second app
---
id: 8
status: open
created: 2026-10-22
updated: 2026-10-22
priority: medium
labels: []
---

mlib reads one app, file and recipes path from [tool.modal-lib] in the repo's pyproject.toml.
speedrun-rta will hold airbench and the GPT-2 speedrun (#1), each with its own app and image.
Options: a table per project ([tool.modal-lib.airbench]) with `mlib -p airbench`, or `app`/`file` per recipe.
Decide when #1 starts.

### Notes

- 2026-10-22 09:06: modal-lib v2 (be47c5c) reads modal.json from the current directory, so GPT-2 can keep its own app and image in its own folder (e.g. gpt2/modal.json) and run mlib there.

## 9. modal-lib: test the reused-container path of Run
---
id: 9
status: open
created: 2026-10-22
updated: 2026-10-22
priority: medium
labels: []
---

Run marks `reused: true` when a container serves a second run, and `mlib runs` then prints the run time from t_enter.
Every job uses single_use_containers=True, so no run has taken that path yet. Untested.
