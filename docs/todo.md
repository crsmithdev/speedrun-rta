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
updated: 2026-10-21
priority: medium
labels: [airbench, modal]
---

Modal gives other A100 and A10 variants in some calls, and the cache misses on each of them (airbench README, "What you ask for is not always what you get").
Also record which device fills each cache.
