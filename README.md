# speedrun-rta

Speedrunners time a run in two ways. In-game time leaves out the loading
screens. Real-time attack (RTA) is the wall clock from start to finish,
loads included.

ML training speedruns report in-game time: the timed training loop. This
repo reruns them and reports real time too: cold start, compile, and the
dollars for the run. Each project reproduces an existing speedrun unchanged,
then measures what the record leaves out.

## Projects

- [airbench](airbench/README.md): CIFAR-10 to 94%.
  - **Reproduction:** an RTX 5070, an A100 and an H100, 200 runs each.
  - **Ablations:** one trick at a time on the RTX 5070.
  - **Real time on Modal:** nine GPU types, cold and warm. From cold, the
    compile takes 77 to 214 s, and the training takes 0.9 to 11 s.

## Setup

[uv](https://docs.astral.sh/uv/) manages the environment, and
[Modal](https://modal.com) runs the cloud GPUs.
[modal-lib](https://github.com/crsmithdev/modal-lib) runs the recipes in
`modal.json` on Modal and records the time, GPU use and cost of each run.

```sh
uv sync
uv run modal setup   # once, for the Modal runs
```

Each project's README has its commands and its results.
