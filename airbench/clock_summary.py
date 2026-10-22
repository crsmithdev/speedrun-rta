"""Summarize results/clock.jsonl: median seconds per phase and dollars to 94%, per GPU.

Rows group by the device that the call got, which is not always the type it asked
for, because the compile cache is keyed by the exact device.

    uv run python airbench/clock_summary.py airbench/results/clock.jsonl
"""

import json
import statistics
import sys
from collections import defaultdict

from modal_lib import PRICE


def phases(r: dict) -> dict:
    wall = statistics.mean(x["wall"] for x in r["runs"])
    to_first = r["t_warm"] - r["t_start"] + r["runs"][0]["wall"]
    return dict(
        start=r["t_start"] - r["t_call"],
        setup=r["t_cuda"] - r["t_start"],
        unpack=r["t_copy"] - r["t_enter"],
        hits=r["inductor"].get("fxgraph_cache_hit", 0),
        compile=r["t_warm"] - r["t_cuda"],
        train=statistics.mean(x["train"] for x in r["runs"]),
        wall=wall,
        acc=statistics.mean(x["acc"] for x in r["runs"]),
        first=to_first,
        hundred=r["t_warm"] - r["t_start"] + 100 * wall,
    )


def main(path: str) -> None:
    groups, errors = defaultdict(list), []
    for line in open(path):
        r = json.loads(line)
        if "error" in r:
            errors.append(r)
        else:
            groups[(r["gpu"], r["gpu_name"], r["cached"])].append(phases(r))

    print("| asked for | got | cache | n | call to start, min / median (s) | setup (s) | of it, cache unpack (s) | FX graph hits | compile (s) | train (s) | wall per run (s) "
          "| acc | first result (s) | $ first | $ 100 runs |")
    print("| - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |")
    order = list(PRICE)
    for (gpu, name, cached), ps in sorted(groups.items(), key=lambda kv: (order.index(kv[0][0]), kv[0][1], kv[0][2])):
        m = {k: statistics.median(p[k] for p in ps) for k in ps[0]}
        name = name.removeprefix("NVIDIA ").replace(" Blackwell Server Edition", "")
        print(f"| {gpu} | {name} | {'warm' if cached else 'cold'} | {len(ps)} | {min(p['start'] for p in ps):.1f} / {m['start']:.1f} | {m['setup']:.1f} | {m['unpack']:.1f} | {m['hits']:.0f} "
              f"| {m['compile']:.1f} | {m['train']:.3f} | {m['wall']:.2f} | {100 * m['acc']:.2f}% "
              f"| {m['first']:.1f} | {m['first'] * PRICE[gpu]:.4f} | {m['hundred'] * PRICE[gpu]:.3f} |")
    for r in errors:
        print(f"error {r['gpu']} cached={r['cached']}: {r['error'][:200]}")


if __name__ == "__main__":
    main(sys.argv[1])
