"""Summarize airbench logs: TTA accuracy and train time over the timed runs.

    uv run python airbench/summarize.py airbench/results/*.log
"""

import statistics
import sys


def runs(text: str) -> list[tuple[float, float]]:
    """Return (tta_val_acc, time_seconds) for each run except the warmup."""
    out, head = [], None
    for line in text.splitlines():
        cols = [c.strip() for c in line.strip().strip("|").split("|")]
        if head is None and "tta_val_acc" in cols:
            head = cols
        elif head and len(cols) == len(head) and cols[head.index("epoch")] == "eval":
            out.append((float(cols[head.index("tta_val_acc")]), float(cols[-1])))
    return out[1:]


def main(paths: list[str]) -> None:
    print("| log | runs | acc mean | acc std | P(acc>=94%) | time mean (s) | time median (s) |")
    print("| - | - | - | - | - | - | - |")
    for path in paths:
        rs = runs(open(path).read())
        accs = [a for a, _ in rs]
        times = [t for _, t in rs]
        hit = sum(a >= 0.94 for a in accs) / len(accs)
        print(f"| {path.rsplit('/', 1)[-1]} | {len(rs)} | {100 * statistics.mean(accs):.2f}% | "
              f"{100 * statistics.stdev(accs):.2f} | {hit:.2f} | "
              f"{statistics.mean(times):.3f} | {statistics.median(times):.3f} |")


if __name__ == "__main__":
    main(sys.argv[1:])
