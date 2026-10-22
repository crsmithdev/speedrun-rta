"""Make the SVG figures for the airbench write-up from the committed results.

    uv run python airbench/figures.py <out_dir>

The SVGs take their colours from the page's CSS variables (--text, --muted,
--accent, --head, --rule), so they follow the page's light and dark themes.
"""

import json
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).parent
RESULTS = HERE / "results"

# The rows of the real-time table in README.md: (asked for, got, warm cache).
CLOCK_ROWS = [
    ("L4", "L4", False), ("L4", "L4", True),
    ("A10", "A10", True),
    ("L40S", "L40S", False), ("L40S", "L40S", True),
    ("A100-40GB", "A100-SXM4-40GB", False),
    ("RTX-PRO-6000", "RTX PRO 6000 Blackwell Server Edition", False),
    ("RTX-PRO-6000", "RTX PRO 6000 Blackwell Server Edition", True),
    ("H100", "H100 80GB HBM3", False), ("H100", "H100 80GB HBM3", True),
    ("H200", "H200", False), ("H200", "H200", True),
    ("B200", "B200", False), ("B200", "B200", True),
]
NAMES = {"A100-40GB": "A100 40GB", "RTX-PRO-6000": "RTX PRO 6000"}


def eval_rows(path: pathlib.Path) -> list[tuple[float, float]]:
    """(val_acc, tta_val_acc) for each run except the warmup."""
    out = []
    for line in path.read_text().splitlines():
        cols = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cols) == 6 and cols[1] == "eval":
            out.append((float(cols[3]), float(cols[4])))
    return out[1:]


def ablations() -> list[tuple[str, float]]:
    base = statistics.mean(t for _, t in eval_rows(RESULTS / "ablate_none_RTX5070.log"))
    rows = [(label, 100 * (statistics.mean(t for _, t in eval_rows(RESULTS / f"ablate_{name}_RTX5070.log")) - base))
            for name, label in [("whiten", "Patch-whitening init"), ("altflip", "Alternating flip"),
                                ("dirac", "Identity init"), ("smoothing", "Label smoothing"),
                                ("bnmom", "BatchNorm momentum")]]
    tta = eval_rows(RESULTS / "airbench94_muon_RTX5070.log")
    rows.append(("Test-time augmentation", 100 * (statistics.mean(v for v, _ in tta) - statistics.mean(t for _, t in tta))))
    return sorted(rows, key=lambda r: r[1])


def clock() -> list[dict]:
    calls = [json.loads(line) for line in (RESULTS / "clock.jsonl").read_text().splitlines()]
    out = []
    for asked, got, cached in CLOCK_ROWS:
        cell = [c for c in calls if c["gpu"] == asked and c["gpu_name"] == f"NVIDIA {got}" and c["cached"] == cached]
        # Draw the call with the median time to the first result, so that the
        # segments add up to the median in the README. Each cell has 3 calls.
        c = sorted(cell, key=lambda c: c["t_warm"] - c["t_start"] + c["runs"][0]["wall"])[len(cell) // 2]
        out.append(dict(
            label=f"{NAMES.get(asked, asked)} {'warm' if cached else 'cold'}",
            setup=c["t_cuda"] - c["t_start"],
            compile=c["t_warm"] - c["t_cuda"],
            train=c["runs"][0]["wall"],
        ))
    return out


def svg(width: int, height: int, body: list[str], title: str) -> str:
    return "\n".join([
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="{title}" '
        'xmlns="http://www.w3.org/2000/svg" style="font: inherit; font-size: 13px; max-width: 100%">',
        f"<title>{title}</title>", *body, "</svg>", ""])


def text(x, y, s, anchor="start", var="--text"):
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}" style="fill: var({var})">{s}</text>'


def rect(x, y, w, h, var):
    return f'<rect x="{x:.1f}" y="{y}" width="{max(w, 0.5):.1f}" height="{h}" style="fill: var({var})"/>'


def ablation_svg(rows: list[tuple[str, float]]) -> str:
    left, width, row, bar = 190, 640, 26, 16
    scale = (width - left - 70) / max(-d for _, d in rows)
    body = []
    for i, (label, delta) in enumerate(rows):
        y = 10 + i * row
        body.append(text(left - 10, y + 12, label, "end"))
        body.append(rect(left, y, -delta * scale, bar, "--accent"))
        body.append(text(left + -delta * scale + 6, y + 12, f"−{-delta:.2f}", var="--muted"))
    return svg(width, 10 + len(rows) * row + 4, body, "Accuracy lost when each trick is turned off, in points")


def clock_svg(rows: list[dict]) -> str:
    left, width, row, bar = 150, 640, 24, 14
    top = 30
    scale = (width - left - 60) / max(r["setup"] + r["compile"] + r["train"] for r in rows)
    body = []
    x = left
    for label, var in [("setup", "--muted"), ("compile", "--accent"), ("train", "--head")]:
        body.append(rect(x, 4, 10, 10, var))
        body.append(text(x + 15, 13, label))
        x += 15 + len(label) * 8 + 20
    for i, r in enumerate(rows):
        y = top + i * row
        body.append(text(left - 10, y + 11, r["label"], "end"))
        x = left
        for key, var in [("setup", "--muted"), ("compile", "--accent"), ("train", "--head")]:
            body.append(rect(x, y, r[key] * scale, bar, var))
            x += r[key] * scale
        body.append(text(x + 6, y + 11, f"{r['setup'] + r['compile'] + r['train']:.0f} s", var="--muted"))
    return svg(width, top + len(rows) * row + 4, body, "Seconds from container start to the first 94% result")


def main(out: str) -> None:
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ablations.svg").write_text(ablation_svg(ablations()))
    (out / "clock.svg").write_text(clock_svg(clock()))
    for r in ablations():
        print(f"{r[0]}: {r[1]:+.2f}")
    for r in clock():
        print(f"{r['label']}: setup {r['setup']:.1f} compile {r['compile']:.1f} train {r['train']:.2f}")


if __name__ == "__main__":
    main(sys.argv[1])
