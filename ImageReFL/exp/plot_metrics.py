#!/usr/bin/env python3
"""
Parse metric blocks from a log file and plot each metric vs epoch.

Expected lines look like:
    epoch          : 8
    HPS            : 0.34033203125
    loss           : -0.00034027099609375
    grad_norm      : 0.0005510378279723227
    test_ClipScore : 0.299140625
    test_PickScore : 22.125
    test_ImageReward: 0.958671875
    test_HPS       : 0.3626953125
"""

import argparse
import os
import re
from typing import List, Dict, Any

import matplotlib.pyplot as plt

LINE_RE = re.compile(r"^\s*([A-Za-z0-9_\.]+)\s*:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*$")


def parse_blocks(path: str) -> List[Dict[str, Any]]:
    """
    Read the file and return a list of dicts, one per block.
    A new block starts at each 'epoch :' line. Keys/values are numeric.
    """
    blocks: List[Dict[str, Any]] = []
    cur: Dict[str, Any] = {}

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            m = LINE_RE.match(line)
            if not m:
                continue

            key, val = m.group(1), m.group(2)

            # convert numbers (epoch as int if it looks integral)
            if key.lower() == "epoch":
                # if we already have content, start a new block
                if cur:
                    blocks.append(cur)
                    cur = {}
                try:
                    # allow floats but cast to int if integral
                    v = float(val)
                    cur[key] = int(v) if abs(v - int(v)) < 1e-9 else v
                except ValueError:
                    # fallback to raw
                    cur[key] = val
            else:
                try:
                    v = float(val)
                except ValueError:
                    v = val
                cur[key] = v

    # flush last block
    if cur:
        blocks.append(cur)

    # only keep blocks that have an epoch and at least one metric
    blocks = [b for b in blocks if "epoch" in b and any(k != "epoch" for k in b.keys())]
    return blocks


def collect_series(blocks: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[float]]]:
    """
    Turn blocks into series per metric.
    Returns: {metric: {"epoch": [...], "value": [...]}}
    """
    # sort blocks by epoch
    blocks_sorted = sorted(blocks, key=lambda b: b["epoch"])
    series: Dict[str, Dict[str, List[float]]] = {}

    # enumerate all metrics that appear at least once (excluding 'epoch')
    metrics = set()
    for b in blocks_sorted:
        for k in b.keys():
            if k != "epoch":
                metrics.add(k)

    # build epoch/value arrays, skipping missing values for a metric
    for m in sorted(metrics):
        epochs: List[float] = []
        values: List[float] = []
        for b in blocks_sorted:
            if m in b:
                epochs.append(b["epoch"])
                values.append(b[m])
        if epochs:
            series[m] = {"epoch": epochs, "value": values}

    return series


def plot_series(series: Dict[str, Dict[str, List[float]]], outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    for metric, data in series.items():
        epochs = data["epoch"]
        values = data["value"]

        plt.figure()  # one chart per figure
        plt.plot(epochs, values, marker="o")
        plt.title(metric)
        plt.xlabel("epoch")
        plt.ylabel(metric)
        plt.grid(True, linestyle="--", alpha=0.4)
        # sanitize filename
        safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in metric)
        outpath = os.path.join(outdir, f"{safe_name}.png")
        plt.tight_layout()
        plt.savefig(outpath, dpi=150)
        plt.close()


def main():
    ap = argparse.ArgumentParser(description="Extract metric blocks and plot per metric.")
    ap.add_argument("-i", "--input", default="log.txt", help="Path to log file (default: log.txt)")
    ap.add_argument("-o", "--outdir", default="plots", help="Directory to save plots (default: plots)")
    args = ap.parse_args()

    blocks = parse_blocks(args.input)
    if not blocks:
        print("No metric blocks found. Check your log format.")
        return

    series = collect_series(blocks)
    if not series:
        print("No metrics found to plot.")
        return

    plot_series(series, args.outdir)
    print(f"Saved {len(series)} plots to: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
