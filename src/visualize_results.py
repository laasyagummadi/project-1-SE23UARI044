"""
visualize_results.py
---------------------
Generates plots from the pipeline's evaluation logs and summary JSON.

Outputs (saved to results/):
  - bar_chart_rouge.png     : ROUGE-1, ROUGE-2, ROUGE-L per variant
  - bar_chart_bleu.png      : BLEU per variant
  - heatmap_metrics.png     : All metrics × variants heatmap
  - long_vs_short.png       : Chunk vs. direct for long docs

Usage
-----
    python src/visualize_results.py --summary results/summary_<run_id>.json
"""

import argparse
import json
import os
import random

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

random.seed(42)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# Color palette
PALETTE = {
    "baseline":   "#8ecae6",
    "role_only":  "#219ebc",
    "cot_only":   "#023047",
    "no_fillers": "#ffb703",
    "full":       "#fb8500",
    "chunk_full": "#d62828",
}
DEFAULT_COLOR = "#adb5bd"


def load_summary(path: str) -> list:
    with open(path) as f:
        data = json.load(f)
    return data["results"]


def bar_chart(rows, metric_keys, title, out_path):
    variants = [r["variant"] for r in rows]
    x = np.arange(len(variants))
    width = 0.22
    n = len(metric_keys)
    fig, ax = plt.subplots(figsize=(max(8, len(variants) * 1.2), 5))

    for i, mk in enumerate(metric_keys):
        vals = [r[mk] for r in rows]
        colors = [PALETTE.get(v, DEFAULT_COLOR) for v in variants]
        offset = (i - n / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=mk.replace("avg_", "").upper(), alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, min(1.0, max(r[metric_keys[0]] for r in rows) * 1.35))
    ax.set_ylabel("Score")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Plot] Saved → {out_path}")


def heatmap(rows, out_path):
    variants = [r["variant"] for r in rows]
    metrics = ["avg_rouge1", "avg_rouge2", "avg_rougeL", "avg_bleu"]
    data = np.array([[r[m] for m in metrics] for r in rows])

    fig, ax = plt.subplots(figsize=(7, max(3, len(variants) * 0.6 + 1)))
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=0.5)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([m.replace("avg_", "").upper() for m in metrics], fontsize=10)
    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels(variants, fontsize=9)

    for i in range(len(variants)):
        for j in range(len(metrics)):
            ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center",
                    fontsize=9, color="black" if data[i, j] < 0.35 else "white")

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    ax.set_title("Metric Heatmap: All Variants", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Plot] Saved → {out_path}")


def long_vs_short_chart(log_path: str, out_path: str):
    """
    Read per-record log and compare 'full' vs 'chunk_full' on long docs.
    """
    if not os.path.exists(log_path):
        print(f"[Plot] Log not found: {log_path}, skipping long_vs_short chart.")
        return

    full_long, full_short, chunk_long = [], [], []
    with open(log_path) as f:
        for line in f:
            r = json.loads(line.strip())
            if r.get("variant") == "full":
                if r.get("is_long"):
                    full_long.append(r["rougeL"])
                else:
                    full_short.append(r["rougeL"])
            elif r.get("variant") == "chunk_full":
                chunk_long.append(r["rougeL"])

    if not full_long or not chunk_long:
        print("[Plot] Not enough data for long_vs_short chart.")
        return

    labels = ["Full (short docs)", "Full (long docs)", "Chunk+Full (long docs)"]
    means  = [np.mean(full_short), np.mean(full_long), np.mean(chunk_long)]
    stds   = [np.std(full_short),  np.std(full_long),  np.std(chunk_long)]
    colors = ["#219ebc", "#ffb703", "#d62828"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, means, yerr=stds, capsize=5, color=colors, alpha=0.85, width=0.5)
    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{mean:.3f}", ha="center", va="bottom", fontsize=10,
        )
    ax.set_ylabel("ROUGE-L (mean ± std)")
    ax.set_title("Chunk Processing vs. Direct (Long Docs)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(means) * 1.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Plot] Saved → {out_path}")


def main(summary_path: str, log_path: str = None):
    rows = load_summary(summary_path)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ROUGE bar chart
    bar_chart(
        rows,
        metric_keys=["avg_rouge1", "avg_rouge2", "avg_rougeL"],
        title="ROUGE Scores by Prompt Variant",
        out_path=os.path.join(RESULTS_DIR, "bar_chart_rouge.png"),
    )

    # BLEU bar chart
    bar_chart(
        rows,
        metric_keys=["avg_bleu"],
        title="BLEU Score by Prompt Variant",
        out_path=os.path.join(RESULTS_DIR, "bar_chart_bleu.png"),
    )

    # Heatmap
    heatmap(rows, os.path.join(RESULTS_DIR, "heatmap_metrics.png"))

    # Long vs short (if log available)
    if log_path:
        long_vs_short_chart(log_path, os.path.join(RESULTS_DIR, "long_vs_short.png"))
    else:
        # try to find latest log
        logs = [f for f in os.listdir(RESULTS_DIR) if f.startswith("eval_log_")]
        if logs:
            latest_log = os.path.join(RESULTS_DIR, sorted(logs)[-1])
            long_vs_short_chart(latest_log, os.path.join(RESULTS_DIR, "long_vs_short.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, help="Path to summary_<run_id>.json")
    parser.add_argument("--log",     default=None,  help="Path to eval_log_<run_id>.jsonl")
    args = parser.parse_args()
    main(args.summary, args.log)
