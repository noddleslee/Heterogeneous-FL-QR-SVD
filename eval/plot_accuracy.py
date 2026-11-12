"""
Plot per-round PathVQA overall accuracy for all FL experiments.
Usage: python plot_accuracy.py [--output path/to/output.png]
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# ── data ──────────────────────────────────────────────────────────────────────

def load_rounds(exp_name, n_rounds=10):
    rounds, accs = [], []
    for r in range(n_rounds):
        path = os.path.join(RESULTS_DIR, exp_name, f"round_{r}", "eval_results.json")
        if os.path.exists(path):
            with open(path) as f:
                d = json.load(f)
            rounds.append(r)
            accs.append(d["summary"]["overall_accuracy"])
    return rounds, accs


EXPERIMENTS = [
    # (label,                      exp_dir,                          color,     marker, linestyle, zorder)
    ("Fixed Rank 4",   "fed_sim_rank4",                   "#888888", "o",  "-",  2),
    ("Fixed Rank 16",  "fed_sim_rank16",                  "#1f3a6e", "s",  "-",  3),
    ("Fixed Rank 64",  "fed_sim_rank64",                  "#d62728", "D",  "-",  4),
    ("FlexLoRA [4,16,32,64]", "fed_sim_flexlora_4_16_32_64", "#2ca02c", "^", "-", 5),
    ("HetLoRA [4,16,32,64]",  "fed_sim_hetlora_4_16_32_64_g099", "#ff7f0e", "v", "-", 1),
]

# ── plot ──────────────────────────────────────────────────────────────────────

def make_plot(output_path):
    fig, ax = plt.subplots(figsize=(12, 7), dpi=200)

    final_points = []  # (y_value, label_text, color) for sorted right-side annotations

    for label, exp, color, marker, ls, zo in EXPERIMENTS:
        rounds, accs = load_rounds(exp)
        if not rounds:
            print(f"  [skip] {exp}: no data")
            continue

        single_point = len(rounds) == 1

        if single_point:
            # Only a final-round reference point (rank64): horizontal dashed line + star
            ax.axhline(accs[0], color=color, linestyle=":", linewidth=1.4,
                       alpha=0.6, zorder=zo)
            ax.plot(rounds[0], accs[0], color=color, marker="*",
                    markersize=14, zorder=zo + 1, label=label, linestyle="none")
        else:
            ax.plot(rounds, accs,
                    color=color, marker=marker, linestyle=ls,
                    linewidth=2.2, markersize=7, zorder=zo, label=label)

        final_points.append((accs[-1], f"{accs[-1]:.3f}", color, rounds[-1]))

    # Right-side value labels — sort by y and spread to avoid overlap
    final_points.sort(key=lambda x: x[0])
    y_vals = [p[0] for p in final_points]
    # Minimum gap between labels (in data units)
    min_gap = 0.008
    # Adjust positions upward where too close
    adjusted = list(y_vals)
    for i in range(1, len(adjusted)):
        if adjusted[i] - adjusted[i - 1] < min_gap:
            adjusted[i] = adjusted[i - 1] + min_gap

    for (y_data, text, color, x_data), y_label in zip(final_points, adjusted):
        ax.annotate(text,
                    xy=(x_data, y_data),
                    xytext=(9.35, y_label),
                    fontsize=10.5, color=color, fontweight="bold",
                    va="center",
                    arrowprops=dict(arrowstyle="-", color=color,
                                   lw=0.8, alpha=0.4) if abs(y_label - y_data) > 0.003 else None)

    # Aesthetics
    ax.set_xlabel("Communication Round", fontsize=13)
    ax.set_ylabel("Overall Accuracy", fontsize=13)
    ax.set_title("Federated PathVQA Accuracy Comparison", fontsize=15, fontweight="bold")
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(0.41, 0.645)
    ax.set_xticks(range(10))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(loc="lower right", fontsize=11, framealpha=0.92,
              bbox_to_anchor=(0.72, 0.02))

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    print(f"Saved → {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=os.path.join(
        os.path.dirname(__file__), "..", "results", "fl_accuracy_comparison.png"))
    args = parser.parse_args()
    make_plot(args.output)
