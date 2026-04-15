"""
Regenerate summary_results.csv from eval_results.json files in results/,
then plot overall accuracy vs communication round for all methods.
"""
import csv
import json
import os

import matplotlib.pyplot as plt

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR  = os.path.join(BASE_DIR, "results")
CSV_PATH     = os.path.join(BASE_DIR, "summary_results.csv")
PLOT_PATH    = os.path.join(BASE_DIR, "federated_accuracy_comparison.png")
NUM_ROUNDS   = 10

# ── source mapping: folder → display label ─────────────────────────────────────
# Folders that exist on disk are loaded from eval_results.json.
# Rows without a folder (hetlora) fall back to the existing CSV.
CONFIGS = [
    ("fed_sim_rank4",   "Fixed Rank 4"),
    ("fed_sim_rank16",  "Fixed Rank 16"),
    ("fed_sim_rank64",  "Fixed Rank 64"),
    ("fed_sim_flexlora","FlexLoRA"),
    ("fed_sim_hetlora", "HetLoRA"),
]

COLORS = {
    "fed_sim_rank4":    "#95a5a6",
    "fed_sim_rank16":   "#34495e",
    "fed_sim_rank64":   "#e74c3c",
    "fed_sim_flexlora": "#2ecc71",
    "fed_sim_hetlora":  "#f1c40f",
}
MARKERS = {
    "fed_sim_rank4":    "o",
    "fed_sim_rank16":   "s",
    "fed_sim_rank64":   "D",
    "fed_sim_flexlora": "^",
    "fed_sim_hetlora":  "v",
}


# ── helpers ────────────────────────────────────────────────────────────────────

def load_from_folder(config):
    """Load per-round metrics from eval_results.json files."""
    rows = []
    folder = os.path.join(RESULTS_DIR, config)
    for r in range(NUM_ROUNDS):
        path = os.path.join(folder, f"round_{r}", "eval_results.json")
        if not os.path.exists(path):
            continue
        try:
            d    = json.load(open(path))
            s    = d.get("summary", d)
            rows.append({
                "config":     config,
                "round":      r,
                "yn_acc":     round(s.get("yn_accuracy",        0.0), 4),
                "open_em":    round(s.get("open_exact_match",   0.0), 4),
                "open_bleu":  round(s.get("open_bleu1",         0.0), 4),
                "overall_acc":round(s.get("overall_accuracy",   0.0), 4),
            })
        except Exception as e:
            print(f"  Warning: could not read {path}: {e}")
    return rows


def load_from_csv(config, csv_path=CSV_PATH):
    """Fall back to rows already saved in summary_results.csv."""
    rows = []
    if not os.path.exists(csv_path):
        return rows
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["config"] == config:
                rows.append({
                    "config":     config,
                    "round":      int(row["round"]),
                    "yn_acc":     float(row["yn_acc"]),
                    "open_em":    float(row["open_em"]),
                    "open_bleu":  float(row["open_bleu"]),
                    "overall_acc":float(row["overall_acc"]),
                })
    return rows


# ── build dataset ──────────────────────────────────────────────────────────────

def build_data():
    all_rows = []
    for config, label in CONFIGS:
        folder = os.path.join(RESULTS_DIR, config)
        if os.path.isdir(folder):
            rows = load_from_folder(config)
            src  = "folder"
        else:
            rows = load_from_csv(config)
            src  = "csv (no folder)"
        print(f"  {label:18s}: {len(rows):2d} rounds  [{src}]")
        all_rows.extend(rows)
    return all_rows


# ── write CSV ──────────────────────────────────────────────────────────────────

def write_csv(rows):
    fieldnames = ["config", "round", "yn_acc", "open_em", "open_bleu", "overall_acc"]
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"CSV saved to {CSV_PATH}")


# ── plot ───────────────────────────────────────────────────────────────────────

def plot(rows):
    plt.rcParams.update({
        "font.size":        14,
        "axes.labelsize":   16,
        "axes.titlesize":   20,
        "xtick.labelsize":  12,
        "ytick.labelsize":  12,
        "legend.fontsize":  12,
        "figure.titlesize": 22,
    })

    fig, ax = plt.subplots(figsize=(12, 7.5))

    plotted_accs = []
    for config, label in CONFIGS:
        subset = sorted(
            [r for r in rows if r["config"] == config],
            key=lambda r: r["round"],
        )
        if not subset:
            print(f"  Skipping {label} (no data)")
            continue

        rounds = [r["round"]      for r in subset]
        accs   = [r["overall_acc"] for r in subset]
        plotted_accs.extend(accs)

        ax.plot(
            rounds, accs,
            label=label,
            marker=MARKERS[config],
            markersize=9,
            linewidth=2.5,
            color=COLORS[config],
            alpha=0.9,
        )
        # Annotate final value
        ax.annotate(
            f"{accs[-1]:.3f}",
            xy=(rounds[-1], accs[-1]),
            xytext=(4, 2), textcoords="offset points",
            fontsize=10, color=COLORS[config],
        )

    ax.set_title("Federated PathVQA Accuracy Comparison",
                 fontsize=20, pad=20, fontweight="bold")
    ax.set_xlabel("Communication Round", fontsize=16)
    ax.set_ylabel("Overall Accuracy",    fontsize=16)
    ax.set_xticks(range(NUM_ROUNDS))

    if plotted_accs:
        ax.set_ylim(
            max(0.0, min(plotted_accs) - 0.02),
            min(1.0, max(plotted_accs) + 0.04),
        )

    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left",
              frameon=True, fontsize=12)

    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {PLOT_PATH}")


# ── main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading results …")
    rows = build_data()
    write_csv(rows)
    plot(rows)

    # Print summary table
    print("\nFinal-round summary (round 9):")
    print(f"  {'Method':20s}  {'Overall':>8}  {'Y/N':>8}  {'Open EM':>8}")
    print("  " + "-" * 48)
    for config, label in CONFIGS:
        subset = [r for r in rows if r["config"] == config and r["round"] == 9]
        if subset:
            r = subset[0]
            print(f"  {label:20s}  {r['overall_acc']:8.4f}  {r['yn_acc']:8.4f}  {r['open_em']:8.4f}")
        else:
            print(f"  {label:20s}  {'N/A':>8}")
