import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Finalized Academic Color Palette (Balanced Academic)
_COLORS = ["#2266AA", "#DD5533", "#339955", "#CC5599"]

def apply_academic_style():
    """Apply professional academic plotting style."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "figure.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

def _load_results(run_dir: Path):
    results_path = run_dir / "results.json"
    if not results_path.exists():
        return None, None
    with open(results_path) as f:
        results = json.load(f)
    return results, results.get("train_loss_curve", [])


def _load_eval(run_dir: Path, epoch: int | None = None):
    """Load the eval JSON. epoch=1 picks the first checkpoint, epoch=2 the last."""
    eval_files = sorted(run_dir.glob("pathvqa_eval_checkpoint-*.json"),
                        key=lambda p: int(p.stem.split("-")[-1]))
    if not eval_files:
        legacy = run_dir / "pathvqa_eval.json"
        if legacy.exists():
            eval_files = [legacy]
    if not eval_files:
        return None
    if epoch == 1:
        chosen = eval_files[0]
    else:
        chosen = eval_files[-1]
    with open(chosen) as f:
        data = json.load(f)
    return data.get("summary", data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranks",          type=int, nargs="+", default=[4, 16, 32, 64])
    parser.add_argument("--epochs",         type=int, default=2)
    parser.add_argument("--compare_epochs", action="store_true",
                        help="Show epoch1 vs epoch2 side by side")
    parser.add_argument("--out_dir",        type=str, default=None)
    args = parser.parse_args()

    apply_academic_style()
    colors = _COLORS

    out_dir = Path(args.out_dir) if args.out_dir else _RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    loss_data  = {}
    eval_data  = {}

    epoch_list = [1, 2] if args.compare_epochs else [args.epochs]

    print(f"\nLoading results from {_RESULTS_DIR}")
    for rank in args.ranks:
        run_dir = _RESULTS_DIR / f"{args.epochs}epoch_rank{rank}"
        results, loss_curve = _load_results(run_dir)
        if results is None:
            print(f"  rank {rank:>2}: no results.json — skipping")
            continue
        loss_data[rank] = loss_curve
        ev = _load_eval(run_dir, epoch=args.epochs if not args.compare_epochs else 2)
        if ev is None:
            print(f"  rank {rank:>2}: training done, eval pending")
        else:
            eval_data[rank] = ev
        n_epochs = results.get("num_epochs", args.epochs)
        batch    = results.get("batch_size", "?")
        final_loss = loss_curve[-1]["loss"] if loss_curve else float("nan")
        ckpts = results.get("checkpoints", [results.get("final_checkpoint", "?")])
        print(f"  rank {rank:>2}: {n_epochs} epochs, batch={batch}, "
              f"final_loss={final_loss:.4f}, checkpoints={len(ckpts)}")

    if not loss_data and not eval_data:
        print("\nNo data found. Run training jobs first.")
        return

    # ── load epoch1 eval if compare_epochs ────────────────────────────────────
    eval_data_ep1 = {}
    if args.compare_epochs:
        for rank in args.ranks:
            run_dir = _RESULTS_DIR / f"{args.epochs}epoch_rank{rank}"
            ev1 = _load_eval(run_dir, epoch=1)
            if ev1:
                eval_data_ep1[rank] = ev1

    # ── plot ───────────────────────────────────────────────────────────────────
    metrics = ["yn_accuracy", "open_exact_match", "open_bleu1", "overall_accuracy"]
    labels  = ["YN Acc", "Open EM", "BLEU-1", "Overall"]

    if args.compare_epochs:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), gridspec_kw={'width_ratios': [0.7, 1, 1]})
        ax_loss, ax_ep1, ax_ep2 = axes
    else:
        has_eval = bool(eval_data)
        ncols = 2 if has_eval else 1
        width_ratios = [0.7, 1] if ncols == 2 else [1]
        fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5), gridspec_kw={'width_ratios': width_ratios})
        if ncols == 1:
            axes = [axes]
        ax_loss = axes[0]

    # Loss curves
    steps_limit = 154 if (args.epochs == 2 and not args.compare_epochs and epoch_list == [1]) else None
    for i, (rank, curve) in enumerate(sorted(loss_data.items())):
        if not curve:
            continue
        pts = [pt for pt in curve if steps_limit is None or pt["step"] <= steps_limit]
        steps  = [pt["step"] for pt in pts]
        losses = [pt["loss"] for pt in pts]
        
        color = colors[i % len(colors)]
        ax_loss.plot(steps, losses, label=f"Rank {rank}",
                     color=color, alpha=0.8)

    ax_loss.set_xlabel("")
    ax_loss.set_ylabel("Training Loss")
    ax_loss.set_title("Training Loss")
    ax_loss.legend(frameon=True)

    def _draw_acc_bars(ax, edata, title):
        if not edata:
            ax.set_title(title + " (no data)")
            return
        x = np.arange(len(labels))
        width_per_bar = 0.8 / len(edata)
        gap_factor = 0.95  # Narrower gaps between bars
        bar_width = width_per_bar * gap_factor
        
        sorted_ranks = sorted(edata)
        for i, rank in enumerate(sorted_ranks):
            ev = edata[rank]
            vals = [ev.get(m, 0) for m in metrics]
            offset = (i - len(edata) / 2 + 0.5) * width_per_bar
            color = colors[i % len(colors)]
            bars = ax.bar(x + offset, vals, bar_width,
                          label=f"Rank {rank}",
                          color=color, alpha=0.85)
        
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Score")
        ax.set_title(title)
        ax.legend(frameon=True, loc="upper right")
        ax.set_ylim(0, 1.1)

    if args.compare_epochs:
        _draw_acc_bars(ax_ep1, eval_data_ep1, "Accuracy — Epoch 1")
        _draw_acc_bars(ax_ep2, eval_data,     "Accuracy — Epoch 2")
        out_suffix = "compare_epochs"
    else:
        if bool(eval_data):
            _draw_acc_bars(axes[1], eval_data,
                           f"Test Accuracy ({args.epochs} Epochs)")
        out_suffix = f"{args.epochs}epoch"

    plt.tight_layout()
    out_path = out_dir / f"rank_comparison_{out_suffix}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved → {out_path}")


if __name__ == "__main__":
    main()
