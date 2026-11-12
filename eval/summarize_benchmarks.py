"""Recompute the supplied five-run benchmark means and Student t intervals.

Uses only the Python standard library. Original run JSON files are read-only.
"""
import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path

T975_DF4 = 2.7764451051977987
DEFAULT_DIR = Path(__file__).resolve().parents[1] / "results" / "benchmarks"


def summarize(directory):
    groups = {}
    for path in sorted(Path(directory).glob("*_run[1-5].json")):
        match = re.fullmatch(r"(.+)_run([1-5])\.json", path.name)
        if not match:
            continue
        prefix, run = match.groups()
        data = json.loads(path.read_text())
        for key, value in data["results"].items():
            groups.setdefault((prefix, key), []).append((int(run), value))
    rows = []
    for (prefix, key), runs in sorted(groups.items()):
        if sorted(run for run, _ in runs) != [1, 2, 3, 4, 5]:
            raise ValueError(f"Expected runs 1-5 for {prefix}:{key}")
        configs = [value["config"] for _, value in runs]
        if any(config != configs[0] for config in configs):
            raise ValueError(f"Configuration changed between runs for {prefix}:{key}")
        row = {"benchmark": prefix, "config_key": key, **configs[0], "n_runs": 5}
        if isinstance(row.get("ranks"), list):
            row["ranks"] = json.dumps(row["ranks"])
        for method in ("flexlora", "qrsvd", "hetlora"):
            values = [value[f"{method}_total_s"] for _, value in runs if f"{method}_total_s" in value]
            if not values:
                continue
            if len(values) != 5:
                raise ValueError(f"Missing {method} time for {prefix}:{key}")
            mean, sd = statistics.mean(values), statistics.stdev(values)
            half = T975_DF4 * sd / math.sqrt(5)
            row.update({f"{method}_mean_s": mean, f"{method}_sd_s": sd, f"{method}_ci95_low_s": mean - half, f"{method}_ci95_high_s": mean + half})
        if "flexlora_mean_s" in row and "qrsvd_mean_s" in row:
            row["speedup_ratio_of_means"] = row["flexlora_mean_s"] / row["qrsvd_mean_s"]
            row["speedup_mean_of_ratios"] = statistics.mean(value["flexlora_total_s"] / value["qrsvd_total_s"] for _, value in runs)
        rows.append(row)
    if not rows:
        raise ValueError(f"No benchmark run JSON found in {directory}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark_dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = summarize(args.benchmark_dir)
    output = args.output or args.benchmark_dir / "summary_all.csv"
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summarized {len(rows)} configurations, five runs each: {output}")


if __name__ == "__main__":
    main()
