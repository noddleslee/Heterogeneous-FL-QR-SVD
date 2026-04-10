import os
import json
import csv

def collect_results():
    base_path = "results"
    configs = ["fed_sim_flexlora", "fed_sim_rank4", "fed_sim_rank16", "fed_sim_hetlora"]
    rounds = range(10)
    
    data = []
    
    for config in configs:
        for r in rounds:
            json_path = os.path.join(base_path, config, f"round_{r}", "eval_results.json")
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r') as f:
                        content = json.load(f)
                        summary = content.get("summary", {})
                        metrics = {
                            "config": config,
                            "round": r,
                            "yn_acc": summary.get("yn_accuracy"),
                            "open_em": summary.get("open_exact_match"),
                            "open_bleu": summary.get("open_bleu1"),
                            "overall_acc": summary.get("overall_accuracy")
                        }
                        data.append(metrics)
                except Exception as e:
                    print(f"Error reading {json_path}: {e}")
            else:
                # print(f"Warning: {json_path} not found")
                pass
                
    if not data:
        print("No data found!")
        return

    fieldnames = ["config", "round", "yn_acc", "open_em", "open_bleu", "overall_acc"]
    with open("summary_results.csv", "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    
    print("Results saved to summary_results.csv")
    
    # Print a text table for quick view
    # Pivot logic without pandas
    pivot_data = {} # {round: {config: overall_acc}}
    all_configs = sorted(list(set(d['config'] for d in data)))
    all_rounds = sorted(list(set(d['round'] for d in data)))
    
    for d in data:
        r = d['round']
        cfg = d['config']
        if r not in pivot_data:
            pivot_data[r] = {}
        pivot_data[r][cfg] = d['overall_acc']
    
    header = "Round | " + " | ".join(all_configs)
    print("\n" + header)
    print("-" * len(header))
    for r in all_rounds:
        row = f"{r:5} | "
        cols = []
        for cfg in all_configs:
            val = pivot_data.get(r, {}).get(cfg, "N/A")
            if isinstance(val, float):
                cols.append(f"{val:.4f}")
            else:
                cols.append(str(val))
        print(row + " | ".join(cols))

if __name__ == "__main__":
    collect_results()
