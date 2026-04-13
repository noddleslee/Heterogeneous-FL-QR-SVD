import os
import json
import csv

def collect_results():
    base_path = "results"
    configs = ["fed_sim_flexlora", "fed_sim_rank4", "fed_sim_rank16", "fed_sim_rank64", "fed_sim_hetlora"]
    rounds = range(10)
    
    # Load existing data first
    existing_data = {} # (config, round) -> row
    csv_path = "summary_results.csv"
    fieldnames = ["config", "round", "yn_acc", "open_em", "open_bleu", "overall_acc"]
    
    if os.path.exists(csv_path):
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Convert types
                    row['round'] = int(row['round'])
                    row['yn_acc'] = float(row['yn_acc']) if row['yn_acc'] else None
                    row['open_em'] = float(row['open_em']) if row['open_em'] else None
                    row['open_bleu'] = float(row['open_bleu']) if row['open_bleu'] else None
                    row['overall_acc'] = float(row['overall_acc']) if row['overall_acc'] else None
                    existing_data[(row['config'], row['round'])] = row
            print(f"Loaded {len(existing_data)} existing entries from {csv_path}")
        except Exception as e:
            print(f"Warning: Could not read existing {csv_path}: {e}")

    # Scan for new data
    new_data_count = 0
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
                        # Update or add
                        existing_data[(config, r)] = metrics
                        new_data_count += 1
                except Exception as e:
                    print(f"Error reading {json_path}: {e}")
                
    if not existing_data:
        print("No data found!")
        return

    # Sort and save
    sorted_data = [existing_data[k] for k in sorted(existing_data.keys())]
    
    with open(csv_path, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted_data)
    
    print(f"Results saved to {csv_path} (updated {new_data_count} entries)")
    
    # Print a text table for quick view
    pivot_data = {} # {round: {config: overall_acc}}
    all_configs = sorted(list(set(d['config'] for d in sorted_data)))
    all_rounds = sorted(list(set(d['round'] for d in sorted_data)))
    
    for d in sorted_data:
        r = d['round']
        cfg = d['config']
        if r not in pivot_data:
            pivot_data[r] = {}
        pivot_data[r][cfg] = d['overall_acc']
    
    header = "Round | " + " | ".join([f"{c:16}" for c in all_configs])
    print("\n" + header)
    print("-" * len(header))
    for r in all_rounds:
        row = f"{r:5} | "
        cols = []
        for cfg in all_configs:
            val = pivot_data.get(r, {}).get(cfg, "N/A")
            if isinstance(val, float):
                cols.append(f"{val:16.4f}")
            else:
                cols.append(f"{str(val):16}")
        print(row + " | ".join(cols))

if __name__ == "__main__":
    collect_results()
