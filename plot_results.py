import matplotlib.pyplot as plt
import csv
import os

def plot_comparison():
    csv_path = 'summary_results.csv'
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Run summarize_results.py first.")
        return

    data = {} # {config: {round: accuracy}}
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cfg = row['config']
            rnd = int(row['round'])
            acc = float(row['overall_acc'])
            if cfg not in data:
                data[cfg] = {}
            data[cfg][rnd] = acc

    plt.figure(figsize=(10, 6))
    
    # Predefined styles for consistency
    styles = {
        'fed_sim_flexlora': {'label': 'FlexLoRA (Hetero 4-64)', 'marker': 'o', 'color': '#1f77b4'},
        'fed_sim_rank16':   {'label': 'Fixed Rank 16',         'marker': 's', 'color': '#ff7f0e'},
        'fed_sim_rank4':    {'label': 'Fixed Rank 4',          'marker': '^', 'color': '#2ca02c'},
        'fed_sim_hetlora':  {'label': 'HetLoRA (EMNLP 2024)',  'marker': 'D', 'color': '#d62728'},
    }

    best_val = -1
    best_pos = (0, 0)

    for cfg in sorted(data.keys()):
        rounds = sorted(data[cfg].keys())
        accs = [data[cfg][r] for r in rounds]
        
        style = styles.get(cfg, {'label': cfg, 'marker': '.', 'color': None})
        plt.plot(rounds, accs, marker=style['marker'], label=style['label'], 
                 linewidth=2, color=style['color'])
        
        # Track global best
        for r, a in zip(rounds, accs):
            if a > best_val:
                best_val = a
                best_pos = (r, a)

    # Styling
    plt.title('Federated PathVQA Performance Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Communication Round', fontsize=12)
    plt.ylabel('Overall Accuracy', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)

    # Annotation
    if best_val > 0:
        plt.annotate(f'Best: {best_val:.4f}', 
                     xy=best_pos, 
                     xytext=(best_pos[0]-0.5, best_pos[1]+0.01),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

    plt.tight_layout()
    plt.savefig('federated_accuracy_comparison.png', dpi=300)
    print("Plot saved to federated_accuracy_comparison.png")

if __name__ == "__main__":
    plot_comparison()
