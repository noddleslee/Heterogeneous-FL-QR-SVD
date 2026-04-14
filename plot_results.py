import csv
import matplotlib.pyplot as plt

def plot_results():
    # Setting a professional style with matplotlib defaults
    plt.rcParams.update({
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 20,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 22
    })
    
    # Load data using csv module
    data = []
    csv_path = 'summary_results.csv'
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert types
                row['round'] = int(row['round'])
                row['overall_acc'] = float(row['overall_acc']) if row['overall_acc'] else 0.0
                data.append(row)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return
    
    # Map config names to more readable ones
    name_map = {
        'fed_sim_flexlora': 'FlexLoRA',
        'fed_sim_rank4': 'Fixed Rank 4',
        'fed_sim_rank16': 'Fixed Rank 16',
        'fed_sim_rank64': 'Fixed Rank 64',
        'fed_sim_hetlora': 'HetLoRA'
    }
    
    # Create the figure
    plt.figure(figsize=(12, 7.5))
    
    # Use distinct markers and colors
    colors = {
        'fed_sim_rank4': '#95a5a6',    # Gray
        'fed_sim_rank16': '#34495e',   # Dark blue
        'fed_sim_rank64': '#e74c3c',   # Red
        'fed_sim_flexlora': '#2ecc71', # Green
        'fed_sim_hetlora': '#f1c40f'   # Yellow/Gold
    }
    
    markers = {
        'fed_sim_rank4': 'o',
        'fed_sim_rank16': 's',
        'fed_sim_rank64': 'D',
        'fed_sim_flexlora': '^',
        'fed_sim_hetlora': 'v'
    }

    plotted = []
    for config in ['fed_sim_rank4', 'fed_sim_rank16', 'fed_sim_rank64', 'fed_sim_flexlora', 'fed_sim_hetlora']:
        # Filter and sort by round
        subset = [d for d in data if d['config'] == config]
        subset.sort(key=lambda x: x['round'])
        
        if not subset:
            continue
            
        rounds = [d['round'] for d in subset]
        accs = [d['overall_acc'] for d in subset]
        
        # Plotting
        plt.plot(rounds, accs, 
                 label=name_map[config], 
                 marker=markers[config], 
                 markersize=10, 
                 linewidth=3, 
                 color=colors[config],
                 alpha=0.9)
        plotted.append((config, rounds, accs))

    plt.title('Federated PathVQA Accuracy Comparison', 
              fontsize=20, pad=20, fontweight='bold')
    plt.xlabel('Communication Round', fontsize=16)
    plt.ylabel('Overall Accuracy', fontsize=16)
    
    plt.xticks(range(10))
    if plotted:
        all_accs = [acc for _, _, accs in plotted for acc in accs]
        ymin = max(0.0, min(all_accs) - 0.02)
        ymax = min(1.0, max(all_accs) + 0.03)
        plt.ylim(ymin, ymax)
    
    # Formatting grid and legend
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, fontsize=12)

    plt.tight_layout()
    plt.savefig('federated_accuracy_comparison.png', dpi=300)
    print("Plot saved to federated_accuracy_comparison.png")

if __name__ == "__main__":
    plot_results()
