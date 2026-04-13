import csv
import matplotlib.pyplot as plt
import numpy as np

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
        'fed_sim_flexlora': 'FlexLoRA (Heterogeneous Rank 4-64)',
        'fed_sim_rank4': 'Fixed Rank 4 (Baseline)',
        'fed_sim_rank16': 'Fixed Rank 16 (Baseline)',
        'fed_sim_rank64': 'Fixed Rank 64 (Ceiling)',
        'fed_sim_hetlora': 'HetLoRA (Self-Pruning Ranks)'
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

    plt.title('Federated PathVQA Accuracy Comparison\n(Impact of Heterogeneous Rank & Algorithmic Fixes)', 
              fontsize=20, pad=20, fontweight='bold')
    plt.xlabel('Communication Round', fontsize=16)
    plt.ylabel('Overall Accuracy', fontsize=16)
    
    plt.xticks(range(10))
    plt.ylim(0.44, 0.65)
    
    # Formatting grid and legend
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, fontsize=12)
    
    # Add annotations for key findings
    # Adjust annotation positions based on new data
    plt.annotate('Rank 64 Ceiling (Best Performance)', 
                 xy=(8, 0.607), xytext=(5, 0.64),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                 fontsize=12, fontweight='bold')
                 
    plt.annotate('FlexLoRA (Surpassing Rank 4)', 
                 xy=(7, 0.574), xytext=(3, 0.59),
                 arrowprops=dict(facecolor='#27ae60', shrink=0.05, width=1, headwidth=8),
                 fontsize=12, color='#1e8449', fontweight='bold')

    plt.tight_layout()
    plt.savefig('federated_accuracy_comparison.png', dpi=300)
    print("Plot saved to federated_accuracy_comparison.png")

if __name__ == "__main__":
    plot_results()
