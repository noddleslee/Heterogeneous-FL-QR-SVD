import os
import re
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def parse_eval_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    metrics = {}
    patterns = {
        'yes_no_acc': r'Yes/No accuracy\s*:\s*([\d.]+)',
        'open_match': r'Open exact match\s*:\s*([\d.]+)',
        'open_bleu': r'Open BLEU-1\s*:\s*([\d.]+)',
        'overall_acc': r'Overall accuracy\s*:\s*([\d.]+)'
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            metrics[key] = float(match.group(1))
    
    return metrics

def analyze():
    configs = [
        {
            'dir': str(Path(__file__).resolve().parents[1] / 'results' / 'runs' / 'my_fedavg'),
            'type': 'fixed',
            'rounds': ['round_0', 'round_1', 'round_2']
        },
        {
            'dir': str(Path(__file__).resolve().parents[1] / 'results' / 'runs' / 'my_flexlora'),
            'type': 'flexlora',
            'rounds': ['round_0', 'round_1', 'round_2']
        }
    ]
    
    data = []

    for config in configs:
        base_dir = config['dir']
        for r in config['rounds']:
            round_dir = os.path.join(base_dir, r)
            if not os.path.exists(round_dir):
                continue
                
            eval_files = [f for f in os.listdir(round_dir) if f.startswith('eval_') and f.endswith('.out')]
            # Sort by modification time
            eval_files.sort(key=lambda x: os.path.getmtime(os.path.join(round_dir, x)))
            
            if config['type'] == 'fixed':
                if len(eval_files) < 2:
                    continue
                # Rank 16 is first (earlier), Rank 4 is second (later)
                for i, rank in enumerate([16, 4]):
                    res = parse_eval_file(os.path.join(round_dir, eval_files[i]))
                    if res:
                        res.update({'round': r, 'rank': rank, 'label': f'LoRA Rank {rank}'})
                        data.append(res)
            else:
                # FlexLoRA: usually only one successful eval per round that we care about
                # But let's check for the largest success
                best_res = None
                for f in eval_files:
                    res = parse_eval_file(os.path.join(round_dir, f))
                    if res and (not best_res or res['overall_acc'] > best_res['overall_acc']):
                        best_res = res
                
                if best_res:
                    best_res.update({'round': r, 'rank': 'Flex', 'label': 'FlexLoRA'})
                    data.append(best_res)

    df = pd.DataFrame(data)
    print(df)
    
    # Visualization
    metrics_to_plot = ['yes_no_acc', 'open_match', 'open_bleu', 'overall_acc']
    metric_names = ['Yes/No Accuracy', 'Open Exact Match', 'Open BLEU-1', 'Overall Accuracy']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Federated Learning Performance: Rank 4 vs 16 vs FlexLoRA', fontsize=18)
    
    rounds_labels = ['Round 0', 'Round 1', 'Round 2']
    
    labels_to_plot = ['LoRA Rank 4', 'LoRA Rank 16', 'FlexLoRA']
    colors = {'LoRA Rank 4': '#e74c3c', 'LoRA Rank 16': '#3498db', 'FlexLoRA': '#2ecc71'}
    markers = {'LoRA Rank 4': 'o', 'LoRA Rank 16': 's', 'FlexLoRA': '^'}

    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i//2, i%2]
        
        for label in labels_to_plot:
            subset = df[df['label'] == label].sort_values('round')
            if not subset.empty:
                ax.plot(subset['round'].apply(lambda x: x.replace('_', ' ').title()), 
                        subset[metric], 
                        marker=markers[label], 
                        label=label, 
                        linewidth=2.5, 
                        color=colors[label])
        
        ax.set_title(metric_names[i], fontsize=14)
        ax.set_ylabel('Score')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # Adjust y-limits
        if metric in ['open_match', 'open_bleu']:
             ax.set_ylim(0, 0.45)
        else:
             ax.set_ylim(0.4, 0.9)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_dir = str(Path(__file__).resolve().parents[1] / 'results' / 'figures')
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'performance_comparison.png')
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved to {plot_path}")
    
    # Save CSV for the report table
    csv_path = os.path.join(output_dir, 'metrics.csv')
    df.to_csv(csv_path, index=False)
    print(f"Data saved to {csv_path}")

if __name__ == '__main__':
    analyze()
