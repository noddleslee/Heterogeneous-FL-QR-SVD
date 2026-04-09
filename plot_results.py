import matplotlib.pyplot as plt
import json
import os

# Data
rounds = list(range(10))
flexlora = [0.492, 0.494, 0.538, 0.514, 0.512, 0.560, 0.558, 0.548, 0.550, 0.548]
rank4 = [0.452, 0.470, 0.520, 0.524, 0.516, 0.558, 0.544, 0.544, 0.546, 0.558]
rank16 = [0.458, 0.522, 0.542, 0.550, 0.552, 0.588, 0.594, 0.582, 0.598, 0.584]

plt.figure(figsize=(10, 6))

# Plotting
plt.plot(rounds, flexlora, marker='o', label='FlexLoRA (Heterogeneous Ranks 4-64)', linewidth=2, color='#1f77b4')
plt.plot(rounds, rank16, marker='s', label='Fixed Rank 16', linewidth=2, color='#ff7f0e')
plt.plot(rounds, rank4, marker='^', label='Fixed Rank 4', linewidth=2, color='#2ca02c')

# Styling
plt.title('Federated PathVQA: Accuracy vs Rounds', fontsize=14, fontweight='bold')
plt.xlabel('Communication Round', fontsize=12)
plt.ylabel('Overall Accuracy', fontsize=12)
plt.xticks(rounds)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10)

# Highlighting best results
best_val = max(max(rank16), max(flexlora), max(rank4))
best_round = rank16.index(best_val)
plt.annotate(f'Best: {best_val:.3f}', 
             xy=(best_round, best_val), 
             xytext=(best_round-1, best_val+0.01),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

plt.tight_layout()
plt.savefig('federated_accuracy_comparison.png', dpi=300)
print("Plot saved to federated_accuracy_comparison.png")
