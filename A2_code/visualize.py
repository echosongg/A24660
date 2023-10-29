import matplotlib.pyplot as plt
import numpy as np

# Data
methods = ['LSTM', 'CNN', 'RandomForest', 'Logistic Regression', 'SVM']
metrics = {
    "Accuracy": [94.87, 46.74, 71.00, 44.03, 42.90],
    "Precision": [96.08, 48.21, 70.87, 43.20, 48.72],
    "Recall": [93.33, 45.90, 70.67, 41.91, 40.54],
    "F1 Score": [94.21, 44.31, 70.74, 37.89, 34.56],
    "Geometric Mean": [84.59, 52.33, 71.59, 49.28, 47.92]
}

# Convert percentages to fractions
for metric, values in metrics.items():
    metrics[metric] = [v / 100 for v in values]

# Visualization
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(methods))
bar_width = 0.15

for idx, (metric_name, values) in enumerate(metrics.items()):
    ax.bar(x + idx * bar_width, values, bar_width, label=metric_name, alpha=0.85)

ax.set_title("Performance Metrics Comparison")
ax.set_xticks(x + bar_width * (len(metrics) - 1) / 2)
ax.set_xticklabels(methods)
ax.set_ylabel("Score")
ax.set_ylim([0, 1])
ax.legend(loc="upper left")
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.set_axisbelow(True)
plt.tight_layout()
plt.show()
