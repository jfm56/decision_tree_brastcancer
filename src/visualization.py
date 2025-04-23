import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict

# 1. Pre-binning feature distribution

def plot_feature_distributions(df, feature_cols, bins=30, out_file=None):
    n = len(feature_cols)
    ncols = 5
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3, nrows*2.5))
    axes = axes.flatten()
    for i, col in enumerate(feature_cols):
        sns.histplot(df[col], bins=bins, ax=axes[i], kde=True)
        axes[i].set_title(col)
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    if out_file:
        plt.savefig(out_file)
    else:
        plt.show()

# 2. Post-binning feature counts

def plot_binned_feature_counts(df, feature_cols, out_file=None):
    n = len(feature_cols)
    ncols = 5
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3, nrows*2.5))
    axes = axes.flatten()
    for i, col in enumerate(feature_cols):
        sns.countplot(x=df[col], order=['low','medium','high'], ax=axes[i])
        axes[i].set_title(col)
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    if out_file:
        plt.savefig(out_file)
    else:
        plt.show()

# 3. Confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrix(y_true, y_pred, labels=None, out_file=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    if out_file:
        plt.savefig(out_file)
    else:
        plt.show()

# 4. Decision tree visualization (text-based)
def print_tree(node, depth=0):
    indent = '  ' * depth
    if node.is_leaf():
        print(f"{indent}Predict: {node.value}")
    else:
        for val, child in node.children.items():
            print(f"{indent}if {node.feature} == {val}:")
            print_tree(child, depth+1)

# 5. Feature importance bar plot
def plot_feature_importance(feature_importances, out_file=None):
    items = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
    features, importances = zip(*items)
    plt.figure(figsize=(8,4))
    sns.barplot(x=list(importances), y=list(features))
    plt.xlabel('Importance (Total Info Gain)')
    plt.title('Feature Importance')
    plt.tight_layout()
    if out_file:
        plt.savefig(out_file)
    else:
        plt.show()
