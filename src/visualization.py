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

# 4b. Decision tree visualization (graphical)
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

def plot_tree_graphical(node, feature_names=None, max_depth=3, out_file=None, fig_width=48, fig_height=28, xlim=44, ylim=28):
    """
    Plots the decision tree using matplotlib. Only supports categorical splits.
    max_depth: maximum depth to display (default 3 for readability)
    fig_width, fig_height: control figure size (default: very large for deep/wide trees)
    xlim, ylim: control plot limits (default: very large for deep/wide trees)
    Spacing: Increased vertical (y-2.5) and horizontal (child_dx*0.6) spacing; larger font sizes for readability.
    """
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')

    def traverse(n, x, y, dx, depth, parent_coords=None, parent_value=None):
        if max_depth is not None and depth > max_depth:
            if not n.is_leaf():
                # Indicate subtree is truncated
                ax.text(x, y-0.2, '...', ha='center', va='center', fontsize=18, color='gray')
            return
        # Draw node
        if n.is_leaf():
            label = f"Predict: {n.value}"
        else:
            feat = n.feature if feature_names is None else feature_names.get(n.feature, n.feature)
            label = f"{feat}"
        box = FancyBboxPatch((x-0.7, y-0.3), 1.4, 0.6, boxstyle="round,pad=0.1", fc="w", ec="k", lw=2)
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center', fontsize=17, zorder=10)
        # Draw edge from parent
        if parent_coords is not None and parent_value is not None:
            ax.plot([parent_coords[0], x], [parent_coords[1]-0.3, y+0.3], 'k-', lw=1.5)
            ax.text((parent_coords[0]+x)/2, (parent_coords[1]+y)/2, str(parent_value), fontsize=15, ha='center', va='center', color='blue')
        # Draw children
        if not n.is_leaf() and (max_depth is None or depth < max_depth):
            n_children = len(n.children)
            child_dx = dx / max(n_children, 1)
            start_x = x - dx/2 + child_dx/2
            for i, (val, child) in enumerate(n.children.items()):
                traverse(child, start_x + i*child_dx, y-2.5, dx=child_dx*0.6, depth=depth+1, parent_coords=(x, y), parent_value=val)
    traverse(node, x=0, y=0, dx=xlim, depth=0)
    ax.set_xlim(-xlim/2, xlim/2)
    ax.set_ylim(-ylim, 2)
    plt.tight_layout()
    if out_file:
        plt.savefig(out_file)
    else:
        plt.show()

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
