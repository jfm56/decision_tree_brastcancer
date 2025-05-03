import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
plt.ion()
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
plt.rcParams['figure.max_open_warning'] = 0
warnings.filterwarnings('ignore', category=RuntimeWarning)

# 1. Pre-binning feature distribution
def plot_top_feature_distributions(
    df, feature_importances, diagnosis_col, top_n, bins, output_prefix=None, show=True
):
    import numpy as np
    # Only use numeric features
    numeric_features = [f for f in feature_importances.keys() if pd.api.types.is_numeric_dtype(df[f])]
    skipped = set(feature_importances.keys()) - set(numeric_features)
    if skipped:
        print(f"[plot_top_feature_distributions] Skipping non-numeric features: {skipped}")
    top_features = [f for f, _ in sorted(feature_importances.items(), key=lambda x: x[1], reverse=True) if f in numeric_features][:top_n]

    """
    Plots raw KDE and boxplots, binned class proportions, and a correlation heatmap for top N features.
    """
    top_features = [f for f, _ in sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:top_n]]
    # 1. KDE plots by diagnosis
    for feat in top_features:
        # Coerce to numeric, drop NaN
        feat_data = pd.to_numeric(df[feat], errors='coerce')
        n_missing = feat_data.isna().sum()
        if n_missing > 0:
            print(f"[plot_top_feature_distributions] {feat}: {n_missing} non-numeric values converted to NaN and dropped for plotting.")
        plot_df = df.copy()
        plot_df[feat] = feat_data
        plot_df = plot_df.dropna(subset=[feat, diagnosis_col])
        plt.figure(figsize=(6, 4))
        sns.kdeplot(data=plot_df, x=feat, hue=diagnosis_col, common_norm=False, fill=True, alpha=0.4)
        plt.title(f"KDE: {feat} by {diagnosis_col}")
        plt.tight_layout()
        if output_prefix:
            plt.savefig(f"{output_prefix}_kde_{feat}.png")
        if show:
            plt.show()
            plt.close('all')
    # 2. Boxplots by diagnosis
    for feat in top_features:
        plt.figure(figsize=(6, 4))
        sns.boxplot(data=df, x=diagnosis_col, y=feat)
        plt.title(f"Boxplot: {feat} by {diagnosis_col}")
        plt.tight_layout()
        if output_prefix:
            plt.savefig(f"{output_prefix}_box_{feat}.png")
        if show:
            plt.show()
            plt.close('all')
    # 3. Binned class proportions
    for feat in top_features:
        bin_col = f"{feat}_bin"
        df[bin_col] = pd.qcut(df[feat], q=bins, duplicates='drop')
        plt.figure(figsize=(7, 4))
        sns.histplot(data=df, x=bin_col, hue=diagnosis_col, multiple='fill', shrink=0.8, stat='probability')
        plt.title(f"Class Proportion by {feat} Bins")
        plt.ylabel('Proportion')
        plt.tight_layout()
        if output_prefix:
            plt.savefig(f"{output_prefix}_bin_{feat}.png")
        if show:
            plt.show()
            plt.close('all')
    # 4. Correlation heatmap for top features + diagnosis
    corr_features = [f for f in top_features if pd.api.types.is_numeric_dtype(df[f])]
    corr_df = df[corr_features + [diagnosis_col]].copy()
    # Convert diagnosis to numeric if needed
    if not pd.api.types.is_numeric_dtype(corr_df[diagnosis_col]):
        corr_df[diagnosis_col] = corr_df[diagnosis_col].astype('category').cat.codes
    corr = corr_df.corr()
    plt.figure(figsize=(1.2*len(corr_features), 1.2*len(corr_features)))
    sns.heatmap(corr, vmin=-1, vmax=1, cmap="RdBu_r", annot=True, fmt=".2f")
    plt.title("Correlation Heatmap (Top Features)")
    plt.tight_layout()
    if output_prefix:
        plt.savefig(f"{output_prefix}_corr_heatmap.png")
    if show:
            plt.show()
            plt.close('all')

# 2. Post-binning feature counts
def plot_binned_feature_counts(
    data_frame, feature_columns, output_file=None, show=True
):
    n = len(feature_columns)
    ncols = 5
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 2.5))
    axes = axes.flatten()
    for i, col in enumerate(feature_columns):
        sns.countplot(x=data_frame[col], order=['low', 'medium', 'high'], ax=axes[i])
        axes[i].set_title(col)
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file)
    else:
        if show:
            plt.show()
            plt.close('all')

# 3. Confusion matrix
def plot_confusion_matrix(
    y_true, y_pred, labels=None, output_file=None, show=True
):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    if output_file:
        plt.savefig(output_file)
    else:
        if show:
            plt.show()
            plt.close('all')

# 4. Decision tree visualization (text-based)
def print_tree(node, depth=0):
    indent = '  ' * depth
    if node.is_leaf():
        print("  " * depth + f"{node.value}")
    else:
        for value, child in node.children.items():
            print(f"{indent}if {node.feature} == {value}:")
            print_tree(child, depth+1)

# 4b. Decision tree visualization (graphical)
def plot_tree_graphical(
    node, max_depth=3, output_file=None, fig_width=16, fig_height=8, xlim=10, ylim=5, show=True
):
    """Improved: Plots the decision tree using matplotlib with non-overlapping leaves.
    Uses a hierarchical layout approach with improved spacing.
    """
    # Create a new figure with white background
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor='white')
    ax.axis('off')
    
    # First, count nodes at each level to determine spacing
    level_counts = {}
    node_positions = {}
    
    def count_nodes_by_level(node, depth=0):
        if max_depth is not None and depth > max_depth:
            return
            
        if depth not in level_counts:
            level_counts[depth] = 0
        level_counts[depth] += 1
        
        if not node.is_leaf() and (max_depth is None or depth < max_depth):
            for child in node.children.values():
                count_nodes_by_level(child, depth+1)
    
    # Count nodes at each level
    count_nodes_by_level(node)
    
    # Calculate the width needed for each level
    max_nodes = max(level_counts.values())
    
    # Assign horizontal positions
    level_positions = {}
    
    def assign_positions(node, depth=0, pos_index=0, total_positions=1.0):
        if max_depth is not None and depth > max_depth:
            return pos_index
            
        # Calculate x position based on the node's position in its level
        if depth not in level_positions:
            level_positions[depth] = 0
            
        position = level_positions[depth]
        level_positions[depth] += 1
        
        # Calculate normalized position (0 to 1)
        x_pos = position / max(level_counts.get(depth, 1), 1)
        node_positions[node] = x_pos
        
        if node.is_leaf() or (max_depth is not None and depth >= max_depth):
            return
            
        # Process children
        children = list(node.children.items())
        for _, child in children:
            assign_positions(child, depth+1)
    
    # Assign initial positions
    assign_positions(node)
    
    # Function to adjust positions to center parents over their children
    def adjust_parent_positions(node, depth=0):
        if max_depth is not None and depth > max_depth:
            return
            
        if node.is_leaf() or (max_depth is not None and depth >= max_depth):
            return
            
        # Get children
        children = list(node.children.values())
        
        # Recursively adjust children first (bottom-up)
        for child in children:
            adjust_parent_positions(child, depth+1)
        
        # Now center this node over its children if it has any
        if children:
            child_x_positions = [node_positions[child] for child in children]
            center_pos = sum(child_x_positions) / len(child_x_positions)
            node_positions[node] = center_pos
    
    # Adjust positions to center parents
    adjust_parent_positions(node)
    
    # Now draw the tree with the calculated positions
    def draw_node(node, depth=0, parent_x=None, parent_y=None, edge_label=None):
        if max_depth is not None and depth > max_depth:
            return
            
        # Convert normalized position to plot coordinates
        x_pos = node_positions[node] * xlim
        y_pos = ylim - (depth * (ylim / (max(level_counts.keys()) + 1)))
        
        # Draw edge from parent if this isn't the root
        if parent_x is not None and parent_y is not None:
            # Draw the connecting line
            ax.plot([parent_x, x_pos], [parent_y, y_pos], 'k-', linewidth=1.0)
            
            # Add edge label with white background for readability
            if edge_label is not None:
                mid_x = (parent_x + x_pos) / 2
                mid_y = (parent_y + y_pos) / 2
                ax.text(mid_x, mid_y, str(edge_label), ha='center', va='center',
                       fontsize=10, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.1'))
        
        # Draw the node itself
        if node.is_leaf():
            # Leaf node (prediction)
            bbox_props = dict(boxstyle="round,pad=0.3", fc="lightblue", ec="blue", lw=1.5)
            ax.text(x_pos, y_pos, str(node.value), ha="center", va="center",
                   size=10, bbox=bbox_props, fontweight='bold')
        else:
            # Decision node (feature)
            bbox_props = dict(boxstyle="round,pad=0.3", fc="wheat", ec="orange", lw=1.5)
            ax.text(x_pos, y_pos, node.feature, ha="center", va="center",
                   size=10, bbox=bbox_props, fontweight='bold')
            
            # Process children
            if max_depth is None or depth < max_depth:
                for value, child in node.children.items():
                    draw_node(child, depth+1, x_pos, y_pos, value)
    
    # Draw the tree
    draw_node(node)
    
    # Add title
    plt.suptitle("Decision Tree", fontsize=14, y=0.98)
    
    # Set axis limits with a margin
    margin = 0.1
    ax.set_xlim(-margin*xlim, xlim*(1+margin))
    ax.set_ylim(-margin*ylim, ylim*(1+margin))
    
    # Save if needed
    if output_file:
        plt.savefig(output_file, bbox_inches='tight', dpi=150)
    
    # Show if requested
    if show:
        plt.show()
        
    # Don't forget to close the figure to free resources
    plt.close(fig)

# 5. Feature importance bar plot
def plot_feature_importance(
    feature_importances, output_file=None, show=True
):
    items = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:10]
    features, importances = zip(*items)


    plt.figure(figsize=(max(10, len(features) * 0.7), 4))


    plt.bar(features, importances, color='skyblue')


    plt.xlabel('Feature')


    plt.ylabel('Importance (Total Info Gain)')


    plt.title('Feature Importance')


    plt.xticks(rotation=45, ha='right', fontsize=10)


    plt.tight_layout()


    if output_file:


        plt.savefig(output_file)


    else:


        if show:
            plt.show()
            plt.close('all')


from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.table as tbl
import pandas as pd
import datetime

def visualize_top_features(
    df, feature_importances, target_col, top_n=5, bins=5, palette="Set2", save_dir=None
):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    # Only use numeric features
    numeric_feature_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    top_features = [f for f, _ in sorted(feature_importances.items(), key=lambda x: x[1], reverse=True) if f in numeric_feature_columns][:top_n]
    print("Top features:", top_features)

    for feat in top_features:
        # KDE overlay per class
        plt.figure(figsize=(7,4))
        labels = sorted(df[target_col].unique())
        colors = sns.color_palette(palette, len(labels))
        for idx, label in enumerate(labels):
            sns.kdeplot(data=df[df[target_col]==label], x=feat, color=colors[idx], lw=2, label=f"{label}")
        plt.title(f"{feat} KDE by {target_col}")
        plt.legend()
        plt.tight_layout()
        if save_dir:
            plt.savefig(f"{save_dir}/{feat}_overlay.png")
        plt.show(block=True)
        plt.close('all')

        # Boxplot
        plt.figure(figsize=(6,4))
        sns.boxplot(data=df, x=target_col, y=feat, palette=palette)
        plt.title(f"{feat} Boxplot by {target_col}")
        plt.tight_layout()
        if save_dir:
            plt.savefig(f"{save_dir}/{feat}_boxplot.png")
        plt.show()
        plt.close('all')

        # Violin plot
        plt.figure(figsize=(6,4))
        sns.violinplot(data=df, x=target_col, y=feat, palette=palette, inner="quartile")
        plt.title(f"{feat} Violin Plot by {target_col}")
        plt.tight_layout()
        if save_dir:
            plt.savefig(f"{save_dir}/{feat}_violinplot.png")
        plt.show()
        plt.close('all')

    # Correlation matrix
    corr_df = df[top_features].copy()
    corr = corr_df.corr()
    plt.figure(figsize=(7, 5))
    plt.title('Correlation Matrix (Top Features)', fontsize=16)
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
    plt.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}/correlation_matrix.png")
    plt.show()
    plt.close('all')

# Helper functions for report sections
def add_cover_page(pdf, title, author, date):
    plt.figure(figsize=(12, 8))
    plt.axis('off')
    plt.text(0.5, 0.8, title, ha='center', va='center', fontsize=20, fontweight='bold')
    plt.text(0.5, 0.6, f'Author: {author}', ha='center', va='center', fontsize=14)
    plt.text(0.5, 0.5, f'Date: {date}', ha='center', va='center', fontsize=12)
    pdf.savefig(); plt.close()

def add_table_of_contents(pdf):
    plt.figure(figsize=(12, 8))
    plt.axis('off')
    plt.title('Table of Contents', fontsize=16, pad=20)
    sections = [
        '1. Model Implementation',
        '   1.1 Source Code',
        '   1.2 Binary Feature Binning',
        '2. Model Analysis',
        '   2.1 Decision Tree Visualization',
        '   2.2 Confusion Matrix',
        '   2.3 Performance Metrics',
        '   2.4 Feature Importance'
    ]
    y = 0.8
    for sec in sections:
        plt.text(0.1, y, sec, fontsize=12)
        y -= 0.1
    pdf.savefig(); plt.close()

def add_source_code_section(pdf):
    code = [
        'def entropy(y):',
        '    p = sum(y=="malignant")/len(y); return -p*log2(p) - (1-p)*log2(1-p)',
        'def information_gain(y, mask):',
        '    # ... implementation ...',
        '# build_tree and DecisionTreeNode methods'
    ]
    plt.figure(figsize=(12, 8)); plt.axis('off'); plt.title('1.1 Source Code', fontsize=16, pad=20)
    y=0.9
    for line in code:
        plt.text(0.05, y, line, fontsize=10, fontfamily='monospace')
        y -= 0.03
    pdf.savefig(); plt.close()

def add_binning_section(pdf, df, feature_columns):
    numeric = [f for f in feature_columns if pd.api.types.is_numeric_dtype(df[f])]
    med = {f: df[f].median() for f in numeric[:3]}
    plt.figure(figsize=(12, 6)); plt.axis('off'); plt.title('1.2 Binary Feature Binning', fontsize=16, pad=20)
    y=0.8
    for f,v in med.items():
        plt.text(0.1, y, f'{f}: median={v:.2f}', fontsize=12); y-=0.1
    pdf.savefig(); plt.close()

def add_tree_section(pdf, clf_root, depth):
    plt.figure(figsize=(15, 10)); plt.title('2.1 Decision Tree Visualization', fontsize=16, pad=20)
    plot_tree_graphical(clf_root, max_depth=depth, show=False)
    pdf.savefig(); plt.close()

def add_confusion_section(pdf, y_test, y_pred):
    plt.figure(figsize=(8,6)); plt.title('2.2 Confusion Matrix', fontsize=16, pad=20)
    plot_confusion_matrix(y_test, y_pred, labels=['malignant','benign'], show=False)
    pdf.savefig(); plt.close()

def add_metrics_section(pdf, y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label='malignant')
    rec = recall_score(y_test, y_pred, pos_label='malignant')
    f1 = f1_score(y_test, y_pred, pos_label='malignant')
    data = [['Accuracy',f'{acc:.3f}'],['Precision',f'{prec:.3f}'],['Recall',f'{rec:.3f}'],['F1 Score',f'{f1:.3f}']]
    plt.figure(figsize=(8,3)); plt.axis('off'); plt.title('2.3 Performance Metrics', fontsize=16, pad=20)
    tbl = plt.table(cellText=data, colLabels=['Metric','Value'], loc='center'); tbl.auto_set_font_size(False); tbl.set_fontsize(12)
    pdf.savefig(); plt.close()

def add_feature_importance_section(pdf, feature_importances):
    top = sorted(feature_importances.items(), key=lambda x:x[1], reverse=True)[:10]
    feats, vals = zip(*top)
    plt.figure(figsize=(10,6)); plt.bar(feats, vals); plt.xticks(rotation=45,ha='right')
    plt.title('2.4 Feature Importance', fontsize=16)
    pdf.savefig(); plt.close()

# Refactored main function
def save_all_visualizations_pdf(df, feature_importances, y_test, y_pred, clf_root, feature_columns, diagnosis_col, pdf_path='model_report.pdf', top_n=5, bins=10, tree_max_depth=5, project_title='Model Report', author='Author', show=False, metrics=None):
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    with PdfPages(pdf_path) as pdf:
        add_cover_page(pdf, project_title, author, today)
        add_table_of_contents(pdf)
        add_source_code_section(pdf)
        add_binning_section(pdf, df, feature_columns)
        add_tree_section(pdf, clf_root, tree_max_depth)
        add_confusion_section(pdf, y_test, y_pred)
        add_metrics_section(pdf, y_test, y_pred)
        add_feature_importance_section(pdf, feature_importances)
    if show:
        import subprocess; subprocess.call(['open', pdf_path])
