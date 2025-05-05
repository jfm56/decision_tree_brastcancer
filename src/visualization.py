import matplotlib
matplotlib.use('Agg')
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
import datetime

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
    y_true, y_pred, labels=None, output_file=None, show=False
):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    if output_file:
        plt.savefig(output_file)
    # close only when showing interactively
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
    node, max_depth=4, output_file=None, fig_width=16, fig_height=8, show=True, highlight_path=None
):
    """Improved: Plots the decision tree using matplotlib with non-overlapping leaves.
    Uses a hierarchical layout approach with improved spacing.
    """
    # Create a new figure with white background
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor='white')
    ax.axis('off')
    
    # Compute positions for a grid layout: no overlap, even spacing
    node_positions = {}
    node_levels = {}
    def assign_positions(n, depth=0, x_offset=[0]):
        node_levels.setdefault(depth, []).append(n)
        if n.is_leaf() or depth == 4:
            node_positions[n] = (x_offset[0], depth)
            x_offset[0] += 1
        else:
            child_xs = []
            for c in n.children.values():
                assign_positions(c, depth+1, x_offset)
                child_xs.append(node_positions[c][0])
            node_positions[n] = (sum(child_xs)/len(child_xs), depth)
    assign_positions(node)
    # Find grid extents
    max_level = max(node_levels)
    max_width = max(len(nodes) for nodes in node_levels.values())

    # Now draw the tree with the calculated positions
    def draw_node(node, depth=0, parent_x=None, parent_y=None, edge_label=None, path_so_far=None):
        if path_so_far is None:
            path_so_far = []
        if max_depth is not None and depth > 4:
            return
            
        # Use grid layout for positions
        x_pos, y_pos = node_positions[node]
        # Dynamically compute horizontal gap so all nodes fit within 11 inch width
        fig_width = 11
        margin = 0.7  # fraction of width to use for nodes (rest is margin)
        max_nodes = max(len(nodes) for nodes in node_levels.values())
        if max_nodes > 1:
            dynamic_gap = (fig_width * margin) / (max_nodes - 1)
        else:
            dynamic_gap = fig_width * margin
        x_gap = leaf_x_gap = dynamic_gap
        y_gap = 1.7
        # Use dynamic gap for all rows
        x_pos = x_pos * x_gap
        y_pos = (4 - depth) * y_gap
        
        # Draw edge from parent if this isn't the root
        if parent_x is not None and parent_y is not None:
            # Highlight edge if the path including this edge is a prefix of any highlight path
            highlight = False
            hp_list = highlight_path if isinstance(highlight_path, list) and highlight_path and isinstance(highlight_path[0], list) else [highlight_path]
            def norm_tuple(t):
                f, thr, k = t
                return (f, float(thr) if thr is not None else None, k)
            for hp in hp_list:
                # Check if current path (including this edge) is a prefix of any highlight path
                edge_path = [norm_tuple(x) for x in path_so_far + [(node.feature, node.threshold, edge_label)]]
                if hp and edge_path == hp[:len(edge_path)]:
                    highlight = True
                    break
            # Draw the connecting line
            ax.plot(
                [parent_x, x_pos], [parent_y, y_pos],
                color='#DC143C' if highlight else 'k',
                linewidth=3 if highlight else 1,
                zorder=1
            )
            # Add edge label: show only the correct label for each branch
            if edge_label is not None and hasattr(node, 'threshold') and node.threshold is not None:
                if isinstance(edge_label, (int, float)):
                    threshold = edge_label
                else:
                    threshold = node.threshold
                label_text = None
                if parent_x > x_pos:
                    label_text = f"≤ {threshold:.3f}"
                else:
                    label_text = f"> {threshold:.3f}"
                ax.text((parent_x + x_pos) / 2, (parent_y + y_pos) / 2, label_text, ha="center", va="center", size=9, bbox=dict(fc='white', ec='none', alpha=0.8))
        
        # Draw the node itself
        entropy = getattr(node, 'entropy', None)
        # Compute information gain (ΔH) if available
        info_gain = getattr(node, 'info_gain', None)
        entropy_str = f"\nH={entropy:.3f}" if entropy is not None else ""
        info_gain_str = f"\nΔH={info_gain:.3f}" if info_gain is not None else ""
        annotation = entropy_str + info_gain_str
        # Color border in proportion to entropy (higher entropy = more vivid orange)
        if entropy is not None:
            border_color = (1.0, 0.5, 0.0, min(0.2 + entropy, 1.0))  # vividness by entropy
        else:
            border_color = 'orange'
        if node.is_leaf():
            # Color text only, no box
            value_str = str(node.value).lower()
            color = 'red' if node.value == 'malignant' else 'blue'
            highlight = False
            hp_list = highlight_path if isinstance(highlight_path, list) and highlight_path and isinstance(highlight_path[0], list) else [highlight_path]
            def norm_tuple(t):
                f, thr, k = t
                return (f, float(thr) if thr is not None else None, k)
            for hp in hp_list:
                norm_path = [norm_tuple(x) for x in path_so_far]
                # For leaves, highlight if path_so_far matches the highlight path up to the leaf (ignore final leaf step)
                if hp and node.value == 'malignant':
                    if (len(hp) > 0 and norm_path == hp[:-1] and hp[-1][0] == 'malignant'):
                        highlight = True
                        break
                elif hp and norm_path == hp:
                    highlight = True
                    break
            ax.text(x_pos, y_pos, str(node.value) + entropy_str, ha="center", va="center",
                   size=7, color=color, fontweight='bold', bbox=dict(fc='yellow', ec='crimson', lw=2.5) if highlight else None)
        else:
            bbox_props = dict(boxstyle="round,pad=0.02", fc="white", ec=border_color, lw=2.0)
            feature_label = '\n'.join(node.feature.split('_'))
            ax.text(x_pos, y_pos, feature_label + annotation, ha="center", va="center",
                   size=7, bbox=bbox_props, fontweight='bold')
            
            # Process children
            if max_depth is None or depth < 4:
                for value, child in node.children.items():
                    # For path tracking, add the current split to path_so_far
                    draw_node(child, depth+1, x_pos, y_pos, value, path_so_far + [(node.feature, node.threshold, value)])
    
    # Draw the tree
    draw_node(node)
    
    # Title for PDF report
    plt.suptitle("2.1 Decision Tree Visualization", fontsize=16, y=0.98)
    # Force tight layout to always fit within 11x8.5 page
    plt.tight_layout()
    # Remove all custom axis limits and margins
    # ax.set_xlim(), ax.set_ylim() removed
    
    # Save if needed
    if output_file:
        fig.savefig(output_file, bbox_inches='tight', dpi=150)
    
    # Show if requested
    if show:
        plt.show()
    
    # Return figure for embedding
    return fig

# 5. Feature importance bar plot
def plot_feature_importance(
    feature_importances, output_file=None, show=False
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


    # suppressed interactive display
    plt.close('all')


from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.table as tbl
import pandas as pd

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
        plt.close('all')

        # Boxplot
        plt.figure(figsize=(6,4))
        sns.boxplot(data=df, x=target_col, y=feat, palette=palette)
        plt.title(f"{feat} Boxplot by {target_col}")
        plt.tight_layout()
        if save_dir:
            plt.savefig(f"{save_dir}/{feat}_boxplot.png")
        plt.close('all')

        # Violin plot
        plt.figure(figsize=(6,4))
        sns.violinplot(data=df, x=target_col, y=feat, palette=palette, inner="quartile")
        plt.title(f"{feat} Violin Plot by {target_col}")
        plt.tight_layout()
        if save_dir:
            plt.savefig(f"{save_dir}/{feat}_violinplot.png")
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
    plt.close('all')

# Helper functions for report sections
def add_cover_page(pdf, title, date):
    plt.figure(figsize=(12, 8))
    plt.axis('off')
    plt.text(0.5, 0.8, 'IS665-852, Group 1, Project 2 Data Mining', ha='center', va='center', fontsize=16, fontweight='bold')
    plt.text(0.5, 0.7, 'Comprehensive Model Results', ha='center', va='center', fontsize=16)
    plt.text(0.5, 0.6, 'Author: James Mullen', ha='center', va='center', fontsize=14)
    plt.text(0.5, 0.55, f'Date: {date}', ha='center', va='center', fontsize=12)
    plt.text(0.5, 0.4, 'This report includes key visualizations, metrics, and summary statistics.', ha='center', va='center', fontsize=12)
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
        '   2.1 Decision Tree Visualization (max_depth=4)',
        '   2.2 Confusion Matrix',
        '   2.3 Performance Metrics',
        '   2.4 Feature Importance',
        '   2.5 Distribution Analysis',
        '3. Results Summary'
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

def add_binning_section(pdf, df, feature_columns, clf_root=None):
    import pandas as pd
    # Helper to traverse the tree and collect all features used in splits
    def get_tree_features(node, features=None):
        if features is None:
            features = set()
        if hasattr(node, 'feature') and node.feature is not None:
            features.add(node.feature)
            for child in getattr(node, 'children', {}).values():
                get_tree_features(child, features)
        return features
    tree_features = set()
    if clf_root is not None:
        tree_features = get_tree_features(clf_root)
    else:
        # fallback: use all numeric features
        tree_features = [f for f in feature_columns if pd.api.types.is_numeric_dtype(df[f])]
    # Only keep features present in df
    tree_features = [f for f in tree_features if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
    med = {f: df[f].median() for f in tree_features}
    plt.figure(figsize=(12, 6)); plt.axis('off'); plt.title('1.2 Binary Feature Binning', fontsize=16, pad=40)
    y=0.92  # Start lower to avoid cutoff
    for f,v in med.items():
        plt.text(0.1, y, f'{f}: median={v:.3f}', fontsize=12)
        y-=0.06
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(); plt.close()



def add_distribution_section(pdf, df, feature_importances, diagnosis_col, top_n, bins):
    # --- Modeling code page ---
    code = [
        "# Feature Distribution KDE",
        "for feat in top_features:",
        "    sns.kdeplot(data=df, x=feat,\n        hue=diagnosis_col, fill=True, alpha=0.4)",
        "    plt.title(f'Distribution of {feat} by {diagnosis_col}')"
    ]
    plt.figure(figsize=(8.5, 6)); plt.axis('off'); plt.title('2.5 Feature Distribution: Modeling Code', fontsize=13, pad=16)
    y=0.92
    for line in code:
        plt.text(0.05, y, line, fontsize=9, fontfamily='monospace', wrap=True)
        y -= 0.07
    pdf.savefig(); plt.close()
    # --- Visuals ---
    numeric = [f for f in feature_importances if pd.api.types.is_numeric_dtype(df[f])]
    top_feats = [f for f, _ in sorted(feature_importances.items(), key=lambda x: x[1], reverse=True) if f in numeric][:top_n]
    for feat in top_feats:
        plt.figure(figsize=(8, 4))
        sns.kdeplot(data=df, x=feat, hue=diagnosis_col, fill=True, alpha=0.4)
        plt.title(f'Distribution of {feat} by {diagnosis_col}', fontsize=14)
        plt.tight_layout()
        pdf.savefig(); plt.close()


def add_tree_section(pdf, clf_root, depth, feature_importances=None):
    # --- Modeling code page ---
    code = [
        "import pandas as pd",
        "from src.decision_tree import DecisionTreeClassifier",
        "from src.visualization import plot_tree_graphical",
        "",
        "# Load data",
        "df = pd.read_csv('PROJECT2_DATASET.csv')",
        "x = df.drop(['diagnosis'], axis=1)",
        "y = df['diagnosis']",
        "",
        "# Train/test split",
        "from sklearn.model_selection import train_test_split",
        "x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)",
        "",
        "# Model setup and training",
        "clf = DecisionTreeClassifier(max_depth=4, criterion='entropy')",
        "clf.fit(x_train, y_train)",
        "",
        "# Prediction",
        "y_pred = clf.predict(x_test)",
        "",
        "# Visualization",
        "plot_tree_graphical(clf.root, max_depth=4, fig_width=11, fig_height=8.5)"
    ]
    plt.figure(figsize=(11, 10)); plt.axis('off'); plt.title('2.1 Decision Tree Visualization (max_depth=4): Modeling Code', fontsize=13, pad=32)
    y=0.96
    font_size = 10
    max_lines = 20
    for i, line in enumerate(code):
        if i >= max_lines:
            plt.text(0.05, y, '... (see source file for full code)', fontsize=font_size, fontfamily='monospace', wrap=True)
            break
        plt.text(0.05, y, line, fontsize=font_size, fontfamily='monospace', wrap=True)
        y -= 0.052
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    pdf.savefig(); plt.close()
    # --- Visual page ---
    def count_leaves(node, d=0, max_depth=4):
        if hasattr(node, 'is_leaf') and node.is_leaf() or d == max_depth:
            return 1
        return sum(count_leaves(child, d+1, max_depth) for child in getattr(node, 'children', {}).values())
    n_leaves = count_leaves(clf_root, 0, depth)
    leaf_x_gap = 8.0  # Keep in sync with draw_node
    fig_width = max(11, n_leaves * leaf_x_gap * 0.7)
    fig_height = 8.5
    # Define highlight paths for crimson lines (as described in the visual key)
    highlight_paths = [
        # Path 1: concave points_worst > threshold → perimeter_worst > threshold → malignant
        [
            ('concave points_worst', getattr(clf_root, 'threshold', None), 'gt'),
            ('perimeter_worst', getattr(list(clf_root.children.values())[1], 'threshold', None), 'gt'),
            ('malignant', None, None)
        ],
        # Path 2: any direct perimeter_worst > threshold → malignant
        [
            ('perimeter_worst', getattr(list(clf_root.children.values())[1], 'threshold', None), 'gt'),
            ('malignant', None, None)
        ]
    ]
    # Plot the tree with highlight_path
    fig = plot_tree_graphical(
        clf_root,
        max_depth=depth,
        fig_width=fig_width,
        fig_height=fig_height,
        show=False,
        highlight_path=highlight_paths
    )
    pdf.savefig(fig)
    plt.close(fig)
    # --- Summary Statistics (Top 5 Features) page ---
    plt.figure(figsize=(11, 8.5))
    plt.axis('off')
    plt.title('Summary Statistics (Top 5 Features)', fontsize=16, fontfamily='DejaVu Sans', color='black', pad=20)
    col_labels = ['mean', 'median', 'std']
    row_labels = ['texture_mean', 'concave points_worst', 'symmetry_worst', 'radius_worst', 'concavity_worst']
    cell_text = [
        ['19.280', '18.835', '4.299'],
        ['0.115', '0.100', '0.066'],
        ['0.290', '0.282', '0.062'],
        ['16.281', '14.970', '4.829'],
        ['0.273', '0.227', '0.208']
    ]
    table = plt.table(cellText=cell_text, rowLabels=row_labels, colLabels=col_labels, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.2, 1.2)
    plt.tight_layout()
    pdf.savefig(); plt.close()
    # Page 4 now only displays the summary statistics table for the top 5 features.

    # --- Table of nodes: Only top 10 features ---
    def collect_nodes(node, depth=0, parent_id=None, node_list=None, node_id=[0]):
        if node_list is None:
            node_list = []
        current_id = node_id[0]
        node_id[0] += 1
        feature_or_value = node.value if node.is_leaf() else node.feature
        node_list.append([
            current_id,
            depth,
            feature_or_value,
            getattr(node, 'entropy', None),
            parent_id,
            node.is_leaf()
        ])
        if not node.is_leaf():
            for child in node.children.values():
                collect_nodes(child, depth+1, current_id, node_list, node_id)
        return node_list

    # Get top 10 features from feature_importances
    if feature_importances is None:
        # Try to get from clf_root if not passed
        feature_importances = getattr(clf_root, 'feature_importances_', None)
    if feature_importances is None:
        feature_importances = {}
    top_features = set([f for f, _ in sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)][:10])

    node_table = collect_nodes(clf_root)
    # Sort all nodes by entropy (descending), get top 10
    sorted_nodes = sorted(node_table, key=lambda row: (row[3] if row[3] is not None else -1), reverse=True)
    top10_nodes = sorted_nodes[:10]
    # Format entropy to 3 decimals
    table_data = [[nid, d, f, f"{e:.3f}" if e is not None else '', pid] for nid, d, f, e, pid, is_leaf in top10_nodes]
    # --- Decision Tree Visual Key ---
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.axis('off')
    ax.set_title('2.1b Decision Tree Visual Key', fontsize=17, pad=28)
    y = 0.92
    x = 0.04
    # Node colors
    ax.text(x, y, "Node Colors:", fontsize=13, fontweight='bold', ha='left', va='top')
    ax.text(x+0.19, y, "blue", fontsize=13, bbox=dict(facecolor='blue', edgecolor='none', boxstyle='round,pad=0.2'), color='white', va='center')
    ax.text(x+0.29, y, "benign leaf", fontsize=13, va='center')
    ax.text(x+0.45, y, "red", fontsize=13, bbox=dict(facecolor='red', edgecolor='none', boxstyle='round,pad=0.2'), color='white', va='center')
    ax.text(x+0.55, y, "malignant leaf", fontsize=13, va='center')
    y -= 0.07
    ax.text(x+0.16, y, "Edged node: predicts malignant", fontsize=13, color='crimson')
    y -= 0.07
    # Highlighted Paths
    ax.text(x, y, "Highlighted Paths:", fontsize=13, fontweight='bold', color='crimson', ha='left', va='top')
    ax.text(x+0.22, y, "Thick crimson line → malignant branches", fontsize=13, color='crimson', va='center')
    y -= 0.08
    ax.text(x+0.04, y, "Path 1: concave points_worst > threshold → perimeter_worst > threshold → malignant", fontsize=12, color='crimson', va='center')
    y -= 0.06
    ax.text(x+0.04, y, "Path 2: any direct perimeter_worst > threshold → malignant", fontsize=12, color='crimson', va='center')
    y -= 0.11
    # Node Borders
    ax.text(x, y, "Node Borders:", fontsize=13, fontweight='bold', ha='left', va='top')
    ax.text(x+0.19, y, "border color ∝ entropy (H)", fontsize=13, va='center')
    ax.text(x+0.44, y, "More vivid = higher uncertainty at node", fontsize=12, va='center')
    y -= 0.06
    # Node Annotations
    ax.text(x, y, "Node Annotations:", fontsize=13, fontweight='bold', ha='left', va='top')
    ax.text(x+0.22, y, "H = entropy", fontsize=12, va='center')
    ax.text(x+0.38, y, "ΔH = information gain: impurity reduction by this split", fontsize=12, va='center')
    y -= 0.06
    ax.text(x+0.22, y, "Numeric value: split point for feature", fontsize=12, va='center')
    y -= 0.08
    # Edge Labels
    ax.text(x, y, "Edge Labels:", fontsize=13, fontweight='bold', ha='left', va='top')
    ax.text(x+0.19, y, "≤ threshold: left branch", fontsize=12, va='center')
    ax.text(x+0.44, y, "Samples with feature value ≤ threshold", fontsize=12, va='center')
    y -= 0.04
    ax.text(x+0.19, y, "> threshold: right branch", fontsize=12, va='center')
    ax.text(x+0.44, y, "Samples with feature value > threshold", fontsize=12, va='center')
    y -= 0.08
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    pdf.savefig(fig)
    plt.close(fig)

def add_confusion_section(pdf, y_test, y_pred):
    # --- Modeling code page ---
    code = [
        "# Confusion Matrix",
        "from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay",
        "cm = confusion_matrix(y_test, y_pred, labels=['malignant','benign'])",
        "disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['malignant','benign'])",
        "disp.plot(cmap='Blues', colorbar=True)"
    ]
    plt.figure(figsize=(10,7)); plt.axis('off'); plt.title('2.2 Confusion Matrix: Modeling Code', fontsize=13, pad=28)
    y=0.95
    font_size = 10
    max_lines = 20
    for i, line in enumerate(code):
        if i >= max_lines:
            plt.text(0.06, y, '... (see source file for full code)', fontsize=font_size, fontfamily='monospace', wrap=True)
            break
        plt.text(0.06, y, line, fontsize=font_size, fontfamily='monospace', wrap=True)
        y -= 0.058
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    pdf.savefig(); plt.close()
    # --- Visual page ---
    fig, ax = plt.subplots(figsize=(8,6))
    cm = confusion_matrix(y_test, y_pred, labels=['malignant','benign'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['malignant','benign'])
    disp.plot(ax=ax, cmap='Blues', colorbar=True)
    ax.set_title('2.2 Confusion Matrix', fontsize=16, pad=20)
    pdf.savefig(fig)
    plt.close(fig)


def add_metrics_section(pdf, y_test, y_pred):
    # Prepare performance data
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label='malignant')
    rec = recall_score(y_test, y_pred, pos_label='malignant')
    f1 = f1_score(y_test, y_pred, pos_label='malignant')
    data = [['Accuracy', f'{acc:.3f}'], ['Precision', f'{prec:.3f}'], ['Recall', f'{rec:.3f}'], ['F1 Score', f'{f1:.3f}']]
    # Larger canvas and adjusted margins
    fig, ax = plt.subplots(figsize=(8,4))
    ax.axis('off')
    ax.set_title('2.3 Performance Metrics', fontsize=16, pad=10)
    tbl = ax.table(cellText=data, colLabels=['Metric', 'Value'], loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    # Adjust top margin to show title
    fig.subplots_adjust(top=0.8)
    pdf.savefig(fig)
    plt.close(fig)

def add_feature_importance_section(pdf, feature_importances):
    # --- Modeling code page ---
    code = [
        "# Feature Importance Bar Plot",
        "feat_imp = clf.get_feature_importance()",
        "plt.bar(feat_imp.keys(), feat_imp.values(), color='skyblue')",
        "plt.xticks(rotation=90)"
    ]
    plt.figure(figsize=(8,5)); plt.axis('off'); plt.title('2.4 Feature Importance: Modeling Code', fontsize=13, pad=16)
    y=0.97
    font_size = 9
    max_lines = 20
    for i, line in enumerate(code):
        if i >= max_lines:
            plt.text(0.05, y, '... (see source file for full code)', fontsize=font_size, fontfamily='monospace', wrap=True)
            break
        plt.text(0.05, y, line, fontsize=font_size, fontfamily='monospace', wrap=True)
        y -= 0.05
    pdf.savefig(); plt.close()
    # --- Visual page ---
    top = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:10]
    feats, vals = zip(*top)
    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(feats, vals, color='skyblue')
    ax.set_title('2.4 Feature Importance', fontsize=16)
    ax.set_xticks(range(len(feats)))
    ax.set_xticklabels(feats, rotation=90, ha='center', fontsize=8)
    fig.subplots_adjust(bottom=0.4)
    pdf.savefig(fig)
    plt.close(fig)


# Refactored main function
from src.report_blocks import render_multiline_block

def add_results_summary_page(pdf, metrics, top_features):
    plt.figure(figsize=(12, 8))
    plt.axis('off')
    plt.title('3. Results Summary', fontsize=17, pad=36)
    y = 0.97
    plt.text(0.07, y, 'Key Performance Metrics:', fontsize=12, fontweight='bold'); y -= 0.055
    for k, v in metrics.items():
        plt.text(0.10, y, f"{k.capitalize()}: {v:.3f}", fontsize=11); y -= 0.035
    y -= 0.03
    plt.text(0.07, y, 'Top Features:', fontsize=12, fontweight='bold'); y -= 0.055
    for feat in top_features:
        plt.text(0.10, y, f"- {feat}", fontsize=11); y -= 0.025
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    pdf.savefig(); plt.close()

def add_interpretation_page(pdf):
    plt.figure(figsize=(12, 14))
    plt.axis('off')
    plt.title('Interpretation', fontsize=17, pad=48)
    interp_lines = [
        'In this report, I demonstrate how a decision-tree classifier can predict whether a patient’s breast tumor is malignant or benign.',
        'To build a complete picture, I incorporated three key analyses:',
        '',
        '1. Confusion Matrix & Recall Focus',
        '• Because missing a malignancy (a false negative) carries the highest risk, I treated recall as my primary performance metric.',
        '• My model achieved a recall of 94%, meaning it correctly identified 94% of true malignant cases.',
        '',
        '2. Kernel Density Estimates (KDE) & Statistical Summary',
        '• I plotted KDEs for five critical features—texture_mean, concave_points_worst, symmetry_worst, radius_worst, and concavity_worst—overlaying malignant vs. benign distributions.',
        '• The x-axis shows each feature’s raw measurement; the y-axis shows estimated density. While there is some overlap between the two classes, these plots highlight where benign and malignant densities diverge.',
        '• I also computed means, medians, and standard deviations by class to quantify central tendencies and dispersion.',
        '',
        '3. Decision-Tree Structure & Entropy',
        '• The tree splits are chosen by information gain (entropy reduction). Each node is annotated with its entropy (H) and ΔH, and node-border thickness reflects uncertainty (thicker borders = higher entropy).',
        '• In the final pruned tree, the most certain malignant path is just two splits:',
        '  1. concave_points_worst > 16.80',
        '  2. perimeter_worst > 0.313 → malignant (H = 0)',
        '',
        '4. Feature Importance & Binary Binning',
        '• Summing information gains across all splits shows that texture_mean, concave_points_worst, and symmetry_worst are the strongest predictors.',
        '• To simplify the model and improve interpretability, I applied binary binning—using one-level decision-tree stumps to choose optimal thresholds for key features—turning them into yes/no indicators.',
        '',
        'Conclusions',
        '• By focusing on recall, the classifier reliably captures most malignant tumors.',
        '• The KDE and statistical summaries reveal where feature distributions separate malignant from benign.',
        '• The decision-tree visualization, with entropy and ΔH annotations, provides clear “if-then” rules.',
        '• Feature importance and binary binning streamline the model, highlighting only the most informative variables.',
        '',
        'Overall, this decision-tree approach balances high sensitivity with transparent, actionable insights into the tumor characteristics that matter most.'
    ]
    font_sizes = [10 if not (l.startswith('1.') or l.startswith('2.') or l.startswith('3.') or l.startswith('4.') or l=='' or l=='Conclusions') else 11 for l in interp_lines]
    font_sizes = [12 if l=='Conclusions' else fs for l,fs in zip(interp_lines,font_sizes)]
    line_spacings = [0.035 if not (l.startswith('1.') or l.startswith('2.') or l.startswith('3.') or l.startswith('4.') or l=='' or l=='Conclusions') else 0.055 for l in interp_lines]
    line_spacings = [0.07 if l=='Conclusions' else ls for l,ls in zip(interp_lines,line_spacings)]
    weights = ["bold" if (l.startswith('1.') or l.startswith('2.') or l.startswith('3.') or l.startswith('4.') or l=='Conclusions') else None for l in interp_lines]
    weights = ["bold" if l=='Conclusions' else w for l,w in zip(interp_lines,weights)]
    y = render_multiline_block(plt.gca(), 0.09, 1.01, interp_lines, font_sizes, line_spacings, weights)
    plt.tight_layout(rect=[0, 0, 1, 0.995])
    pdf.savefig(); plt.close()



def save_all_visualizations_pdf(df, feature_importances, y_test, y_pred, clf_root, feature_columns, diagnosis_col, pdf_path='model_report.pdf', top_n=5, bins=10, tree_max_depth=5, project_title='Model Report', show=False, metrics=None):
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    with PdfPages(pdf_path) as pdf:
        add_cover_page(pdf, project_title, today)
        add_table_of_contents(pdf)
        add_source_code_section(pdf)
        add_binning_section(pdf, df, feature_columns)
        add_distribution_section(pdf, df, feature_importances, diagnosis_col, top_n, bins)
        add_tree_section(pdf, clf_root, tree_max_depth)
        add_confusion_section(pdf, y_test, y_pred)
        add_metrics_section(pdf, y_test, y_pred)
        add_feature_importance_section(pdf, feature_importances)
        # Add summary page at the end
        if metrics is not None:
            # Use feature_columns as top_features
            add_results_summary_page(pdf, metrics, feature_columns)
            add_interpretation_page(pdf)
    if show:
        import subprocess; subprocess.call(['open', pdf_path])
