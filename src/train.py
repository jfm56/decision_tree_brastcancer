import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from src import preprocessing
from src.decision_tree import DecisionTreeClassifier
from src.visualization import (
    plot_feature_importance,
    plot_confusion_matrix,
    print_tree,
    save_all_visualizations_pdf
)  # removed unused plot_top_feature_distributions
from src.model_selection import grid_search_tree_cv
import os

if __name__ == "__main__":
    # Load dataset
    print("Loading data...")
    df = pd.read_csv("PROJECT2_DATASET.csv")
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    feature_columns = [col for col in df.columns if col not in ('diagnosis', 'target')]

    # Ensure workflow_screenshots directory exists
    os.makedirs('workflow_screenshots', exist_ok=True)

    # Save data loading workflow
    plt.figure(figsize=(10, 6))
    plt.text(0.1, 0.9, 'Data Loading and Preprocessing Workflow:', fontsize=14, fontweight='bold')
    plt.text(0.1, 0.8, '1. Load data from PROJECT2_DATASET.csv')
    plt.text(0.1, 0.7, '2. Remove ID column if present')
    plt.text(0.1, 0.6, '3. Identify feature columns (excluding diagnosis/target)')
    plt.text(0.1, 0.5, f'4. Total features identified: {len(feature_columns)}')
    plt.text(0.1, 0.4, f'5. Total samples: {len(df)}')
    plt.axis('off')
    plt.savefig('workflow_screenshots/data_loading.png')
    plt.close()
    TARGET_COLUMN = 'diagnosis'

    # 1. Pre-binning feature distribution
    # print("Plotting pre-binning feature distributions...")
    # Only plot numeric columns for pre-binning visualization
    # numeric_feature_columns = [col for col in feature_columns if pd.api.types.is_numeric_dtype(df[col])]
    # print(f"Pre-binning visualization: using only numeric columns: {numeric_feature_columns}")
    # plot_top_feature_distributions(df, {col:1 for col in numeric_feature_columns}, 'diagnosis', top_n=len(numeric_feature_columns), bins=3, show=False)

    # Preprocess
    df_prep = preprocessing.preprocess_data(df, feature_columns, TARGET_COLUMN)
    x = df_prep[feature_columns]
    y = df_prep[TARGET_COLUMN]

    # Save preprocessing workflow
    plt.figure(figsize=(10, 6))
    plt.text(0.1, 0.9, 'Data Preprocessing Steps:', fontsize=14, fontweight='bold')
    plt.text(0.1, 0.8, '1. Scale numeric features')
    plt.text(0.1, 0.7, '2. Bin features into categories (low, medium, high)')
    plt.text(0.1, 0.6, '3. Handle any missing values')
    plt.text(0.1, 0.5, '4. Prepare features (X) and target (y)')
    plt.axis('off')
    plt.savefig('workflow_screenshots/preprocessing.png')
    plt.close()

    # Train/test split
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Save model configuration
    plt.figure(figsize=(10, 6))
    plt.text(0.1, 0.9, 'Model Configuration:', fontsize=14, fontweight='bold')
    plt.text(0.1, 0.8, 'Decision Tree Parameters:')
    plt.text(0.2, 0.7, '- max_depth: 4')
    plt.text(0.2, 0.6, '- criterion: entropy')
    plt.text(0.2, 0.5, '- split strategy: binary classification')
    plt.text(0.1, 0.4, 'Training Configuration:')
    plt.text(0.2, 0.3, '- train/test split: 80%/20%')
    plt.text(0.2, 0.2, '- random_state: 42')
    plt.axis('off')
    plt.savefig('workflow_screenshots/model_config.png')
    plt.close()

    # Skip grid search - go straight to PDF generation with fixed max_depth
    print("Skipping grid search. Using fixed max_depth=4 and entropy criterion for model and report.")
    best_params = {'max_depth': 4}

    # 2. Fit model with best max_depth
    clf = DecisionTreeClassifier(max_depth=4, criterion='entropy')
    clf.fit(x_train, y_train)
    # Predict
    y_pred = clf.predict(x_test)
    acc = (y_pred == y_test).mean()
    print(f"Test accuracy: {acc:.3f}")

    # Submodel: use top 10 features only
    feat_imp_all = clf.get_feature_importance()
    top_features = [f for f, _ in sorted(feat_imp_all.items(), key=lambda x: x[1], reverse=True)][:10]
    print("Retraining on top features:", top_features)
    clf = DecisionTreeClassifier(max_depth=4, criterion='entropy')
    clf.fit(x_train[top_features], y_train)
    y_pred = clf.predict(x_test[top_features])
    acc_sub = (y_pred == y_test).mean()
    print(f"Submodel accuracy (top 10 features): {acc_sub:.3f}")

    # 3. Confusion matrix
    print("Plotting confusion matrix...")
    plot_confusion_matrix(y_test, y_pred, labels=['malignant', 'benign'])

    # 4. Feature importance
    print("Plotting feature importance...")
    feat_imp = clf.get_feature_importance()
    plot_feature_importance(feat_imp)

    # 5. Print tree structure
    print("\nDecision Tree Structure:")
    print_tree(clf.root)

    # 7. Save ALL results to a single PDF report
    print("\nSaving ALL results to model_report.pdf...")

    # Visualizations for top 5 features (no PDF, just interactive plots)
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    top_n = 5
    bins = 10
    # Only use numeric features
    numeric_feature_columns = [col for col in feature_columns if pd.api.types.is_numeric_dtype(df[col])]
    top_features = [f for f, _ in sorted(feat_imp.items(), key=lambda x: x[1], reverse=True) if f in numeric_feature_columns][:top_n]
    print("Top 5 features by importance:", top_features)

    # Summary statistics
    stats = df[top_features].agg(['mean', 'median', 'std']).T.round(3)
    print("\nSummary Statistics (Top 5 Features):\n", stats)
    plt.figure(figsize=(8, 2 + len(top_features) * 0.4))
    plt.axis('off')
    plt.title('Summary Statistics (Top 5 Features)', fontsize=16, pad=20)
    table = plt.table(cellText=stats.values, rowLabels=stats.index, colLabels=stats.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)
    plt.tight_layout()
    plt.show()

    # Custom interactive visualizations (top features)
    from src.visualization import visualize_top_features
    save_dir = "visualizations"
    os.makedirs(save_dir, exist_ok=True)
    visualize_top_features(df, feat_imp, TARGET_COLUMN, top_n=5, bins=5, palette="Set2", save_dir=save_dir)
    print(f"Overlay plots saved to directory: {save_dir}")
    
    # Calculate performance metrics
    from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, pos_label='malignant'),
        'recall': recall_score(y_test, y_pred, pos_label='malignant'),
        'f1': f1_score(y_test, y_pred, pos_label='malignant')
    }
    
    # Print metrics to console
    print("\nPerformance Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1 Score: {metrics['f1']:.3f}")

    # Find path to first malignant leaf for highlighting
    def find_malignant_path(node, path=None):
        if path is None:
            path = []
        if hasattr(node, 'value') and node.value == 'malignant':
            return path
        if hasattr(node, 'children') and node.children:
            for value, child in node.children.items():
                # Use node.feature, node.threshold, and value to match highlight_path signature
                child_path = path + [(node.feature, node.threshold, value)]
                found = find_malignant_path(child, child_path)
                if found is not None:
                    return found
        return None
    # Highlight path: concave points_worst > threshold, perimeter_worst > threshold, malignant
    # We need to extract the actual thresholds from the tree
    node = clf.root
    highlight_path = []
    if node and hasattr(node, 'feature') and node.feature == 'concave points_worst':
        first_threshold = node.threshold
        right_child = node.children.get('gt')
        highlight_path.append((node.feature, first_threshold, 'gt'))
        if right_child and hasattr(right_child, 'feature') and right_child.feature == 'perimeter_worst':
            second_threshold = right_child.threshold
            right2_child = right_child.children.get('gt')
            highlight_path.append((right_child.feature, second_threshold, 'gt'))
            # The leaf should be malignant
            if right2_child and hasattr(right2_child, 'value') and right2_child.value == 'malignant':
                # Add the leaf step for completeness
                highlight_path.append(('malignant', None, None))
    from src.visualization import plot_tree_graphical
    plot_tree_graphical(clf.root, max_depth=best_params['max_depth'], highlight_path=highlight_path, show=True)

    # Generate comprehensive PDF report
    save_all_visualizations_pdf(
        df=df,
        feature_importances=feat_imp_all,
        y_test=y_test,
        y_pred=y_pred,
        clf_root=clf.root,
        feature_columns=top_features,
        diagnosis_col=TARGET_COLUMN,
        pdf_path="model_report.pdf",
        top_n=5,
        bins=bins,
        tree_max_depth=best_params['max_depth'],
        metrics=metrics
    )
    print("\nComprehensive model report saved to model_report.pdf")
