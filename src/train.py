import pandas as pd
from sklearn.datasets import load_breast_cancer
from preprocessing import preprocess_data
from decision_tree import DecisionTreeClassifier
from visualization import (
    plot_feature_distributions,
    plot_binned_feature_counts,
    plot_confusion_matrix,
    print_tree,
    plot_feature_importance,
    plot_tree_graphical
)

if __name__ == "__main__":
    # Load dataset
    # Load CSV dataset
    import pandas as pd
    df = pd.read_csv('PROJECT2_DATASET.csv')
    # Remove ID column if present
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    feature_cols = [col for col in df.columns if col not in ('diagnosis', 'target')]
    target_col = 'diagnosis'

    # 1. Pre-binning feature distribution
    print("Plotting pre-binning feature distributions...")
    plot_feature_distributions(df, feature_cols)

    # Preprocess
    df_prep = preprocess_data(df, feature_cols, target_col)
    X = df_prep[feature_cols]
    y = df_prep[target_col]

    # 2. Post-binning feature counts
    print("Plotting post-binning feature counts...")
    plot_binned_feature_counts(df_prep, feature_cols)

    # Train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Fit model
    clf = DecisionTreeClassifier(max_depth=5)
    clf.fit(X_train, y_train)
    # Predict
    y_pred = clf.predict(X_test)
    acc = (y_pred == y_test).mean()
    print(f"Test accuracy: {acc:.3f}")

    # 3. Confusion matrix
    print("Plotting confusion matrix...")
    plot_confusion_matrix(y_test, y_pred, labels=['malignant','benign'])

    # 4. Feature importance
    print("Plotting feature importance...")
    feat_imp = clf.get_feature_importance()
    plot_feature_importance(feat_imp)

    # 5. Print tree structure
    print("\nDecision Tree Structure:")
    print_tree(clf.root)

    # 6. Graphical tree visualization
    print("\nSaving graphical decision tree as PDF (tree_visualization.pdf)...")
    # Save a readable PDF of the tree (top 3 levels, large page). Adjust max_depth/fig_width as needed.
    plot_tree_graphical(clf.root, max_depth=4, out_file="tree_visualization.pdf", fig_width=48, fig_height=28, xlim=44, ylim=28)
    print("Graphical tree (top 4 levels, extra spacing) saved to tree_visualization.pdf. For more/less detail, adjust max_depth or figure size in train.py.")
