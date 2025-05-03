import pandas as pd
from src.decision_tree import DecisionTreeClassifier
from src.visualization import plot_tree_graphical

# Minimal test to check improved decision tree plot

def test_plot_tree_graphical(tmp_path):
    # Create a simple dataset
    X = pd.DataFrame({
        'feature1': ['A', 'A', 'B', 'B', 'C', 'C'],
        'feature2': ['X', 'Y', 'X', 'Y', 'X', 'Y']
    })
    y = pd.Series([0, 1, 0, 1, 0, 1])
    clf = DecisionTreeClassifier(max_depth=3)
    clf.fit(X, y)
    # Plot and save to a temp file
    out_file = tmp_path / 'tree_test.png'
    plot_tree_graphical(clf.root, output_file=str(out_file))
    assert out_file.exists() and out_file.stat().st_size > 0
