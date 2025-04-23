import pandas as pd
from src.decision_tree import DecisionTreeClassifier

def test_decision_tree_fit_predict():
    X = pd.DataFrame({
        'f1': ['low', 'medium', 'high', 'low'],
        'f2': ['high', 'medium', 'low', 'medium']
    })
    y = pd.Series(['A', 'B', 'A', 'B'])
    clf = DecisionTreeClassifier(max_depth=2)
    clf.fit(X, y)
    preds = clf.predict(X)
    assert set(preds) <= set(['A', 'B'])
    assert len(preds) == len(y)
