import pandas as pd
from src.decision_tree import DecisionTreeClassifier

def test_decision_tree_fit_predict():
    features_data = pd.DataFrame({
        'f1': ['low', 'medium', 'high', 'low'],
        'f2': ['high', 'medium', 'low', 'medium']
    })
    target_data = pd.Series([0, 1, 0, 1])
    clf = DecisionTreeClassifier(max_depth=2)
    clf.fit(features_data, target_data)
    preds = clf.predict(features_data)
    assert set(preds) <= set([0, 1])
    assert len(preds) == len(target_data)
