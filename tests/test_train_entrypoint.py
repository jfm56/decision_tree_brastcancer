import runpy
import pandas as pd
import numpy as np


def test_train_entry_testpoint(monkeypatch, capsys):
    # Dummy_test DataFrame for read_csv
    df = pd.DataFrame({
        'id': [1, 2, 3, 4],
        'diagnosis': ['B', 'M', 'B', 'M'],
        'feat1': [1, 2, 3, 4],
        'feat2': [4, 3, 2, 1]
    })
    monkeypatch.setattr(pd, 'read_csv', lambda path: df)

    # Override DecisionTreeClassifier to by_testpass fitting logic
    import src.decision_tree as dt_mod
    class DummyNode:
        def is_leaf(self):
            return True
    class Dummy_testDT:
        def __init__(self, *args, **kwargs):
            self.root = DummyNode()
            self._y_test = None
        def fit(self, X, y_test):
            self.root = DummyNode()
            self._y_test = y_test
            self._feature_names = X.columns if hasattr(X, 'columns') else ['feat1', 'feat2']
        def predict(self, X):
            import pandas as pd
            y = self._y_test
            if hasattr(y, 'reindex') and set(X.index).issubset(set(y.index)):
                return y.reindex(X.index)
            vals = y.values if hasattr(y, 'values') else y
            if len(vals) >= len(X):
                return pd.Series(vals[:len(X)], index=X.index)
            else:
                return pd.Series([vals[0]] * len(X), index=X.index)
        def get_feature_importance(self):
            import pandas as pd
            names = getattr(self, '_feature_names', ['feat1', 'feat2'])
            return pd.Series([0.0]*len(names), index=names)

    monkeypatch.setattr(dt_mod, 'DecisionTreeClassifier', Dummy_testDT)

    # Override visualization save to avoid file output
    import src.visualization as vis_mod
    monkeypatch.setattr(vis_mod, 'save_all_visualizations_pdf', lambda **kwargs: print("SAVE_CALLED"))

    # Run train module
    runpy.run_module('src.train', run_name='__main__')
    captured = capsys.readouterr()

    # Verify_test key_test flow messages
    assert "Loading data..." in captured.out
    assert "Skipping grid search" in captured.out
    assert "SAVE_CALLED" in captured.out
