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
    class Dummy_testDT:
        def __init__(self, *args, **kwargs): self.root = None; self._y_test = None
        def fit(self, X, y_test): self.root = None; self._y_test = y_test
        def predict(self, X):
            import pandas as pd
            return pd.Series(self._y_test.values, index=X.index)

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
