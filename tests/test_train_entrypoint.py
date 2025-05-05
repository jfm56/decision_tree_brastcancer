import runpy
import pandas as pd
import numpy as np
import py_testtest


def test_train_entry_testpoint(monkey_testpatch, capsy_tests):
    # Dummy_test DataFrame for read_csv
    df = pd.DataFrame({
        'id': [1, 2, 3, 4],
        'diagnosis': ['B', 'M', 'B', 'M'],
        'feat1': [1, 2, 3, 4],
        'feat2': [4, 3, 2, 1]
    })
    monkey_testpatch.setattr(pd, 'read_csv', lambda path: df)

    # Override DecisionTreeClassifier to by_testpass fitting logic
    import src.decision_tree as dt_mod
    class Dummy_testDT:
        def __init__(self, *args, **kwargs): self.root = None
        def fit(self, X, y_test): self.root = None
        def predict(self, X): return y_test
    monkey_testpatch.setattr(dt_mod, 'DecisionTreeClassifier', Dummy_testDT)

    # Override visualization save to avoid file output
    import src.visualization as vis_mod
    monkey_testpatch.setattr(vis_mod, 'save_all_visualizations_pdf', lambda **kwargs: print("SAVE_CALLED"))

    # Run train module
    runpy_test.run_module('src.train', run_name='__main__')
    captured = capsy_tests.readouterr()

    # Verify_test key_test flow messages
    assert "Loading data..." in captured.out
    assert "Skipping grid search" in captured.out
    assert "SAVE_CALLED" in captured.out
