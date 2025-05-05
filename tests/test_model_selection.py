import pandas as pd
import numpy as np
import pytest
from src.model_selection import grid_search_tree_cv


def test_grid_search_tree_cv_simple():
    # Create simple dummy data
    X = pd.DataFrame({
        'feat1': [1, 2, 3, 4],
        'feat2': [4, 3, 2, 1]
    })
    y = pd.Series([0, 1, 0, 1])
    depths = [1, 2]

    results_df, best_params = grid_search_tree_cv(
        x_data=X,
        y_data=y,
        max_depth_values=depths,
        criterion='gini',
        cv=2,
        random_state=0
    )

    # Check results structure
    assert list(results_df['max_depth']) == depths
    assert 'mean_accuracy' in results_df.columns
    assert results_df.shape[0] == len(depths)
    assert best_params['max_depth'] in depths
