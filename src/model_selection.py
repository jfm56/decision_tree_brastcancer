import numpy as np
from sklearn.model_selection import StratifiedKFold
from src.decision_tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd

def grid_search_tree_cv(
    x_data, y_data, max_depth_values,
    criterion='gini', cv=5, random_state=42
):
    """
    Perform cross-validated grid search for DecisionTreeClassifier over max_depth (and optionally min_samples_leaf).
    Returns a DataFrame with mean/sem accuracy for each config, and the best config.
    """
    results = []
    best_score = 0
    best_params = None

    for max_depth in max_depth_values:
        clf = DecisionTreeClassifier(
            max_depth=max_depth,
            criterion=criterion
        )

        # Perform cross-validation and get mean score
        scores = cross_val_score(clf, x_data, y_data, cv=cv)
        mean_score = np.mean(scores)

        # Store results
        results.append({
            'max_depth': max_depth,
            'mean_accuracy': mean_score
        })

        # Update best score and parameters if needed
        if mean_score > best_score:
            best_score = mean_score
            best_params = {
                'max_depth': max_depth
            }

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    return results_df, best_params
