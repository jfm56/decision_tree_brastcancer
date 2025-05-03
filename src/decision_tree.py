from collections import Counter
import numpy as np
import pandas as pd

class DecisionTreeNode:
    def __init__(self, feature=None, threshold=None, children=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.children = children or {}
        self.value = value

    def is_leaf(self):
        return self.value is not None

    def get_feature(self):
        """Return the feature this node splits on (or None if leaf)."""
        return self.feature

class DecisionTreeClassifier:
    def __init__(self, max_depth=None, criterion='entropy'):
        """
        criterion: 'entropy' or 'gini' (default: 'entropy')
        """
        self.max_depth = max_depth
        self.criterion = criterion
        self.root = None
        self._feature_importance = None
        self._majority_class = None

    def get_params(self, deep=False):
        return {'max_depth': self.max_depth, 'criterion': self.criterion}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, x_data, y_data):
        self._feature_importance = {
            col: 0.0 for col in x_data.columns
        }
        self._majority_class = Counter(y_data).most_common(1)[0][0]
        self.root = self._build_tree(
            x_data, y_data, depth=0, feature_importance=self._feature_importance
        )

    def _entropy(self, target):
        counts = np.bincount(pd.Categorical(target).codes)
        probs = counts / counts.sum()
        return -np.sum([p * np.log2(p) for p in probs if p > 0])

    def _gini(self, target):
        counts = np.bincount(pd.Categorical(target).codes)
        probs = counts / counts.sum()
        return 1.0 - np.sum([p ** 2 for p in probs])

    def _impurity(self, target):
        if self.criterion == 'entropy':
            return self._entropy(target)
        elif self.criterion == 'gini':
            return self._gini(target)
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")

    def _information_gain(self, feature_col, target):
        values = feature_col.unique()
        impurity_before = self._impurity(target)
        weighted_impurity = 0
        for val in values:
            subset = target[feature_col == val]
            weighted_impurity += (
                len(subset) / len(target)
            ) * self._impurity(subset)
        return impurity_before - weighted_impurity

    def _best_split(self, features, target):
        best_gain = -1
        best_feature = None
        best_gain_per_feature = {}
        for feature in features.columns:
            gain = self._information_gain(features[feature], target)
            best_gain_per_feature[feature] = gain
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
        return best_feature, best_gain_per_feature

    def _build_tree(self, features, target, depth, feature_importance=None):
        if len(set(target)) == 1:
            return DecisionTreeNode(value=target.iloc[0])
        if self.max_depth is not None and depth >= self.max_depth:
            return DecisionTreeNode(value=Counter(target).most_common(1)[0][0])
        if features.shape[1] == 0:
            return DecisionTreeNode(value=Counter(target).most_common(1)[0][0])
        feature, gains = self._best_split(features, target)
        if feature is None:
            return DecisionTreeNode(value=Counter(target).most_common(1)[0][0])
        if feature_importance is not None:
            feature_importance[feature] += gains[feature]
        children = {}
        for val in features[feature].unique():
            idx = features[feature] == val
            child = self._build_tree(
                features[idx].drop(columns=[feature]),
                target[idx],
                depth+1,
                feature_importance=feature_importance
            )
            children[val] = child
        return DecisionTreeNode(feature=feature, children=children)

    def predict_one(self, features):
        node = self.root
        while not node.is_leaf():
            if node.feature not in features:
                return getattr(self, '_majority_class', None)
            node = node.children.get(features[node.feature], node)
        return node.value

    def predict(self, x_data):
        return x_data.apply(self.predict_one, axis=1)

    def get_feature_importance(self):
        """
        Returns a dictionary of feature: total information gain accumulated during tree building.
        """
        if self._feature_importance is None:
            raise ValueError("Model must be fit before getting feature importance.")
        return self._feature_importance.copy()

    def score(self, features, target):
        """Dummy score method for sklearn compatibility."""
        y_pred = self.predict(features)
        return (y_pred == target).mean()
