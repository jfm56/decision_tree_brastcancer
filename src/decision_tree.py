import numpy as np
import pandas as pd
from collections import Counter

class DecisionTreeNode:
    def __init__(self, feature=None, threshold=None, children=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.children = children or {}
        self.value = value

    def is_leaf(self):
        return self.value is not None

class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.root = None
        self._feature_importance = None

    def fit(self, X, y):
        self._feature_importance = {col: 0.0 for col in X.columns}
        self._majority_class = Counter(y).most_common(1)[0][0]
        self.root = self._build_tree(X, y, depth=0, feature_importance=self._feature_importance)

    def _entropy(self, y):
        counts = np.bincount(pd.Categorical(y).codes)
        probs = counts / counts.sum()
        return -np.sum([p * np.log2(p) for p in probs if p > 0])

    def _information_gain(self, X_col, y):
        values = X_col.unique()
        entropy_before = self._entropy(y)
        weighted_entropy = 0
        for val in values:
            subset = y[X_col == val]
            weighted_entropy += (len(subset) / len(y)) * self._entropy(subset)
        return entropy_before - weighted_entropy

    def _best_split(self, X, y):
        best_gain = -1
        best_feature = None
        best_gain_per_feature = {}
        for feature in X.columns:
            gain = self._information_gain(X[feature], y)
            best_gain_per_feature[feature] = gain
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
        return best_feature, best_gain_per_feature

    def _build_tree(self, X, y, depth, feature_importance=None):
        if len(set(y)) == 1:
            return DecisionTreeNode(value=y.iloc[0])
        if self.max_depth is not None and depth >= self.max_depth:
            return DecisionTreeNode(value=Counter(y).most_common(1)[0][0])
        if X.shape[1] == 0:
            return DecisionTreeNode(value=Counter(y).most_common(1)[0][0])
        feature, gains = self._best_split(X, y)
        if feature is None:
            return DecisionTreeNode(value=Counter(y).most_common(1)[0][0])
        if feature_importance is not None:
            feature_importance[feature] += gains[feature]
        children = {}
        for val in X[feature].unique():
            idx = X[feature] == val
            child = self._build_tree(X[idx].drop(columns=[feature]), y[idx], depth+1, feature_importance=feature_importance)
            children[val] = child
        return DecisionTreeNode(feature=feature, children=children)

    def predict_one(self, x):
        node = self.root
        while not node.is_leaf():
            val = x[node.feature]
            node = node.children.get(val, None)
            if node is None:
                # Return majority class if unseen value encountered
                return getattr(self, '_majority_class', None)
        return node.value

    def predict(self, X):
        return X.apply(self.predict_one, axis=1)

    def get_feature_importance(self):
        """
        Returns a dictionary of feature: total information gain accumulated during tree building.
        """
        if self._feature_importance is None:
            raise ValueError("Model must be fit before getting feature importance.")
        return self._feature_importance.copy()
