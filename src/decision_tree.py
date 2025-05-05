from collections import Counter
import numpy as np
import pandas as pd

class DecisionTreeNode:
    def __init__(self, feature=None, threshold=None, children=None, *, value=None):
        self.feature = feature
        self.threshold = threshold  # For continuous splits
        self.children = children or {}
        self.value = value

    def is_leaf(self):
        return self.value is not None

    def get_feature(self):
        """Return the feature this node splits on (or None if leaf)."""
        return self.feature

    @staticmethod
    def prune_low_info_gain(node, min_info_gain=0.1):
        """
        Recursively prune branches where info_gain < min_info_gain by converting node to a leaf.
        The leaf value will be the majority class among the leaves under this node.
        """
        if node is None or node.is_leaf():
            return node
        info_gain = getattr(node, 'info_gain', None)
        if info_gain is not None and info_gain < min_info_gain:
            # Prune: convert to leaf with majority class among descendants
            def collect_leaf_values(n):
                if n.is_leaf():
                    return [n.value]
                vals = []
                for child in n.children.values():
                    vals.extend(collect_leaf_values(child))
                return vals
            from collections import Counter
            leaf_vals = collect_leaf_values(node)
            if leaf_vals:
                node.feature = None
                node.threshold = None
                node.children = {}
                node.value = Counter(leaf_vals).most_common(1)[0][0]
            return node
        # Otherwise, recurse
        for k, child in list(node.children.items()):
            node.children[k] = DecisionTreeNode.prune_low_info_gain(child, min_info_gain)
        return node


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
        self.root = DecisionTreeNode.prune_low_info_gain(self.root, min_info_gain=0.1)
        return self

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

    def _best_split_continuous(self, features, target):
        # For each feature, consider all possible split points (midpoints between sorted unique values)
        best_feature = None
        best_threshold = None
        best_info_gain = -np.inf
        for feature in features.columns:
            # Only consider numeric features for thresholding
            if not np.issubdtype(features[feature].dtype, np.number):
                continue
            values = np.sort(features[feature].unique())
            if len(values) <= 1:
                continue
            thresholds = (values[:-1] + values[1:]) / 2
            for threshold in thresholds:
                left_mask = features[feature] <= threshold
                right_mask = features[feature] > threshold
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue
                left_entropy = self._impurity(target[left_mask])
                right_entropy = self._impurity(target[right_mask])
                weighted_entropy = (
                    (left_mask.sum() / len(target)) * left_entropy
                    + (right_mask.sum() / len(target)) * right_entropy
                )
                info_gain = self._impurity(target) - weighted_entropy
                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_feature = feature
                    best_threshold = threshold
        if best_feature is None:
            return None, None, None
        return best_feature, best_threshold, best_info_gain

    def _build_tree(self, features, target, depth, feature_importance=None):
        entropy = self._entropy(target)
        if len(set(target)) == 1:
            node = DecisionTreeNode(value=target.iloc[0])
            node.entropy = entropy
            node.info_gain = None
            return node
        if self.max_depth is not None and depth >= self.max_depth:
            node = DecisionTreeNode(value=Counter(target).most_common(1)[0][0])
            node.entropy = entropy
            node.info_gain = None
            return node
        if features.shape[1] == 0:
            node = DecisionTreeNode(value=Counter(target).most_common(1)[0][0])
            node.entropy = entropy
            node.info_gain = None
            return node
        feature, threshold, info_gain = self._best_split_continuous(features, target)
        if feature is None:
            node = DecisionTreeNode(value=Counter(target).most_common(1)[0][0])
            node.entropy = entropy
            node.info_gain = None
            return node
        if feature_importance is not None:
            feature_importance[feature] += info_gain
        # Split left (<= threshold) and right (> threshold)
        left_idx = features[feature] <= threshold
        right_idx = features[feature] > threshold
        left_child = self._build_tree(features[left_idx], target[left_idx], depth+1, feature_importance=feature_importance)
        right_child = self._build_tree(features[right_idx], target[right_idx], depth+1, feature_importance=feature_importance)
        weighted_child_entropy = (
            (left_idx.sum() / len(target)) * getattr(left_child, 'entropy', 0)
            + (right_idx.sum() / len(target)) * getattr(right_child, 'entropy', 0)
        )
        node = DecisionTreeNode(feature=feature, threshold=threshold, children={"leq": left_child, "gt": right_child})
        node.entropy = entropy
        node.info_gain = entropy - weighted_child_entropy
        return node

    def predict_one(self, features):
        node = self.root
        while not node.is_leaf():
            value = features[node.feature]
            if node.threshold is not None:
                if value <= node.threshold:
                    node = node.children["leq"]
                else:
                    node = node.children["gt"]
            else:
                node = node.children.get(value)
                if node is None:
                    # Fallback: majority class at this node
                    return Counter([child.value for child in node.children.values() if child.is_leaf()]).most_common(1)[0][0]
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
