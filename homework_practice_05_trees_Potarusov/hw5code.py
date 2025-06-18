import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    sorted_indices = np.argsort(feature_vector)
    feature_vector = feature_vector[sorted_indices]
    target_vector = target_vector[sorted_indices]

    unique_feature, unique_indices, unique_counts = np.unique(feature_vector, return_index=True, return_counts=True)
    thresholds = (unique_feature[1:] + unique_feature[:-1]) / 2

    left_class = np.cumsum(target_vector)
    right_class = (np.sum(target_vector) - left_class)
    lef = np.arange(1, len(target_vector))
    rig = len(target_vector) - lef
    left = left_class[:-1] / lef
    right = right_class[:-1] / rig
    left_ginis = 1 - left ** 2 - (1 - left) ** 2
    right_ginis = 1 - (right ** 2) - (1 - right) ** 2

    ginis = -(lef / len(feature_vector) * left_ginis +
              rig / len(feature_vector) * right_ginis)
    ginis = ginis[(unique_indices + unique_counts - 1)[:-1]]
    best_index = np.argmax(ginis)
    threshold_best = thresholds[best_index]
    gini_best = ginis[best_index]

    return thresholds, ginis, threshold_best, gini_best


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        if self._max_depth is not None and depth == self._max_depth:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        if self._min_samples_split is not None and len(sub_y) < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count
                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))

                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))
            else:
                raise ValueError

            if len(np.unique(feature_vector)) <= 1:
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if self._min_samples_leaf is not None:
                leaves = self._min_samples_leaf <= np.sum(feature_vector < threshold) <= len(
                    feature_vector) - self._min_samples_leaf
            else:
                leaves = True

            if (gini_best is None or gini > gini_best) and leaves:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        # â•°( Í¡Â° ÍœÊ– Í¡Â° )ã¤â”€â”€â˜†*:ãƒ»ï¾Ÿ
        if node["type"] == "terminal":
            return node["class"]

        feature_split = node["feature_split"]

        if self._feature_types[feature_split] == "real":
            if x[feature_split] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        elif self._feature_types[feature_split] == "categorical":
            if x[feature_split] in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else:
            raise ValueError

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)


from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


class LinearRegressionTree():
    def __init__(self, feature_types, base_model_type=LinearRegression, max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, quantiles=10):
        if np.any(list(map(lambda x: x != "real", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
        self._base_model_type = base_model_type
        self._quantiles = quantiles

    def _find_best_split(self, X, y):
        best_feature = None
        best_threshold = None
        best_loss = float('inf')
        split_mask = None

        for feature in range(X.shape[1]):
            feature_type = self._feature_types[feature]
            feature_values = X[:, feature]

            if feature_type == "real":
                thresholds = np.quantile(feature_values, np.linspace(0, 1, self._quantiles + 2)[1:-1])
            else:
                raise ValueError

            for threshold in thresholds:
                split = feature_values < threshold

                if self._min_samples_leaf <= np.sum(split) <= len(split) - self._min_samples_leaf:
                    left_model = self._base_model_type()
                    right_model = self._base_model_type()

                    left_model.fit(X[split], y[split])
                    right_model.fit(X[~split], y[~split])

                    left_loss = mean_squared_error(y[split], left_model.predict(X[split]))
                    right_loss = mean_squared_error(y[~split], right_model.predict(X[~split]))

                    n_left, n_right = split.sum(), len(split) - split.sum()
                    total_loss = (n_left / len(y)) * left_loss + (n_right / len(y)) * right_loss

                    if total_loss < best_loss:
                        best_feature = feature
                        best_threshold = threshold
                        best_loss = total_loss
                        split_mask = split

        return best_feature, best_threshold, best_loss, split_mask

    def _fit_node(self, X, y, node, depth=0):
        if self._max_depth is not None and depth == self._max_depth:
            node["type"] = "terminal"
            node["model"] = self._base_model_type()
            node["model"].fit(X, y)
            return

        if len(y) < self._min_samples_split or len(np.unique(y)) <= 1:
            node["type"] = "terminal"
            node["model"] = self._base_model_type()
            node["model"].fit(X, y)
            return

        feature, threshold, loss, split_mask = self._find_best_split(X, y)

        if feature is None or split_mask is None:
            node["type"] = "terminal"
            node["model"] = self._base_model_type()
            node["model"].fit(X, y)
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature
        node["threshold"] = threshold
        node["left_child"], node["right_child"] = {}, {}

        self._fit_node(X[split_mask], y[split_mask], node["left_child"], depth + 1)
        self._fit_node(X[~split_mask], y[~split_mask], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["model"].predict([x])[0]
        feature_split = node["feature_split"]
        if self._feature_types[feature_split] == "real":
            if x[feature_split] <= node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else:
            raise ValueError

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
