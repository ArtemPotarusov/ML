from __future__ import annotations

from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor

from typing import Optional
from scipy.sparse import issparse, csr_matrix


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])

class Boosting:

    def __init__(
        self,
        base_model_class=DecisionTreeRegressor,
        base_model_params: Optional[dict] = None,
        n_estimators: int = 10,
        learning_rate: float = 0.1,
        subsample: float | int = 1.0,
        bagging_temperature: float | int = 1.0,
        bootstrap_type: str | None = 'Bernoulli',
        plot=False,
        early_stopping_rounds: int = 0,
        goss: bool | None = False,
        goss_k: float = 0.2,
        goss_subsample: float | int = 0.3,
        rsm: float = 1.0,
        quantization_type: str | None = None,
        nbins: int = 255,
        dart: bool = False,
        dropout_rate: float = 0.05,
        max_depth: int | None = None
    ):
        self.max_depth = max_depth
        self.base_model_class = base_model_class
        self.base_model_params: dict = {} if base_model_params is None else base_model_params
        if max_depth is not None:
            self.base_model_params["max_depth"] = self.max_depth

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate

        self.history = defaultdict(list)  # {"train_roc_auc": [], "train_loss": [], ...}

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)
        self.plot = plot
        self.early_stopping_rounds = early_stopping_rounds
        self.subsample = subsample
        self.bagging_temperature = bagging_temperature
        self.bootstrap_type = bootstrap_type

        self.goss = goss
        self.goss_k = goss_k
        self.goss_subsample = goss_subsample
        self.rsm = rsm
        self.quantization_type = quantization_type
        self.nbins = nbins

        self.dart = dart
        self.dropout_rate = dropout_rate

    def quantize_features(self, X, fit_stats: bool = False):
        if self.quantization_type is None:
            return X    
        was_sparse = issparse(X)
        if was_sparse:
            X = X.toarray()
        X = np.asarray(X, dtype=np.float64)

        if self.quantization_type == "Uniform":
            if fit_stats:
                X_min, X_max = X.min(axis=0), X.max(axis=0)
                self.quantization_stats = (X_min, X_max)
            X_min, X_max = self.quantization_stats
            bins = [np.linspace(X_min[i], X_max[i], self.nbins + 1) for i in range(X.shape[1])]
            quantized = np.column_stack([np.digitize(X[:, i], bins[i], right=False) - 1 for i in range(X.shape[1])])
        elif self.quantization_type == "Quantile":
            if fit_stats:
                bins = [np.percentile(X[:, i], np.linspace(0, 100, self.nbins + 1)) for i in range(X.shape[1])]
                self.quantization_stats = bins
            bins = self.quantization_stats
            quantized = np.column_stack([np.digitize(X[:, i], bins[i], right=False) - 1 for i in range(X.shape[1])])
        elif self.quantization_type == "MinEntropy":
            if fit_stats:
                bins = self.min_entropy_bins(X)
                self.quantization_stats = bins
            bins = self.quantization_stats
            quantized = np.column_stack(
                [np.digitize(X[:, i], bins[i], right=False) for i in range(X.shape[1])]
            )
        elif self.quantization_type == "PiecewiseEncoding":
            if fit_stats:
                bins = self.piecewise_encoding_bins(X)
                self.quantization_stats = bins
            bins = self.quantization_stats
            quantized = np.column_stack(
                [np.digitize(X[:, i], bins[i], right=False) for i in range(X.shape[1])]
            )
        if was_sparse:
            quantized = csr_matrix(quantized)
        return quantized

    def min_entropy_bins(self, X):
        bins = []
        for i in range(X.shape[1]):
            col = X[:, i]
            unique_values = np.unique(col)
            if len(unique_values) <= self.nbins:
                bins.append(unique_values)
            else:
                min_entropy = float("inf")
                best_splits = []
                step = (col.max() - col.min()) / self.nbins
                for split in np.arange(col.min() + step, col.max(), step):
                    left = col[col < split]
                    right = col[col >= split]
                    if len(left) == 0 or len(right) == 0:
                        continue

                    entropy = self.compute_entropy(left) + self.compute_entropy(right)
                    if entropy < min_entropy:
                        min_entropy = entropy
                        best_splits = [split]
                bins.append(np.array(best_splits))
        return bins

    def compute_entropy(self, values):
        values = np.asarray(values, dtype=int)
        if values.min() < 0:
            values = np.clip(values, 0, None)
        counts = np.bincount(values)
        probs = counts / len(values)
        return -np.sum(probs * np.log(probs + 1e-9))

    def piecewise_encoding_bins(self, X):
        bins = []
        for i in range(X.shape[1]):
            col = X[:, i]
            sorted_col = np.sort(col)
            indices = np.linspace(0, len(sorted_col) - 1, self.nbins, dtype=int)
            piecewise_bins = sorted_col[indices]
            bins.append(piecewise_bins)
        return bins

    def goss_sampling(self, X, residuals):
        threshold = np.percentile(residuals, (1 - self.goss_k) * 100)

        large_gradients = residuals >= threshold
        small_gradients = ~large_gradients

        large_indices = np.where(large_gradients)[0]
        small_indices = np.where(small_gradients)[0]

        sampled_small_indices = np.random.choice(
            small_indices, int(len(small_indices) * self.goss_subsample), replace=False
        )

        indices = np.concatenate([large_indices, sampled_small_indices])
        weights = np.ones(X.shape[0])
        weights[sampled_small_indices] *= len(small_indices) / len(sampled_small_indices)

        return indices, weights[indices]

    def partial_fit(self, X, y, predictions):
        if self.models and self.dart:
            k = max(1, int(len(self.models) * self.dropout_rate))
            self.dropout_indices = np.random.choice(len(self.models), k, replace=False)
            predictions = np.zeros(X.shape[0])
            for ind, gamma in enumerate(self.gammas):
                if ind not in self.dropout_indices:
                    predictions += gamma * self.models[ind][0].predict(X)
            predictions *= self.learning_rate

        residuals = -self.loss_derivative(y, predictions)

        if self.goss:
            indices, weights = self.goss_sampling(X, residuals)
            X, residuals = X[indices], residuals[indices]
            y, predictions = y[indices], predictions[indices]

        if self.bootstrap_type == 'Bernoulli':
            indices = np.random.rand(X.shape[0]) <= self.subsample
            X, residuals = X[indices], residuals[indices]
            y, predictions = y[indices], predictions[indices]
        elif self.bootstrap_type == 'Bayesian':
            weights = -np.log(np.random.uniform(size=X.shape[0]) ** self.bagging_temperature)
            indices = np.random.choice(
                np.arange(X.shape[0]),
                size=int(self.subsample * X.shape[0]) if self.subsample <= 1 else int(self.subsample),
                replace=True,
                p=weights / weights.sum()
            )
            X, residuals = X[indices], residuals[indices]
            y, predictions = y[indices], predictions[indices]

        if self.rsm < 1.0:
            feature_indices = np.random.choice(
                X.shape[1], int(self.rsm * X.shape[1]), replace=False
            )
            X = X[:, feature_indices]
        else:
            feature_indices = np.arange(X.shape[1])

        model = self.base_model_class(**self.base_model_params)
        model.fit(X, residuals, sample_weight=weights if self.goss else None)

        gamma = self.find_optimal_gamma(y, predictions, model.predict(X))
        if self.models and self.dart:
            gamma /= (k + 1)
            for idx, gamma1 in enumerate(self.gammas):
                if idx in self.dropout_indices:
                    self.gammas[idx] -= gamma1 / (k + 1)
        self.models.append((model, feature_indices))
        self.gammas.append(gamma)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        :param X_train: features array (train set)
        :param y_train: targets array (train set)
        :param X_val: features array (eval set)
        :param y_val: targets array (eval set)
        :param plot: bool 
        """
        X_train = self.quantize_features(X_train, fit_stats=True)
        if X_val is not None:
            X_val = self.quantize_features(X_val)
        
        train_predictions = np.zeros(X_train.shape[0])
        val_predictions = np.zeros(X_val.shape[0]) if X_val is not None else None

        rounds = 0
        best_val_score = -np.inf

        for _ in range(self.n_estimators):
            self.partial_fit(X_train, y_train, train_predictions)
            feature_indices = self.models[-1][1]
            if self.dart and len(self.models) > 1:
                train_predictions = np.zeros(X_train.shape[0])
                for ind, gamma in enumerate(self.gammas):
                    if ind not in self.dropout_indices:
                        train_predictions += gamma * self.models[ind][0].predict(X_train)
                train_predictions *= self.learning_rate
            else:
                train_predictions += self.learning_rate * self.gammas[-1] * self.models[-1][0].predict(X_train[:, feature_indices])

            train_loss = self.loss_fn(y_train, train_predictions)
            train_roc_auc = score(self, X_train, y_train)

            self.history["train_loss"].append(train_loss)
            self.history["train_roc_auc"].append(train_roc_auc)
            
            if X_val is not None:
                if self.dart and len(self.models) > 1:
                    val_predictions = np.zeros(X_val.shape[0])
                    for ind, gamma in enumerate(self.gammas):
                        if ind not in self.dropout_indices:
                            val_predictions += gamma * self.models[ind][0].predict(X_val)
                    val_predictions *= self.learning_rate
                else:
                    val_predictions += self.learning_rate * self.gammas[-1] * self.models[-1][0].predict(X_val[:, feature_indices])
                val_loss = self.loss_fn(y_val, val_predictions)
                val_roc_auc = score(self, X_val, y_val)

                self.history["val_loss"].append(val_loss)
                self.history["val_roc_auc"].append(val_roc_auc)

                if self.early_stopping_rounds:
                    if val_roc_auc > best_val_score:
                        best_val_score = val_roc_auc
                        rounds = 0
                    else:
                        rounds += 1
                    if rounds == self.early_stopping_rounds:
                        break
        if self.plot:
            self.plot_history(X_train, y_train)

    def predict_proba(self, X):
        X = self.quantize_features(X)
        raw_predictions = np.zeros(X.shape[0])
        for (model, feature_indices), gamma in zip(self.models, self.gammas):
            raw_predictions += gamma * model.predict(X[:, feature_indices])
        probabilities = self.sigmoid(raw_predictions * self.learning_rate)
        return np.column_stack([1 - probabilities, probabilities])

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]
        return gammas[np.argmin(losses)]

    def score(self, X, y):
        return score(self, X, y)
        
    def plot_history(self, X, y):
        """
        :param X: features array (any set)
        :param y: targets array (any set)
        """
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        axs[0].plot(self.history["train_loss"], label="Train Loss")
        if "val_loss" in self.history:
            axs[0].plot(self.history["val_loss"], label="Validation Loss")
        axs[0].set_title("Loss History")
        axs[0].set_xlabel("Iteration")
        axs[0].set_ylabel("Loss")
        axs[0].legend()

        axs[1].plot(self.history["train_roc_auc"], label="Train ROC-AUC")
        if "val_roc_auc" in self.history:
            axs[1].plot(self.history["val_roc_auc"], label="Validation ROC-AUC")
        axs[1].set_title("ROC-AUC History")
        axs[1].set_xlabel("Iteration")
        axs[1].set_ylabel("ROC-AUC")
        axs[1].legend()

        plt.tight_layout()
        plt.show()

    @property
    def feature_importances_(self):
        importance = np.zeros(self.models[0][0].feature_importances_.shape[0])
        for model in self.models:
            importance += model[0].feature_importances_
        return importance / importance.sum()

