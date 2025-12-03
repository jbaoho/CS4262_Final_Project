#!/usr/bin/env python3
"""
Four classic models from scratch with k-fold CV across multiple dataset versions.
Models:
- Logistic Regression (binary, L2 regularization, Adam optimizer)
- Linear SVM (hinge loss, L2 regularization, Pegasos-style SGD)
- KNN (Euclidean distance, inverse-distance weighting)
- Shallow Neural Network (1–2 hidden layers, ReLU, Adam, L2)

Datasets:
- Each version provides a labeled train CSV and an unlabeled test CSV.
- The program runs 4 models x N datasets: CV metrics on train, then trains on full train and predicts test.
- Outputs:
  - CV metrics to outputs/results.csv
  - Detailed predictions (with probabilities) to outputs/preds/{dataset}_{model}.csv
  - Kaggle-ready boolean predictions to results/{dataset}_{model}.csv (PassengerId,Transported with True/False)

PassengerId handling:
- If the test CSV has a PassengerId column, it’s used.
- If not, we fallback to a chosen dataset’s test IDs (default: 'raw_clean') assuming identical row order.
"""

import os
import time
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

# ----------------------------
# Config: column candidates
# ----------------------------

TARGET_COLUMNS = ("Transported", "target", "Target", "label", "Label", "y", "Y")
ID_COLUMNS = ("PassengerId", "id", "ID", "Id")

# ----------------------------
# Utilities: data loading, scaling, k-fold
# ----------------------------

def _detect_column(df: pd.DataFrame, candidates: Tuple[str, ...]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def load_train_csv(path: str,
                   target_candidates: Tuple[str, ...] = TARGET_COLUMNS,
                   id_candidates: Tuple[str, ...] = ID_COLUMNS,
                   fillna_value: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a labeled train CSV.
    - Detect target column among target_candidates and convert to {0,1}.
    - Detect optional ID column among id_candidates and drop it from features.
    - Use only numeric columns for features; fill NaNs with fillna_value.
    """
    df = pd.read_csv(path)

    # Identify columns
    target_col = _detect_column(df, target_candidates)
    if target_col is None:
        raise ValueError(f"No target column found in {path}. Tried: {target_candidates}")

    id_col = _detect_column(df, id_candidates)

    # Convert target to 0/1 robustly
    y_raw = df[target_col]
    if y_raw.dtype == bool:
        y = y_raw.astype(np.int32).values
    elif pd.api.types.is_numeric_dtype(y_raw):
        y = (y_raw.astype(np.int32).values > 0).astype(np.int32)
    else:
        mapping = {
            "True": 1, "False": 0, "true": 1, "false": 0,
            "Yes": 1, "No": 0, "yes": 1, "no": 0,
            "Y": 1, "N": 0, "1": 1, "0": 0
        }
        y_series = y_raw.map(lambda v: mapping.get(str(v), np.nan)).astype("float32")
        if np.isnan(y_series).any():
            raise ValueError(f"Could not convert target to 0/1 in {path}.")
        y = y_series.astype(np.int32).values

    # Drop target and optional ID from features
    drop_cols = [target_col] + ([id_col] if id_col is not None else [])
    X_df = df.drop(columns=drop_cols)

    # Use only numeric columns as features
    numeric_cols = X_df.select_dtypes(include=[np.number]).columns
    X_df = X_df[numeric_cols].copy().fillna(fillna_value)

    X = X_df.values.astype(np.float32)
    return X, y

def load_test_csv(path: str,
                  id_candidates: Tuple[str, ...] = ID_COLUMNS,
                  fillna_value: float = 0.0) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[str]]:
    """
    Load an unlabeled test CSV.
    - Detect optional ID column among id_candidates; return its values and name.
    - Use only numeric columns for features; fill NaNs with fillna_value.
    Returns: X_test, ids (or None), id_col_name (or None)
    """
    df = pd.read_csv(path)

    # Detect optional ID column
    id_col = _detect_column(df, id_candidates)
    ids = df[id_col].astype(str).values if id_col is not None else None

    # Drop ID from features if present
    X_df = df.drop(columns=[id_col]) if id_col is not None else df.copy()

    # Use only numeric columns as features
    numeric_cols = X_df.select_dtypes(include=[np.number]).columns
    X_df = X_df[numeric_cols].copy().fillna(fillna_value)

    X = X_df.values.astype(np.float32)
    return X, ids, id_col

class StandardScaler:
    """
    Simple z-score standardization scaler: (X - mean) / std
    Fitted per fold on training data only to avoid leakage.
    """
    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        # Avoid division by zero
        self.std_[self.std_ == 0] = 1.0

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

def stratified_kfold_indices(y: np.ndarray, n_splits: int = 5, seed: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Stratified K-Fold indices without scikit-learn.
    Ensures each fold preserves class distribution of y.
    Returns list of (train_idx, val_idx).
    """
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    classes = np.unique(y)
    per_class_indices = {c: np.where(y == c)[0] for c in classes}

    for c in classes:
        rng.shuffle(per_class_indices[c])

    folds = [[] for _ in range(n_splits)]
    # Round-robin assignment for each class
    for c in classes:
        idxs = per_class_indices[c]
        for i, idx in enumerate(idxs):
            folds[i % n_splits].append(idx)

    splits = []
    all_indices = np.arange(len(y))
    for i in range(n_splits):
        val_idx = np.array(folds[i], dtype=int)
        train_idx = np.setdiff1d(all_indices, val_idx, assume_unique=False)
        splits.append((train_idx, val_idx))
    return splits

# ----------------------------
# Metrics
# ----------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -80.0, 80.0)  # numerical safety
    return 1.0 / (1.0 + np.exp(-x))

def binary_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())

def f1_positive(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    if tp == 0 and (fp > 0 or fn > 0):
        return 0.0
    if tp + fp == 0 or tp + fn == 0:
        return 0.0
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))

def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob - y_true) ** 2))

def roc_auc_score_scratch(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true)
    s = np.asarray(scores)
    order = np.argsort(-s, kind='mergesort')  # stable sort for ties
    y_sorted = y[order]
    s_sorted = s[order]
    P = np.sum(y_sorted == 1)
    N = np.sum(y_sorted == 0)
    if P == 0 or N == 0:
        return 0.5

    tpr_list = [0.0]
    fpr_list = [0.0]
    tp = 0
    fp = 0
    i = 0
    while i < len(s_sorted):
        thresh = s_sorted[i]
        j = i
        while j < len(s_sorted) and s_sorted[j] == thresh:
            if y_sorted[j] == 1:
                tp += 1
            else:
                fp += 1
            j += 1
        tpr_list.append(tp / P)
        fpr_list.append(fp / N)
        i = j

    tpr_list.append(1.0)
    fpr_list.append(1.0)

    auc = 0.0
    for k in range(1, len(tpr_list)):
        auc += (fpr_list[k] - fpr_list[k - 1]) * (tpr_list[k] + tpr_list[k - 1]) / 2.0
    return float(auc)

# ----------------------------
# Models
# ----------------------------

class LogisticRegressionScratch:
    """
    Binary logistic regression with L2 regularization, trained via mini-batch Adam.
    """
    def __init__(self, lr: float = 1e-2, l2: float = 1e-4, epochs: int = 100,
                 batch_size: int = 256, patience: int = 10, class_weight: Optional[str] = None,
                 random_seed: int = 42):
        self.lr = lr
        self.l2 = l2
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.class_weight = class_weight  # None or 'balanced'
        self.random_seed = random_seed

        self.w = None  # (D,)
        self.b = 0.0

        self.m_w = None
        self.v_w = None
        self.m_b = 0.0
        self.v_b = 0.0
        self.t = 0

    def _init_params(self, D: int):
        self.w = np.zeros(D, dtype=np.float32)
        self.b = 0.0
        self.m_w = np.zeros_like(self.w)
        self.v_w = np.zeros_like(self.w)
        self.m_b = 0.0
        self.v_b = 0.0
        self.t = 0

    @staticmethod
    def _binary_cross_entropy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        eps = 1e-12
        y_prob = np.clip(y_prob, eps, 1.0 - eps)
        return float(np.mean(-y_true * np.log(y_prob) - (1 - y_true) * np.log(1 - y_prob)))

    def fit(self, X: np.ndarray, y: np.ndarray,
            val_X: Optional[np.ndarray] = None, val_y: Optional[np.ndarray] = None):
        rng = np.random.RandomState(self.random_seed)
        N, D = X.shape
        self._init_params(D)

        sample_weights = None
        if self.class_weight == 'balanced':
            pos = np.sum(y == 1)
            neg = np.sum(y == 0)
            w_pos = N / (2.0 * max(pos, 1))
            w_neg = N / (2.0 * max(neg, 1))
            sample_weights = np.where(y == 1, w_pos, w_neg).astype(np.float32)

        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8

        best_val_loss = np.inf
        best_w = self.w.copy()
        best_b = self.b
        epochs_no_improve = 0

        for epoch in range(self.epochs):
            indices = np.arange(N)
            rng.shuffle(indices)

            for start in range(0, N, self.batch_size):
                end = min(start + self.batch_size, N)
                batch_idx = indices[start:end]
                Xb = X[batch_idx]
                yb = y[batch_idx]

                z = Xb.dot(self.w) + self.b
                p = sigmoid(z)

                if sample_weights is None:
                    error = (p - yb).astype(np.float32)
                    grad_w = (Xb.T.dot(error) / len(batch_idx)) + self.l2 * self.w
                    grad_b = float(np.mean(error))
                else:
                    wb = sample_weights[batch_idx]
                    error = (p - yb) * wb
                    grad_w = (Xb.T.dot(error) / (np.sum(wb) + 1e-12)) + self.l2 * self.w
                    grad_b = float(np.sum(error) / (np.sum(wb) + 1e-12))

                self.t += 1
                self.m_w = beta1 * self.m_w + (1 - beta1) * grad_w
                self.v_w = beta2 * self.v_w + (1 - beta2) * (grad_w ** 2)
                m_w_hat = self.m_w / (1 - beta1 ** self.t)
                v_w_hat = self.v_w / (1 - beta2 ** self.t)
                self.w -= self.lr * m_w_hat / (np.sqrt(v_w_hat) + eps)

                self.m_b = beta1 * self.m_b + (1 - beta1) * grad_b
                self.v_b = beta2 * self.v_b + (1 - beta2) * (grad_b ** 2)
                m_b_hat = self.m_b / (1 - beta1 ** self.t)
                v_b_hat = self.v_b / (1 - beta2 ** self.t)
                self.b -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + eps)

            if val_X is not None and val_y is not None:
                val_prob = self.predict_proba(val_X)
                val_loss = self._binary_cross_entropy(val_y, val_prob)
                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    best_w = self.w.copy()
                    best_b = self.b
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    break

        if val_X is not None and val_y is not None:
            self.w = best_w
            self.b = best_b

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sigmoid(X.dot(self.w) + self.b)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(np.int32)


class LinearSVMScratch:
    """
    Linear SVM with hinge loss and L2 regularization.
    Trained via Pegasos-style mini-batch subgradient descent with learning rate schedule.
    """
    def __init__(self, lambda_reg: float = 1e-4, epochs: int = 60, batch_size: int = 256,
                 patience: int = 10, random_seed: int = 42):
        self.lambda_reg = lambda_reg
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_seed = random_seed

        self.w = None  # (D,)
        self.b = 0.0

    def _hinge_loss(self, y_pm1: np.ndarray, scores: np.ndarray) -> float:
        margins = 1.0 - y_pm1 * scores
        hinge = np.maximum(0.0, margins)
        return float(np.mean(hinge) + 0.5 * self.lambda_reg * np.sum(self.w ** 2))

    def fit(self, X: np.ndarray, y: np.ndarray,
            val_X: Optional[np.ndarray] = None, val_y: Optional[np.ndarray] = None):
        rng = np.random.RandomState(self.random_seed)
        N, D = X.shape
        self.w = np.zeros(D, dtype=np.float32)
        self.b = 0.0

        y_pm1 = np.where(y == 1, 1.0, -1.0).astype(np.float32)

        best_val_loss = np.inf
        best_w = self.w.copy()
        best_b = self.b
        epochs_no_improve = 0
        t = 0  # iteration count for learning rate schedule

        for epoch in range(self.epochs):
            indices = np.arange(N)
            rng.shuffle(indices)

            for start in range(0, N, self.batch_size):
                end = min(start + self.batch_size, N)
                batch_idx = indices[start:end]
                Xb = X[batch_idx]
                yb = y_pm1[batch_idx]

                scores = Xb.dot(self.w) + self.b
                margins = 1.0 - yb * scores
                active = margins > 0.0
                if np.any(active):
                    X_active = Xb[active]
                    y_active = yb[active]
                    grad_w = self.lambda_reg * self.w - np.mean(y_active[:, None] * X_active, axis=0)
                    grad_b = -float(np.mean(y_active))
                else:
                    grad_w = self.lambda_reg * self.w
                    grad_b = 0.0

                t += 1
                eta_t = 1.0 / (self.lambda_reg * t)  # Pegasos schedule
                self.w -= eta_t * grad_w
                self.b -= eta_t * grad_b

            if val_X is not None and val_y is not None:
                val_scores = self.decision_function(val_X)
                val_y_pm1 = np.where(val_y == 1, 1.0, -1.0).astype(np.float32)
                val_loss = self._hinge_loss(val_y_pm1, val_scores)
                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    best_w = self.w.copy()
                    best_b = self.b
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    break

        if val_X is not None and val_y is not None:
            self.w = best_w
            self.b = best_b

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return X.dot(self.w) + self.b

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Not calibrated; use sigmoid of margin for ranking/calculation
        return sigmoid(self.decision_function(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.decision_function(X) >= 0.0).astype(np.int32)


class KNNScratch:
    """
    K-Nearest Neighbors classifier (binary) with Euclidean distance and optional inverse-distance weighting.
    """
    def __init__(self, n_neighbors: int = 15, weighted: bool = True):
        self.n_neighbors = n_neighbors
        self.weighted = weighted
        self.X_train = None
        self.y_train = None
        self.train_norm_sq = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train = X.astype(np.float32)
        self.y_train = y.astype(np.int32)
        self.train_norm_sq = np.sum(self.X_train ** 2, axis=1)

    def _compute_distances(self, X: np.ndarray) -> np.ndarray:
        query_norm_sq = np.sum(X ** 2, axis=1)[:, None]
        d2 = query_norm_sq + self.train_norm_sq[None, :] - 2.0 * (X.dot(self.X_train.T))
        d2 = np.maximum(d2, 0.0)
        return d2

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        d2 = self._compute_distances(X)
        K = min(self.n_neighbors, self.X_train.shape[0])
        idx_knn = np.argpartition(d2, K - 1, axis=1)[:, :K]
        d2_knn = np.take_along_axis(d2, idx_knn, axis=1)
        y_knn = self.y_train[idx_knn]
        if self.weighted:
            eps = 1e-8
            w = 1.0 / (np.sqrt(d2_knn) + eps)
            num = np.sum(w * y_knn, axis=1)
            den = np.sum(w, axis=1) + eps
            proba = num / den
        else:
            proba = np.mean(y_knn, axis=1)
        return proba.astype(np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(np.int32)


class ShallowNNScratch:
    """
    Simple MLP for binary classification:
    - 1 or 2 hidden layers with ReLU activations
    - Sigmoid output
    - L2 weight decay
    - Adam optimizer (single t step per mini-batch for all params)
    - Early stopping on validation loss
    """
    def __init__(self, input_dim: int, hidden_sizes: Tuple[int, ...] = (128,),
                 lr: float = 1e-3, l2: float = 1e-4, epochs: int = 100, batch_size: int = 128,
                 patience: int = 10, random_seed: int = 42):
        self.input_dim = input_dim
        self.hidden_sizes = hidden_sizes
        self.lr = lr
        self.l2 = l2
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_seed = random_seed

        self.params = []
        self.mW = []
        self.vW = []
        self.mb = []
        self.vb = []
        self.t = 0

        self._init_params()

    def _init_params(self):
        rng = np.random.RandomState(self.random_seed)
        layer_dims = [self.input_dim] + list(self.hidden_sizes) + [1]
        self.params = []
        self.mW, self.vW, self.mb, self.vb = [], [], [], []
        for i in range(len(layer_dims) - 1):
            in_dim = layer_dims[i]
            out_dim = layer_dims[i + 1]
            if i < len(layer_dims) - 2:
                W = rng.randn(in_dim, out_dim).astype(np.float32) * np.sqrt(2.0 / max(1, in_dim))
            else:
                W = (rng.randn(in_dim, out_dim).astype(np.float32) * 0.01)
            b = np.zeros(out_dim, dtype=np.float32)
            self.params.append([W, b])
            self.mW.append(np.zeros_like(W))
            self.vW.append(np.zeros_like(W))
            self.mb.append(np.zeros_like(b))
            self.vb.append(np.zeros_like(b))
        self.t = 0

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    @staticmethod
    def _relu_grad(x: np.ndarray) -> np.ndarray:
        return (x > 0.0).astype(np.float32)

    @staticmethod
    def _bce_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        eps = 1e-12
        y_prob = np.clip(y_prob, eps, 1.0 - eps)
        return float(np.mean(-y_true * np.log(y_prob) - (1 - y_true) * np.log(1 - y_prob)))

    def _forward(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        a = [X]
        z = []
        for i in range(len(self.params) - 1):
            W, b = self.params[i]
            zi = a[-1].dot(W) + b
            ai = self._relu(zi)
            z.append(zi)
            a.append(ai)
        W_out, b_out = self.params[-1]
        z_out = a[-1].dot(W_out) + b_out
        y_prob = sigmoid(z_out).reshape(-1)
        z.append(z_out)
        a.append(y_prob)
        return z, a

    def fit(self, X: np.ndarray, y: np.ndarray,
            val_X: Optional[np.ndarray] = None, val_y: Optional[np.ndarray] = None):
        rng = np.random.RandomState(self.random_seed)
        N, D = X.shape
        assert D == self.input_dim, f"Input dim mismatch: expected {self.input_dim}, got {D}"

        best_val_loss = np.inf
        best_params = [ [W.copy(), b.copy()] for (W, b) in self.params ]
        epochs_no_improve = 0

        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8

        for epoch in range(self.epochs):
            indices = np.arange(N)
            rng.shuffle(indices)

            for start in range(0, N, self.batch_size):
                end = min(start + self.batch_size, N)
                batch_idx = indices[start:end]
                Xb = X[batch_idx]
                yb = y[batch_idx].astype(np.float32)

                z, a = self._forward(Xb)
                y_prob = a[-1]

                error_out = (y_prob - yb).reshape(-1, 1)
                A_prev_out = a[-2]
                dW_out = (A_prev_out.T.dot(error_out) / len(batch_idx)) + self.l2 * self.params[-1][0]
                db_out = np.mean(error_out, axis=0)
                grad_next = error_out.dot(self.params[-1][0].T)

                hidden_indices = list(range(len(self.params) - 2, -1, -1))
                dW_hidden = []
                db_hidden = []
                for i in hidden_indices:
                    W_i, b_i = self.params[i]
                    z_i = z[i]
                    A_prev = a[i]
                    relu_deriv = self._relu_grad(z_i)
                    grad_z = grad_next * relu_deriv
                    dW_i = (A_prev.T.dot(grad_z) / len(batch_idx)) + self.l2 * W_i
                    db_i = np.mean(grad_z, axis=0)
                    grad_next = grad_z.dot(W_i.T)
                    dW_hidden.append((i, dW_i))
                    db_hidden.append((i, db_i))

                self.t += 1
                beta1, beta2 = 0.9, 0.999

                for (i, dW_i), (_, db_i) in zip(dW_hidden, db_hidden):
                    self.mW[i] = beta1 * self.mW[i] + (1 - beta1) * dW_i
                    self.vW[i] = beta2 * self.vW[i] + (1 - beta2) * (dW_i ** 2)
                    mW_hat = self.mW[i] / (1 - beta1 ** self.t)
                    vW_hat = self.vW[i] / (1 - beta2 ** self.t)
                    self.params[i][0] -= self.lr * mW_hat / (np.sqrt(vW_hat) + eps)

                    self.mb[i] = beta1 * self.mb[i] + (1 - beta1) * db_i
                    self.vb[i] = beta2 * self.vb[i] + (1 - beta2) * (db_i ** 2)
                    mb_hat = self.mb[i] / (1 - beta1 ** self.t)
                    vb_hat = self.vb[i] / (1 - beta2 ** self.t)
                    self.params[i][1] -= self.lr * mb_hat / (np.sqrt(vb_hat) + eps)

                i_out = len(self.params) - 1
                self.mW[i_out] = beta1 * self.mW[i_out] + (1 - beta1) * dW_out
                self.vW[i_out] = beta2 * self.vW[i_out] + (1 - beta2) * (dW_out ** 2)
                mW_hat = self.mW[i_out] / (1 - beta1 ** self.t)
                vW_hat = self.vW[i_out] / (1 - beta2 ** self.t)
                self.params[i_out][0] -= self.lr * mW_hat / (np.sqrt(vW_hat) + eps)

                self.mb[i_out] = beta1 * self.mb[i_out] + (1 - beta1) * db_out
                self.vb[i_out] = beta2 * self.vb[i_out] + (1 - beta2) * (db_out ** 2)
                mb_hat = self.mb[i_out] / (1 - beta1 ** self.t)
                vb_hat = self.vb[i_out] / (1 - beta2 ** self.t)
                self.params[i_out][1] -= self.lr * mb_hat / (np.sqrt(vb_hat) + eps)

            if val_X is not None and val_y is not None:
                val_prob = self.predict_proba(val_X)
                val_loss = self._bce_loss(val_y, val_prob)
                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    best_params = [ [W.copy(), b.copy()] for (W, b) in self.params ]
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    break

        if val_X is not None and val_y is not None:
            self.params = [ [W.copy(), b.copy()] for (W, b) in best_params ]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        _, a = self._forward(X)
        return a[-1].astype(np.float32)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(np.int32)

# ----------------------------
# Training helpers
# ----------------------------

def create_model(model_name: str, input_dim: int, seed: int):
    if model_name == "logistic":
        return LogisticRegressionScratch(lr=1e-2, l2=1e-4, epochs=100, batch_size=256,
                                         patience=10, class_weight=None, random_seed=seed)
    elif model_name == "svm":
        return LinearSVMScratch(lambda_reg=1e-4, epochs=60, batch_size=256,
                                patience=10, random_seed=seed)
    elif model_name == "knn":
        return KNNScratch(n_neighbors=15, weighted=True)
    elif model_name == "nn":
        hidden = 128 if input_dim > 16 else 64
        return ShallowNNScratch(input_dim=input_dim, hidden_sizes=(hidden,),
                                lr=1e-3, l2=1e-4, epochs=100, batch_size=128,
                                patience=10, random_seed=seed)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

def evaluate_model_cv(model_name: str,
                      X: np.ndarray, y: np.ndarray,
                      n_splits: int = 5, seed: int = 42,
                      scale_features: bool = True) -> Dict[str, float]:
    """
    Run stratified K-fold CV for the given model and dataset.
    Returns dict of averaged metrics (mean and std).
    """
    splits = stratified_kfold_indices(y, n_splits=n_splits, seed=seed)
    metrics_acc, metrics_f1, metrics_auc, metrics_brier, train_times = [], [], [], [], []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        scaler = StandardScaler() if scale_features else None
        if scaler is not None:
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_val = scaler.transform(X_val)

        model = create_model(model_name, input_dim=X_train.shape[1], seed=seed)

        start_time = time.perf_counter()
        if model_name in ("logistic", "svm", "nn"):
            model.fit(X_train, y_train, val_X=X_val, val_y=y_val)
        else:
            model.fit(X_train, y_train)
        train_time = time.perf_counter() - start_time

        y_prob = model.predict_proba(X_val)
        y_pred = (y_prob >= 0.5).astype(np.int32)

        metrics_acc.append(binary_accuracy(y_val, y_pred))
        metrics_f1.append(f1_positive(y_val, y_pred))
        metrics_auc.append(roc_auc_score_scratch(y_val, y_prob))
        metrics_brier.append(brier_score(y_val, y_prob))
        train_times.append(train_time)

    return {
        "accuracy_mean": float(np.mean(metrics_acc)),
        "accuracy_std": float(np.std(metrics_acc)),
        "f1_mean": float(np.mean(metrics_f1)),
        "f1_std": float(np.std(metrics_f1)),
        "auc_mean": float(np.mean(metrics_auc)),
        "auc_std": float(np.std(metrics_auc)),
        "brier_mean": float(np.mean(metrics_brier)),
        "brier_std": float(np.std(metrics_brier)),
        "train_time_mean_sec": float(np.mean(train_times)),
        "train_time_std_sec": float(np.std(train_times)),
        "n_folds": len(splits),
    }

def train_full_and_predict_test(model_name: str,
                                X_train: np.ndarray, y_train: np.ndarray,
                                X_test: np.ndarray,
                                scale_features: bool = True,
                                seed: int = 42) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Fit model on full training data and predict probabilities and labels on test data.
    Returns: y_prob_test, y_pred_test, train_time_sec
    """
    scaler = StandardScaler() if scale_features else None
    if scaler is not None:
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    model = create_model(model_name, input_dim=X_train.shape[1], seed=seed)

    start_time = time.perf_counter()
    if model_name in ("logistic", "svm", "nn"):
        model.fit(X_train, y_train)  # full training (no validation)
    else:
        model.fit(X_train, y_train)
    train_time = time.perf_counter() - start_time

    y_prob_test = model.predict_proba(X_test)
    y_pred_test = (y_prob_test >= 0.5).astype(np.int32)
    return y_prob_test, y_pred_test, train_time

# ----------------------------
# Orchestration across datasets x models
# ----------------------------

def run_all(dataset_pairs: Dict[str, Dict[str, str]],
            models: List[str] = ("logistic", "svm", "knn", "nn"),
            n_splits: int = 5,
            seed: int = 42,
            scale_features: bool = True,
            results_csv_path: Optional[str] = "outputs/results.csv",
            preds_dir: Optional[str] = "outputs/preds",
            kaggle_results_dir: Optional[str] = "results",
            id_fallback_dataset_key: Optional[str] = "raw_clean"):
    """
    For each dataset version, load train/test, run CV on train, then train on full train and predict test.
    Save:
      - CV metrics (CSV)
      - test predictions with probabilities (preds_dir)
      - Kaggle-ready boolean predictions (PassengerId, Transported) to kaggle_results_dir
    If test lacks PassengerId, use IDs from the fallback dataset's test (same row order).
    """
    # Prepare output dirs
    if results_csv_path:
        results_dir = os.path.dirname(results_csv_path)
        if results_dir:
            os.makedirs(results_dir, exist_ok=True)
    if preds_dir:
        os.makedirs(preds_dir, exist_ok=True)
    if kaggle_results_dir:
        os.makedirs(kaggle_results_dir, exist_ok=True)

    # Prepare fallback PassengerId list if needed
    fallback_ids = None
    if id_fallback_dataset_key and id_fallback_dataset_key in dataset_pairs:
        fallback_test_path = dataset_pairs[id_fallback_dataset_key].get("test")
        if fallback_test_path:
            _, fallback_ids, fallback_id_col = load_test_csv(fallback_test_path)
            if fallback_ids is None:
                print(f"Warning: Fallback dataset '{id_fallback_dataset_key}' test has no PassengerId column.")
        else:
            print(f"Warning: Fallback dataset '{id_fallback_dataset_key}' has no test path.")

    rows = []
    for ds_name, paths in dataset_pairs.items():
        train_path = paths.get("train")
        test_path = paths.get("test")
        if not train_path or not test_path:
            raise ValueError(f"Dataset '{ds_name}' must have 'train' and 'test' paths.")

        print(f"\n=== Dataset: {ds_name} ===")
        print(f"Train: {train_path}")
        print(f"Test:  {test_path}")

        X_train, y_train = load_train_csv(train_path)
        X_test, test_ids, id_col_name = load_test_csv(test_path)

        print(f"Loaded train shape: {X_train.shape}, positives: {np.sum(y_train == 1)}, negatives: {np.sum(y_train == 0)}")
        print(f"Loaded test shape:  {X_test.shape}, ID column: {id_col_name if id_col_name else 'None'}")

        # If no test IDs, try fallback
        if test_ids is None and fallback_ids is not None:
            if len(fallback_ids) != X_test.shape[0]:
                print(f"Warning: Fallback IDs length ({len(fallback_ids)}) != test rows ({X_test.shape[0]}) for '{ds_name}'. Using row index instead.")
            else:
                test_ids = fallback_ids.copy()
                id_col_name = "PassengerId"

        for model_name in models:
            print(f"\n -> Model: {model_name} (CV {n_splits}-fold)")
            metrics = evaluate_model_cv(model_name, X_train, y_train,
                                        n_splits=n_splits, seed=seed,
                                        scale_features=scale_features)
            print(f"accuracy: {metrics['accuracy_mean']:.4f} ± {metrics['accuracy_std']:.4f}")
            print(f"F1(pos): {metrics['f1_mean']:.4f} ± {metrics['f1_std']:.4f}")
            print(f"AUC:      {metrics['auc_mean']:.4f} ± {metrics['auc_std']:.4f}")
            print(f"Brier:    {metrics['brier_mean']:.4f} ± {metrics['brier_std']:.4f}")

            # Full train, test prediction
            print("Training on full train and predicting test...")
            y_prob_test, y_pred_test, train_time_full = train_full_and_predict_test(
                model_name, X_train, y_train, X_test, scale_features=scale_features, seed=seed
            )
            print(f"Full-train time (s): {train_time_full:.3f}")

            # Save detailed predictions (with prob) to preds_dir
            if preds_dir:
                if test_ids is None:
                    pred_df = pd.DataFrame({
                        "row": np.arange(len(y_pred_test)),
                        "Transported": y_pred_test.astype(int),
                        "prob": y_prob_test.astype(np.float32),
                    })
                else:
                    pred_df = pd.DataFrame({
                        id_col_name if id_col_name else "PassengerId": test_ids,
                        "Transported": y_pred_test.astype(int),
                        "prob": y_prob_test.astype(np.float32),
                    })
                pred_path = os.path.join(preds_dir, f"{ds_name}_{model_name}_preds.csv")
                pred_df.to_csv(pred_path, index=False)
                print(f"Saved predictions: {pred_path}")

            # Save Kaggle-ready boolean predictions (PassengerId, Transported) to results/
            # If we still don't have IDs, create a synthetic PassengerId from row index, but prefer real IDs.
            if kaggle_results_dir:
                if test_ids is None:
                    passenger_ids_out = [str(i) for i in range(len(y_pred_test))]
                else:
                    passenger_ids_out = test_ids.astype(str)

                transported_bool = y_pred_test.astype(bool)  # 0->False, 1->True
                kaggle_df = pd.DataFrame({
                    "PassengerId": passenger_ids_out,
                    "Transported": transported_bool
                })
                kaggle_path = os.path.join(kaggle_results_dir, f"{ds_name}_{model_name}.csv")
                kaggle_df.to_csv(kaggle_path, index=False)
                print(f"Saved Kaggle-ready predictions: {kaggle_path}")

            row = {"dataset": ds_name, "model": model_name}
            row.update(metrics)
            rows.append(row)

    # Save CV results summary
    if results_csv_path:
        out_df = pd.DataFrame(rows)
        out_df.to_csv(results_csv_path, index=False)
        print(f"\nSaved CV results to {results_csv_path}")

# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":
    # Update these paths to your actual files
    dataset_pairs = {
        "raw_clean": {
            "train": "data/clean/train_clean.csv",
            "test":  "data/clean/test_clean.csv",
        },
        "pca8": {
            "train": "data/pca_8_pcs/pca_transformed_8components_train.csv",
            "test":  "data/pca_8_pcs/pca_transformed_8components_test.csv",
        },
        "sigmoid_kpca": {
            "train": "data/kernel-sigmoid/train_sigmoid.csv",
            "test":  "data/kernel-sigmoid/test_sigmoid.csv",
        },
        "ae16": {
            "train": "data/ae/train_ae_k16.csv",
            "test":  "data/ae/test_ae_k16.csv",
        },
    }

    run_all(dataset_pairs=dataset_pairs,
            models=["logistic", "svm", "knn", "nn"],
            n_splits=5,
            seed=42,
            scale_features=True,
            results_csv_path="outputs/results.csv",
            preds_dir="outputs/preds",
            kaggle_results_dir="results",            # <- new: 16 CSVs here (PassengerId, Transported)
            id_fallback_dataset_key="raw_clean")     # <- use raw_clean test IDs for datasets lacking IDs