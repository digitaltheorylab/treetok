"""XGBoost merge classifier wrapper."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xgboost as xgb

from .features import FEATURE_SPEC


def _f1_binary(y_true, y_pred):
    """Compute binary F1 score (with zero_division=0 behavior).

    Parameters
    ----------
    y_true : np.ndarray
        True labels (0/1 or bool)
    y_pred : np.ndarray
        Predicted labels (0/1 or bool)

    Returns
    -------
    float
        F1 score
    """
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)

    tp = np.sum(y_true & y_pred)
    fp = np.sum(~y_true & y_pred)
    fn = np.sum(y_true & ~y_pred)

    if tp == 0:
        return 0.0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return float(2 * precision * recall / (precision + recall))


def _pick_pr_idx(precision, recall, target_precision):
    """Pick a PR-curve index subject to a precision constraint.

    Selects the index with the highest recall among those with `precision` >=
    `target_precision`. If no index satisfies the constraint, falls back to the
    index with the best F1

    Parameters
    ----------
    precision : np.ndarray
        Precision values
    recall : np.ndarray
        Recall values
    target_precision : float
        Minimum acceptable precision

    Returns
    -------
    int
        Selected index
    """
    n = len(precision)
    if n == 0:
        return 0

    best_recall = np.where(precision >= target_precision, recall, -np.inf)
    idx = np.argmax(best_recall)
    if np.isfinite(best_recall[idx]):
        return int(idx)

    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)

    return int(np.argmax(f1))


def _precision_recall_curve(y_true, y_score):
    """Compute a precision-recall curve.

    Parameters
    ----------
    y_true : np.ndarray
        True labels (0/1 or bool)
    y_score : np.ndarray
        Continuous label scores

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Precision, recall, and thresholds, where thresholds are sorted in
        descending order
    """
    y_true = np.asarray(y_true, dtype=np.int8)
    y_score = np.asarray(y_score, dtype=np.float64)

    order = np.argsort(-y_score, kind="stable")
    y = y_true[order]
    s = y_score[order]

    # Cumulative TP/FP at each prefix; threshold is the score at a prefix
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    n_pos = np.sum(y_true == 1)

    precision = tp / np.clip(tp + fp, 1, None)
    recall = tp / max(n_pos, 1)

    # Append a trailing sentinel point
    precision = np.concatenate([precision, [1.0]])
    recall = np.concatenate([recall, [0.0]])

    return precision, recall, s


def _stratified_split(X, y, val_size, random_state):
    """Perform a stratified train/validation split.

    Parameters
    ----------
    X : np.ndarray
        Dataset (2D)
    y : np.ndarray
        Labels (1D)
    val_size : float
        Fraction of samples to put in validation (0 < val_size < 1)
    random_state : int
        Seed for shuffling

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        X train, X val, y train, y val

    Raises
    ------
    ValueError
        If `val_size` isn't in (0, 1)
    """
    if not (0.0 < val_size < 1.0):
        raise ValueError("val_size must be in (0, 1)")

    rng = np.random.default_rng(random_state)

    train_parts = []
    val_parts = []

    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        idx = rng.permutation(idx)

        n_val = int(len(idx) * val_size)
        val_parts.append(idx[:n_val])
        train_parts.append(idx[n_val:])

    train_idx = np.concatenate(train_parts) if train_parts else np.array([])
    val_idx = np.concatenate(val_parts) if val_parts else np.array([])

    train_idx = rng.permutation(train_idx)
    val_idx = rng.permutation(val_idx)

    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]


@dataclass
class TrainReport:
    """Diagnostics produced by `MergeClassifier.fit`."""

    edge_threshold: float
    merge_threshold: float
    train_f1: float
    val_f1: float
    val_edge_precision: float
    val_edge_recall: float
    val_merge_precision: float
    val_merge_recall: float
    n_train: int
    n_val: int


class MergeClassifier:
    """XGBoost-backed binary merge classifier with tuned operating thresholds.

    The persisted artifact bundles the booster, feature spec version, feature
    names, and two operating thresholds tuned on a held-out validation split:

    - `edge_threshold`: cutoff above which a pair is considered a positive
      merge candidate
    - `merge_threshold`: stricter cutoff for bridging two non-singleton
      components. Always >= `edge_threshold`
    """

    def __init__(
        self,
        edge_threshold: float = 0.5,
        merge_threshold: float | None = None,
        booster: xgb.Booster | None = None,
        feature_spec_version: int | None = None,
        feature_names: list[str] | None = None,
        params: dict | None = None,
    ):
        """Initialize the classifier.

        Parameters
        ----------
        edge_threshold : float
            Cutoff above which a pair is considered a positive merge candidate
        merge_threshold : float or None
            Stricter cutoff for bridging two non-singleton components
        booster : xgb.Booster or None
            Booster
        feature_spec_version : int or None
            Version number for features
        feature_names : list[str] or None
            Feature names
        params : dict or None
            Parameters for the booster
        """
        self.feature_spec_version = (
            feature_spec_version
            if feature_spec_version is not None
            else FEATURE_SPEC["version"]
        )

        self.feature_names = (
            list(feature_names)
            if feature_names is not None
            else FEATURE_SPEC["names"]
        )

        self.params = params if params is not None else self._default_params()

        # Initialization defaults; learned values are stored with `_`
        self.edge_threshold = edge_threshold
        self.merge_threshold = (
            merge_threshold
            if merge_threshold is not None
            else max(self.edge_threshold, 0.85)
        )

        # Learned values
        self.booster_ = booster
        self.edge_threshold_ = None
        self.merge_threshold_ = None
        self.report_ = None

        # If a booster is injected, assume provided thresholds should be used
        if booster is not None:
            self.edge_threshold_ = self.edge_threshold
            self.merge_threshold_ = max(
                self.merge_threshold, self.edge_threshold
            )

    @staticmethod
    def _default_params():
        """Default XGBoost training parameters.

        Returns
        -------
        dict
            XGBoost parameters
        """
        return {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 6,
            "eta": 0.1,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "tree_method": "hist",
        }

    def _ensure_fitted(self):
        """Raise if the classifier has not been fitted or loaded."""
        if (
            self.booster_ is None
            or self.edge_threshold_ is None
            or self.merge_threshold_ is None
        ):
            raise RuntimeError(
                "MergeClassifier is not fitted. Call .fit() or .load() first."
            )

    def _validate_X_y(self, X, y) -> tuple[np.ndarray, np.ndarray]:
        """Validate the dataset and training labels and normalize labels.

        Parameters
        ----------
        X : np.ndarray
            Dataset (2D)
        y : np.ndarray
            Labels (1D)

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Dataset and labels normalized to {0,1}
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D; got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D; got shape {y.shape}")
        if len(X) != len(y):
            raise ValueError(
                f"X and y must have the same length; got {len(X)} and {len(y)}"
            )
        if X.shape[1] != len(self.feature_names):
            raise ValueError(
                f"X has shape {X.shape[1]} feature columns; expected "
                f"{len(self.feature_names)} for FEATURE_SPEC "
                f"v{self.feature_spec_version}"
            )

        # Normalize labels to {0,1}
        y = y.astype(np.int8, copy=False)
        uniq = np.unique(y)
        if not np.all(np.isin(uniq, (0, 1))):
            raise ValueError(f"y must be binary (0/1 or bool). Got {uniq!r}")

        return X, y

    def _make_train_report(
        self,
        *,
        split,
        preds,
        pr,
        idx: tuple[int, int],
    ) -> TrainReport:
        """Make a TrainReport.

        Parameters
        ----------
        split : tuple[np.ndarray, np.ndarray]
            Training/validation labels
        preds : tuple[np.ndarray, np.ndarray]
            Training/validation predicted labels
        pr : tuple[np.ndarray, np.ndarray]
            Precision/recall values
        idx : tuple[int, int]
            Edge/merge indices

        Returns
        -------
        TrainReport
            Report about the trained classifier
        """
        ytr, yv = split
        train_pred, val_pred = preds
        precision, recall = pr
        edge_idx, merge_idx = idx

        val_edge_precision = precision[edge_idx] if len(precision) else 0.0
        val_edge_recall = recall[edge_idx] if len(recall) else 0.0
        val_merge_precision = precision[merge_idx] if len(precision) else 0.0
        val_merge_recall = recall[merge_idx] if len(recall) else 0.0

        return TrainReport(
            edge_threshold=self.edge_threshold_,
            merge_threshold=self.merge_threshold_,
            train_f1=_f1_binary(ytr, train_pred),
            val_f1=_f1_binary(yv, val_pred),
            val_edge_precision=val_edge_precision,
            val_edge_recall=val_edge_recall,
            val_merge_precision=val_merge_precision,
            val_merge_recall=val_merge_recall,
            n_train=len(ytr),
            n_val=len(yv),
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        num_boost_round: int = 400,
        early_stopping_rounds: int = 30,
        val_size: float = 0.2,
        random_state: int = 0,
        target_precision: float = 0.99,
        threshold_floor: float = 0.5,
        merge_target_precision: float = 0.999,
        merge_threshold_floor: float = 0.85,
    ) -> "MergeClassifier":
        """Train the classifier and tune both operating thresholds.

        Splits stratified by `y`, fits on train, then chooses two cutoffs.
        Threshold selection chooses the index with the highest recall among
        those meeting the precision constraint. If none meet it, falls back to
        the best F1

        Raises
        ------
        ValueError
            If `val_size` is not in (0, 1)
        """
        if not (0.0 < val_size < 1.0):
            raise ValueError("val_size must be in (0, 1)")

        X, y = self._validate_X_y(X, y)
        Xtr, Xv, ytr, yv = _stratified_split(X, y, val_size, random_state)

        dtr = xgb.DMatrix(Xtr, label=ytr, feature_names=self.feature_names)
        dv = xgb.DMatrix(Xv, label=yv, feature_names=self.feature_names)

        self.booster_ = xgb.train(
            self.params,
            dtr,
            num_boost_round=num_boost_round,
            evals=[(dv, "val")],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
        )

        pv = self.booster_.predict(dv)
        precisions, recalls, thresholds = _precision_recall_curve(yv, pv)

        # Drop sentinel
        precision = precisions[:-1]
        recall = recalls[:-1]

        edge_idx = _pick_pr_idx(precision, recall, target_precision)
        merge_idx = _pick_pr_idx(precision, recall, merge_target_precision)

        if len(thresholds):
            edge_val = thresholds[edge_idx]
            merge_val = thresholds[merge_idx]
        else:
            edge_val = 0.5
            merge_val = 0.85

        self.edge_threshold_ = max(edge_val, threshold_floor)
        self.merge_threshold_ = max(
            merge_val, merge_threshold_floor, self.edge_threshold_
        )

        ptr = self.booster_.predict(dtr)
        train_pred = (ptr >= self.edge_threshold_).astype(np.int8)
        val_pred = (pv >= self.edge_threshold_).astype(np.int8)

        self.report_ = self._make_train_report(
            split=(ytr, yv),
            preds=(train_pred, val_pred),
            pr=(precision, recall),
            idx=(edge_idx, merge_idx),
        )

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict raw probabilities for merging.

        Parameters
        ----------
        X : np.ndarray
            Feature data

        Returns
        -------
        np.ndarray
            P(merge) for each row of X
        """
        self._ensure_fitted()

        X = np.asarray(X, dtype=np.float32)
        d = xgb.DMatrix(X, feature_names=self.feature_names)

        return self.booster_.predict(d)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict merge labels at the tuned edge threshold.

        Parameters
        ----------
        X : np.ndarray
            Feature data

        Returns
        -------
        np.ndarray
            Binary merge decisions
        """
        return (self.predict_proba(X) >= self.edge_threshold_).astype(np.int8)

    def save(self, path: str | Path) -> None:
        """Persist booster + thresholds + metadata to a single JSON file.

        Parameters
        ----------
        path : str or Path
            Path to saved model
        """
        self._ensure_fitted()
        path = Path(path)

        booster_json = json.loads(
            self.booster_.save_raw(raw_format="json").decode("utf-8")
        )
        payload = {
            "feature_spec_version": self.feature_spec_version,
            "feature_names": self.feature_names,
            "edge_threshold": self.edge_threshold_,
            "merge_threshold": self.merge_threshold_,
            "params": self.params,
            "booster": booster_json,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "MergeClassifier":
        """Load a classifier.

        Parameters
        ----------
        path : str or Path
            Path to saved model

        Returns
        -------
        MergeClassifier
            The trained classifier

        Raises
        ------
        ValueError
            If model feature-spec doesn't match current one
        """
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))

        spec_v = int(payload["feature_spec_version"])
        if spec_v != int(FEATURE_SPEC["version"]):
            raise ValueError(
                f"model was trained against FEATURE_SPEC v{spec_v}, current is "
                f"v{FEATURE_SPEC['version']}. Retrain or pin features."
            )

        booster = xgb.Booster()
        booster.load_model(
            bytearray(json.dumps(payload["booster"]).encode("utf-8"))
        )

        return cls(
            edge_threshold=float(payload["edge_threshold"]),
            merge_threshold=float(payload["merge_threshold"]),
            booster=booster,
            feature_spec_version=spec_v,
            feature_names=list(payload["feature_names"]),
            params=dict(payload.get("params") or {}),
        )
