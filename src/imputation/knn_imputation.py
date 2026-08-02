"""K-nearest-neighbour imputation with data-driven k diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer


@dataclass(frozen=True)
class KScore:
    """Validation score for one candidate number of neighbours."""

    k: int
    rmse: float
    nrmse: float


class KNNImputation:
    """Select and apply a KNN imputer for numerical tabular data.

    ``evaluate_k`` hides a reproducible sample of observed cells, imputes them,
    and records RMSE/NRMSE. ``plot_k_scores`` visualises the result so that k can
    be selected before fitting the final imputer.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        *,
        weights: str = "uniform",
        metric: str = "nan_euclidean",
        random_state: int = 42,
    ) -> None:
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be at least 1")
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.random_state = random_state

    @staticmethod
    def _frame(data: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        frame = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        bad = frame.select_dtypes(exclude=np.number).columns.tolist()
        if bad:
            raise TypeError(f"KNNImputation requires numeric columns; encode first: {bad}")
        if frame.empty:
            raise ValueError("data must not be empty")
        return frame.astype(float)

    def evaluate_k(
        self,
        data: pd.DataFrame | np.ndarray,
        k_values: Iterable[int] = range(1, 16),
        *,
        validation_fraction: float = 0.1,
    ) -> pd.DataFrame:
        """Score candidate k values by masking observed cells.

        NRMSE is calculated feature-wise using each feature's observed range and
        then averaged, preventing large-scale columns from dominating selection.
        """
        if not 0 < validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1")
        frame = self._frame(data)
        observed = np.argwhere(frame.notna().to_numpy())
        if len(observed) < 2:
            raise ValueError("at least two observed values are required")

        rng = np.random.default_rng(self.random_state)
        n_mask = max(1, round(len(observed) * validation_fraction))
        held_out = observed[rng.choice(len(observed), size=n_mask, replace=False)]
        masked = frame.copy()
        truth = np.array([frame.iat[row, col] for row, col in held_out])
        for row, col in held_out:
            masked.iat[row, col] = np.nan

        ranges = frame.max(skipna=True) - frame.min(skipna=True)
        scores: list[KScore] = []
        for k in sorted(set(int(value) for value in k_values)):
            if k < 1:
                raise ValueError("all k values must be at least 1")
            imputed = KNNImputer(
                n_neighbors=k, weights=self.weights, metric=self.metric
            ).fit_transform(masked)
            predicted = np.array([imputed[row, col] for row, col in held_out])
            errors = predicted - truth
            rmse = float(np.sqrt(np.mean(errors**2)))
            per_feature: list[float] = []
            for col in np.unique(held_out[:, 1]):
                use = held_out[:, 1] == col
                scale = float(ranges.iloc[col])
                if scale > 0:
                    per_feature.append(float(np.sqrt(np.mean(errors[use] ** 2)) / scale))
            scores.append(KScore(k, rmse, float(np.mean(per_feature)) if per_feature else 0.0))

        self.k_scores_ = pd.DataFrame([score.__dict__ for score in scores])
        self.best_k_ = int(self.k_scores_.loc[self.k_scores_["nrmse"].idxmin(), "k"])
        return self.k_scores_.copy()

    def plot_k_scores(self, *, metric: str = "nrmse", ax=None, show: bool = False):
        """Plot candidate scores and highlight the best k; returns the axes."""
        if not hasattr(self, "k_scores_"):
            raise RuntimeError("call evaluate_k before plot_k_scores")
        if metric not in {"rmse", "nrmse"}:
            raise ValueError("metric must be 'rmse' or 'nrmse'")
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(self.k_scores_["k"], self.k_scores_[metric], marker="o")
        best = self.k_scores_.loc[self.k_scores_[metric].idxmin()]
        ax.scatter([best["k"]], [best[metric]], color="crimson", zorder=3,
                   label=f"best k = {int(best['k'])}")
        ax.set(xlabel="Number of neighbours (k)", ylabel=metric.upper(),
               title="KNN imputation validation score")
        ax.grid(alpha=0.25)
        ax.legend()
        if show:
            plt.show()
        return ax

    def fit(self, data: pd.DataFrame | np.ndarray, *, use_best_k: bool = False):
        frame = self._frame(data)
        k = self.best_k_ if use_best_k and hasattr(self, "best_k_") else self.n_neighbors
        self.imputer_ = KNNImputer(n_neighbors=k, weights=self.weights, metric=self.metric)
        self.imputer_.fit(frame)
        self.columns_ = frame.columns
        self.n_features_in_ = frame.shape[1]
        self.selected_k_ = k
        return self

    def transform(self, data: pd.DataFrame | np.ndarray):
        if not hasattr(self, "imputer_"):
            raise RuntimeError("fit must be called before transform")
        frame = self._frame(data)
        values = self.imputer_.transform(frame)
        if isinstance(data, pd.DataFrame):
            return pd.DataFrame(values, index=data.index, columns=self.columns_)
        return values

    def fit_transform(self, data: pd.DataFrame | np.ndarray, *, use_best_k: bool = False):
        return self.fit(data, use_best_k=use_best_k).transform(data)
