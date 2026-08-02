"""Scalable mixed-type KNN imputation with k-selection diagnostics."""

from __future__ import annotations

from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler


class KNNImputation:
    """Impute numeric columns with KNN and categorical columns explicitly.

    KNN is fitted on a reproducible sample to keep large datasets tractable.
    Numeric values are standardised before distances are calculated. Categorical
    missing values are represented by ``categorical_fill_value`` because an
    absent category can itself be predictive in fraud data.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        *,
        categorical_columns: Sequence[str] | None = None,
        exclude_columns: Sequence[str] = ("isFraud",),
        categorical_fill_value: str = "__MISSING__",
        add_missing_indicators: bool = True,
        max_fit_rows: int = 30_000,
        transform_batch_size: int = 25_000,
        weights: str = "distance",
        random_state: int = 42,
    ) -> None:
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be at least 1")
        if max_fit_rows < n_neighbors:
            raise ValueError("max_fit_rows must be at least n_neighbors")
        self.n_neighbors = n_neighbors
        self.categorical_columns = list(categorical_columns or [])
        self.exclude_columns = list(exclude_columns)
        self.categorical_fill_value = categorical_fill_value
        self.add_missing_indicators = add_missing_indicators
        self.max_fit_rows = max_fit_rows
        self.transform_batch_size = transform_batch_size
        self.weights = weights
        self.random_state = random_state

    @staticmethod
    def _frame(data: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(data, pd.DataFrame) or data.empty:
            raise TypeError("data must be a non-empty pandas DataFrame")
        return data.copy()

    def _split_columns(self, frame: pd.DataFrame) -> None:
        excluded = [column for column in self.exclude_columns if column in frame]
        explicit = [column for column in self.categorical_columns if column in frame]
        inferred = frame.select_dtypes(exclude=np.number).columns.tolist()
        self.excluded_columns_ = excluded
        self.categorical_columns_ = list(dict.fromkeys(explicit + inferred))
        self.categorical_columns_ = [c for c in self.categorical_columns_ if c not in excluded]
        self.numeric_columns_ = [
            c for c in frame.select_dtypes(include=np.number).columns
            if c not in excluded and c not in self.categorical_columns_
        ]
        if not self.numeric_columns_:
            raise ValueError("no numeric feature columns are available for KNN")

    def _sample(self, frame: pd.DataFrame, maximum: int) -> pd.DataFrame:
        if len(frame) <= maximum:
            return frame
        return frame.sample(maximum, random_state=self.random_state)

    def evaluate_k(
        self,
        data: pd.DataFrame,
        k_values: Iterable[int] = range(1, 16),
        *,
        validation_fraction: float = 0.05,
        max_eval_rows: int = 10_000,
    ) -> pd.DataFrame:
        """Evaluate k on a training-only sample and store RMSE/NRMSE scores."""
        if not 0 < validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1")
        frame = self._frame(data)
        self._split_columns(frame)
        sample = self._sample(frame[self.numeric_columns_], max_eval_rows)
        usable = sample.columns[sample.notna().any()].tolist()
        sample = sample[usable].astype(float)

        rng = np.random.default_rng(self.random_state)
        observed = np.argwhere(sample.notna().to_numpy())
        n_mask = max(1, round(len(observed) * validation_fraction))
        held_out = observed[rng.choice(len(observed), n_mask, replace=False)]
        masked = sample.copy()
        truth = np.array([sample.iat[r, c] for r, c in held_out])
        for row, column in held_out:
            masked.iat[row, column] = np.nan

        scaler = StandardScaler().fit(masked)
        scaled = scaler.transform(masked)
        ranges = sample.max() - sample.min()
        records: list[dict[str, float | int]] = []
        for k in sorted(set(map(int, k_values))):
            if k < 1:
                raise ValueError("all k values must be at least 1")
            predicted_scaled = KNNImputer(n_neighbors=k, weights=self.weights).fit_transform(scaled)
            predicted = scaler.inverse_transform(predicted_scaled)
            estimates = np.array([predicted[r, c] for r, c in held_out])
            errors = estimates - truth
            feature_scores = []
            for column in np.unique(held_out[:, 1]):
                use = held_out[:, 1] == column
                scale = float(ranges.iloc[column])
                if scale > 0:
                    feature_scores.append(np.sqrt(np.mean(errors[use] ** 2)) / scale)
            records.append({
                "k": k,
                "rmse": float(np.sqrt(np.mean(errors**2))),
                "nrmse": float(np.mean(feature_scores)) if feature_scores else 0.0,
            })
        self.k_scores_ = pd.DataFrame(records)
        self.best_k_ = int(self.k_scores_.loc[self.k_scores_["nrmse"].idxmin(), "k"])
        return self.k_scores_.copy()

    def plot_k_scores(self, *, metric: str = "nrmse", ax=None, show: bool = False):
        if not hasattr(self, "k_scores_"):
            raise RuntimeError("call evaluate_k before plot_k_scores")
        if metric not in {"rmse", "nrmse"}:
            raise ValueError("metric must be 'rmse' or 'nrmse'")
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(self.k_scores_["k"], self.k_scores_[metric], marker="o")
        best = self.k_scores_.loc[self.k_scores_[metric].idxmin()]
        ax.scatter(best["k"], best[metric], color="crimson", zorder=3,
                   label=f"best k = {int(best['k'])}")
        ax.set(xlabel="Number of neighbours (k)", ylabel=metric.upper(),
               title="Sampled KNN imputation validation")
        ax.grid(alpha=0.25)
        ax.legend()
        if show:
            plt.show()
        return ax

    def fit(self, data: pd.DataFrame, *, use_best_k: bool = False):
        frame = self._frame(data)
        self._split_columns(frame)
        self.columns_ = frame.columns.copy()
        self.missing_columns_ = [c for c in frame.columns if frame[c].isna().any()]
        numeric = self._sample(frame[self.numeric_columns_], self.max_fit_rows).astype(float)
        all_missing = numeric.columns[numeric.isna().all()].tolist()
        if all_missing:
            raise ValueError(f"training sample has entirely missing columns: {all_missing}")
        self.scaler_ = StandardScaler().fit(numeric)
        selected_k = self.best_k_ if use_best_k and hasattr(self, "best_k_") else self.n_neighbors
        self.imputer_ = KNNImputer(n_neighbors=selected_k, weights=self.weights)
        self.imputer_.fit(self.scaler_.transform(numeric))
        self.selected_k_ = selected_k
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "imputer_"):
            raise RuntimeError("fit must be called before transform")
        frame = self._frame(data)
        missing = [c for c in self.columns_ if c not in frame]
        if missing:
            raise ValueError(f"transform is missing fitted columns: {missing}")
        result = frame.copy()
        if self.add_missing_indicators:
            for column in self.missing_columns_:
                result[f"{column}_missing"] = result[column].isna().astype("int8")
        for column in self.categorical_columns_:
            result[column] = result[column].astype("string").fillna(self.categorical_fill_value)
        for start in range(0, len(result), self.transform_batch_size):
            rows = result.index[start:start + self.transform_batch_size]
            numeric = result.loc[rows, self.numeric_columns_].astype(float)
            values = self.imputer_.transform(self.scaler_.transform(numeric))
            result.loc[rows, self.numeric_columns_] = self.scaler_.inverse_transform(values)
        return result

    def fit_transform(self, data: pd.DataFrame, *, use_best_k: bool = False) -> pd.DataFrame:
        return self.fit(data, use_best_k=use_best_k).transform(data)
