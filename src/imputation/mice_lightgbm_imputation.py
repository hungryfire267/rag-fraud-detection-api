"""Scalable MICE-style imputation with tuned LightGBM regressors."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, RandomizedSearchCV


class MICELightGBMImputation:
    """Impute numeric features using chained, tuned LightGBM equations.

    Hyperparameters are tuned once per target and reused in later iterations.
    Training is sampled for scalability, categorical features receive an explicit
    missing level, and extremely sparse numeric columns use median fallback.
    The number of chained-equation iterations is capped at ten.
    """

    DEFAULT_PARAM_DISTRIBUTIONS = {
        "n_estimators": [100, 200, 300, 500],
        "learning_rate": [0.02, 0.05, 0.1],
        "num_leaves": [15, 31, 63, 127],
        "max_depth": [-1, 5, 8, 12],
        "min_child_samples": [10, 20, 50, 100],
        "subsample": [0.75, 0.9, 1.0],
        "colsample_bytree": [0.75, 0.9, 1.0],
        "reg_alpha": [0.0, 0.1, 1.0],
        "reg_lambda": [0.0, 1.0, 5.0],
    }

    def __init__(
        self,
        *,
        max_iter: int = 10,
        n_iter_search: int = 5,
        cv: int = 3,
        tol: float = 1e-3,
        categorical_columns: Sequence[str] | None = None,
        exclude_columns: Sequence[str] = ("isFraud",),
        categorical_fill_value: str = "__MISSING__",
        add_missing_indicators: bool = True,
        max_train_rows: int = 50_000,
        max_model_missing_fraction: float = 0.70,
        transform_batch_size: int = 50_000,
        random_state: int = 42,
        n_jobs: int = -1,
        param_distributions: dict[str, Any] | None = None,
    ) -> None:
        if not 1 <= max_iter <= 10:
            raise ValueError("max_iter must be between 1 and 10")
        if n_iter_search < 0:
            raise ValueError("n_iter_search cannot be negative")
        if not 0 < max_model_missing_fraction < 1:
            raise ValueError("max_model_missing_fraction must be between 0 and 1")
        self.max_iter = max_iter
        self.n_iter_search = n_iter_search
        self.cv = cv
        self.tol = tol
        self.categorical_columns = list(categorical_columns or [])
        self.exclude_columns = list(exclude_columns)
        self.categorical_fill_value = categorical_fill_value
        self.add_missing_indicators = add_missing_indicators
        self.max_train_rows = max_train_rows
        self.max_model_missing_fraction = max_model_missing_fraction
        self.transform_batch_size = transform_batch_size
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.param_distributions = param_distributions or self.DEFAULT_PARAM_DISTRIBUTIONS

    @staticmethod
    def _frame(data: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(data, pd.DataFrame) or data.empty:
            raise TypeError("data must be a non-empty pandas DataFrame")
        return data.copy()

    def _split_columns(self, frame: pd.DataFrame) -> None:
        excluded = [column for column in self.exclude_columns if column in frame]
        inferred = frame.select_dtypes(exclude=np.number).columns.tolist()
        explicit = [column for column in self.categorical_columns if column in frame]
        self.excluded_columns_ = excluded
        self.categorical_columns_ = [
            column
            for column in dict.fromkeys(explicit + inferred)
            if column not in excluded
        ]
        self.numeric_columns_ = [
            column
            for column in frame.select_dtypes(include=np.number).columns
            if column not in excluded and column not in self.categorical_columns_
        ]
        if not self.numeric_columns_:
            raise ValueError("no numeric feature columns are available")

    def _base_model(self, **params) -> LGBMRegressor:
        return LGBMRegressor(
            objective="regression",
            random_state=self.random_state,
            n_jobs=1,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
            **params,
        )

    def _tune(self, x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        folds = min(self.cv, len(y))
        if folds < 2 or self.n_iter_search == 0:
            return {}
        search = RandomizedSearchCV(
            estimator=self._base_model(),
            param_distributions=self.param_distributions,
            n_iter=self.n_iter_search,
            scoring="neg_root_mean_squared_error",
            cv=KFold(folds, shuffle=True, random_state=self.random_state),
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            refit=False,
            error_score="raise",
        )
        search.fit(x, y)
        return search.best_params_

    def fit(self, data: pd.DataFrame):
        frame = self._frame(data)
        self._split_columns(frame)
        self.columns_ = frame.columns.copy()
        self.missing_columns_ = [column for column in frame if frame[column].isna().any()]
        sample = frame.sample(
            n=min(len(frame), self.max_train_rows),
            random_state=self.random_state,
        )
        numeric = sample[self.numeric_columns_].astype(float)
        if numeric.isna().all().any():
            columns = numeric.columns[numeric.isna().all()].tolist()
            raise ValueError(f"training sample has entirely missing columns: {columns}")

        self.initial_imputer_ = SimpleImputer(strategy="median")
        filled = self.initial_imputer_.fit_transform(numeric)
        missing = numeric.isna().to_numpy()
        missing_rates = numeric.isna().mean()
        self.modelled_columns_ = [
            column
            for column in self.numeric_columns_
            if 0 < missing_rates[column] <= self.max_model_missing_fraction
        ]
        self.fallback_columns_ = [
            column
            for column in self.numeric_columns_
            if missing_rates[column] > self.max_model_missing_fraction
        ]
        self.best_params_: dict[str, dict[str, Any]] = {}
        self.models_: dict[str, LGBMRegressor] = {}
        positions = {column: index for index, column in enumerate(self.numeric_columns_)}

        # Tune once on the median-initialised training sample.
        for column in self.modelled_columns_:
            target = positions[column]
            observed = ~missing[:, target]
            predictors = np.arange(filled.shape[1]) != target
            self.best_params_[column] = self._tune(
                filled[observed][:, predictors],
                numeric.loc[observed, column].to_numpy(),
            )

        self.convergence_: list[float] = []
        for _ in range(self.max_iter):
            previous = filled.copy()
            for column in self.modelled_columns_:
                target = positions[column]
                observed = ~missing[:, target]
                predictors = np.arange(filled.shape[1]) != target
                model = self._base_model(**self.best_params_[column])
                model.fit(
                    filled[observed][:, predictors],
                    numeric.loc[observed, column],
                )
                missing_rows = missing[:, target]
                filled[missing_rows, target] = model.predict(
                    filled[missing_rows][:, predictors]
                )
                self.models_[column] = model
            changed = np.abs(filled - previous)[missing]
            delta = float(changed.max()) if changed.size else 0.0
            self.convergence_.append(delta)
            if delta <= self.tol:
                break
        self.n_iter_ = len(self.convergence_)
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "models_"):
            raise RuntimeError("fit must be called before transform")
        frame = self._frame(data)
        missing_columns = [column for column in self.columns_ if column not in frame]
        if missing_columns:
            raise ValueError(f"transform is missing fitted columns: {missing_columns}")
        result = frame.copy()
        if self.add_missing_indicators:
            for column in self.missing_columns_:
                result[f"{column}_missing"] = result[column].isna().astype("int8")
        for column in self.categorical_columns_:
            result[column] = (
                result[column]
                .astype("string")
                .fillna(self.categorical_fill_value)
            )

        positions = {column: index for index, column in enumerate(self.numeric_columns_)}
        for start in range(0, len(result), self.transform_batch_size):
            rows = result.index[start:start + self.transform_batch_size]
            numeric = result.loc[rows, self.numeric_columns_].astype(float)
            missing = numeric.isna().to_numpy()
            filled = self.initial_imputer_.transform(numeric)
            for _ in range(self.max_iter):
                previous = filled.copy()
                for column, model in self.models_.items():
                    target = positions[column]
                    missing_rows = missing[:, target]
                    if missing_rows.any():
                        predictors = np.arange(filled.shape[1]) != target
                        filled[missing_rows, target] = model.predict(
                            filled[missing_rows][:, predictors]
                        )
                changed = np.abs(filled - previous)[missing]
                if not changed.size or float(changed.max()) <= self.tol:
                    break
            result.loc[rows, self.numeric_columns_] = filled
        return result

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.fit(data).transform(data)
