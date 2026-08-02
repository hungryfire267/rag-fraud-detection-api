"""MICE-style chained-equation imputation using tuned XGBoost regressors."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, RandomizedSearchCV
from xgboost import XGBRegressor


class MICEXGBoostImputation:
    """Iteratively impute numeric features with tuned XGBoost models.

    A separate regressor is tuned for every incomplete feature. The chained
    equations are repeated at most ten times, as requested. For small targets
    that cannot support cross-validation, a default regressor is used.
    """

    DEFAULT_PARAM_DISTRIBUTIONS = {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [2, 3, 4, 6, 8],
        "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
        "min_child_weight": [1, 3, 5, 10],
        "reg_alpha": [0.0, 0.01, 0.1, 1.0],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0],
    }

    def __init__(
        self,
        *,
        max_iter: int = 10,
        n_iter_search: int = 10,
        cv: int = 3,
        tol: float = 1e-3,
        random_state: int = 42,
        n_jobs: int = -1,
        param_distributions: dict[str, Any] | None = None,
    ) -> None:
        if not 1 <= max_iter <= 10:
            raise ValueError("max_iter must be between 1 and 10")
        if n_iter_search < 1:
            raise ValueError("n_iter_search must be at least 1")
        self.max_iter = max_iter
        self.n_iter_search = n_iter_search
        self.cv = cv
        self.tol = tol
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.param_distributions = param_distributions or self.DEFAULT_PARAM_DISTRIBUTIONS

    @staticmethod
    def _frame(data: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        frame = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        bad = frame.select_dtypes(exclude=np.number).columns.tolist()
        if bad:
            raise TypeError(f"MICEXGBoostImputation requires numeric columns; encode first: {bad}")
        if frame.empty:
            raise ValueError("data must not be empty")
        if frame.isna().all().any():
            names = frame.columns[frame.isna().all()].tolist()
            raise ValueError(f"cannot impute entirely missing columns: {names}")
        return frame.astype(float)

    def _base_model(self) -> XGBRegressor:
        return XGBRegressor(
            objective="reg:squarederror",
            random_state=self.random_state,
            n_jobs=1,
            tree_method="hist",
            verbosity=0,
        )

    def _fit_target(self, x: np.ndarray, y: np.ndarray):
        folds = min(self.cv, len(y))
        if folds < 2:
            return self._base_model().fit(x, y), {}
        search = RandomizedSearchCV(
            self._base_model(),
            self.param_distributions,
            n_iter=self.n_iter_search,
            scoring="neg_root_mean_squared_error",
            cv=KFold(folds, shuffle=True, random_state=self.random_state),
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            refit=True,
            error_score="raise",
        )
        search.fit(x, y)
        return search.best_estimator_, search.best_params_

    def fit_transform(self, data: pd.DataFrame | np.ndarray):
        frame = self._frame(data)
        missing = frame.isna().to_numpy()
        self.initial_imputer_ = SimpleImputer(strategy="median")
        filled = self.initial_imputer_.fit_transform(frame)
        incomplete = np.flatnonzero(missing.any(axis=0))
        self.models_: dict[int, XGBRegressor] = {}
        self.best_params_: dict[Any, dict[str, Any]] = {}
        self.convergence_: list[float] = []

        for _ in range(self.max_iter):
            previous = filled.copy()
            for target in incomplete:
                observed = ~missing[:, target]
                predictors = np.arange(filled.shape[1]) != target
                model, params = self._fit_target(filled[observed][:, predictors], frame.iloc[observed, target].to_numpy())
                if missing[:, target].any():
                    filled[missing[:, target], target] = model.predict(filled[missing[:, target]][:, predictors])
                self.models_[target] = model
                self.best_params_[frame.columns[target]] = params
            changed = np.abs(filled - previous)[missing]
            delta = float(changed.max()) if changed.size else 0.0
            self.convergence_.append(delta)
            if delta <= self.tol:
                break

        self.n_iter_ = len(self.convergence_)
        self.columns_ = frame.columns
        self.n_features_in_ = frame.shape[1]
        if isinstance(data, pd.DataFrame):
            return pd.DataFrame(filled, index=data.index, columns=frame.columns)
        return filled

    def fit(self, data: pd.DataFrame | np.ndarray):
        self.fit_transform(data)
        return self

    def transform(self, data: pd.DataFrame | np.ndarray):
        if not hasattr(self, "models_"):
            raise RuntimeError("fit must be called before transform")
        frame = self._frame(data)
        missing = frame.isna().to_numpy()
        filled = self.initial_imputer_.transform(frame)
        for _ in range(self.max_iter):
            previous = filled.copy()
            for target, model in self.models_.items():
                rows = missing[:, target]
                if rows.any():
                    predictors = np.arange(filled.shape[1]) != target
                    filled[rows, target] = model.predict(filled[rows][:, predictors])
            changed = np.abs(filled - previous)[missing]
            if not changed.size or float(changed.max()) <= self.tol:
                break
        if isinstance(data, pd.DataFrame):
            return pd.DataFrame(filled, index=data.index, columns=self.columns_)
        return filled
