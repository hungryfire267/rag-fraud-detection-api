"""Guard-railed, model-selectable LLM imputation for small experiments."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    model: str
    base_url: str | None = None


class LLMImputation:
    """Impute small tabular samples using a selectable chat model.

    The class deliberately refuses large inputs by default: the fraud dataset
    would otherwise require over 100,000 API calls. Fit statistics provide a
    deterministic fallback when a response or API call fails validation.
    """

    PAPER_MODELS = (
        ModelSpec("openrouter", "xiaomi/mimo-v2-flash", "https://openrouter.ai/api/v1"),
        ModelSpec("openrouter", "mistralai/devstral-2512", "https://openrouter.ai/api/v1"),
        ModelSpec("google", "gemini-3-flash"),
        ModelSpec("openrouter", "openai/gpt-4.1-nano", "https://openrouter.ai/api/v1"),
        ModelSpec("anthropic", "claude-sonnet-4.5"),
    )

    def __init__(
        self,
        model: str,
        *,
        client=None,
        completion_fn: Callable[[str, str, float], str] | None = None,
        exclude_columns: Sequence[str] = ("isFraud",),
        temperature: float = 0.1,
        max_rows_per_request: int = 40,
        max_columns_per_request: int = 10,
        max_total_rows: int = 1_000,
        allow_large_dataset: bool = False,
        max_retries: int = 3,
        retry_base_seconds: float = 1.0,
        dataset_name: str = "IEEE-CIS Fraud Detection",
    ) -> None:
        if client is None and completion_fn is None:
            raise ValueError("provide an OpenAI-compatible client or completion_fn")
        self.model = model
        self.client = client
        self.completion_fn = completion_fn
        self.exclude_columns = list(exclude_columns)
        self.temperature = temperature
        self.max_rows_per_request = max_rows_per_request
        self.max_columns_per_request = max_columns_per_request
        self.max_total_rows = max_total_rows
        self.allow_large_dataset = allow_large_dataset
        self.max_retries = max_retries
        self.retry_base_seconds = retry_base_seconds
        self.dataset_name = dataset_name

    @classmethod
    def paper_models(cls) -> list[dict[str, str | None]]:
        return [spec.__dict__.copy() for spec in cls.PAPER_MODELS]

    def available_models(self) -> list[str]:
        if self.client is None or not hasattr(self.client, "models"):
            return [spec.model for spec in self.PAPER_MODELS]
        return sorted(item.id for item in self.client.models.list().data)

    def set_model(self, model: str, *, validate: bool = False):
        if validate and model not in self.available_models():
            raise ValueError(f"model is not available: {model}")
        self.model = model
        return self

    def fit(self, data: pd.DataFrame):
        if not isinstance(data, pd.DataFrame) or data.empty:
            raise TypeError("data must be a non-empty pandas DataFrame")
        self.columns_ = data.columns.copy()
        self.feature_columns_ = [c for c in data if c not in self.exclude_columns]
        self.fallback_values_: dict[str, object] = {}
        for column in self.feature_columns_:
            values = data[column].dropna()
            if pd.api.types.is_numeric_dtype(data[column]):
                self.fallback_values_[column] = float(values.mean()) if not values.empty else 0.0
            else:
                modes = values.mode()
                self.fallback_values_[column] = modes.iloc[0] if not modes.empty else "__MISSING__"
        self.total_batches_ = 0
        self.fallback_batches_ = 0
        return self

    def _fallback(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        for column in result:
            result[column] = result[column].fillna(self.fallback_values_[column])
        return result

    def _prompt(self, frame: pd.DataFrame) -> str:
        records = frame.astype(object).where(frame.notna(), None).to_dict(orient="records")
        return f"""You are a careful data analyst performing missing-data imputation.
I am providing a subset of the {self.dataset_name} dataset.

TASK
Replace every null using row context, column semantics, and dataset context.

STRICT RULES
- Return only a JSON array of objects, without Markdown or explanation.
- Keep exactly the same rows, columns, order, and non-null values.
- Never return NaN, null, '?', or a new column.
- Do not predict or discuss the fraud target.

INPUT
{json.dumps(records, ensure_ascii=False, default=str)}
"""

    def _complete(self, prompt: str) -> str:
        if self.completion_fn is not None:
            return self.completion_fn(prompt, self.model, self.temperature)
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    @staticmethod
    def _same_observed(original: pd.Series, candidate: pd.Series) -> bool:
        observed = original.notna()
        left = original.loc[observed]
        right = candidate.loc[observed]
        if pd.api.types.is_numeric_dtype(original):
            return bool(np.allclose(
                pd.to_numeric(left), pd.to_numeric(right, errors="coerce"),
                rtol=1e-9, atol=1e-12, equal_nan=False,
            ))
        return left.astype("string").equals(right.astype("string"))

    def _validate(self, original: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
        if candidate.shape != original.shape or list(candidate.columns) != list(original.columns):
            raise ValueError("LLM response has unexpected rows or columns")
        for column in original:
            if pd.api.types.is_numeric_dtype(original[column]):
                candidate[column] = pd.to_numeric(candidate[column], errors="coerce")
            else:
                candidate[column] = candidate[column].astype("string")
            if candidate[column].isna().any():
                raise ValueError(f"LLM returned an invalid value in {column!r}")
            if not self._same_observed(original[column], candidate[column]):
                raise ValueError(f"LLM changed observed values in {column!r}")
        return candidate

    def _impute_batch(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not frame.isna().any().any():
            return frame.copy()
        self.total_batches_ += 1
        for attempt in range(self.max_retries):
            try:
                payload = self._complete(self._prompt(frame)).strip()
                if payload.startswith("```"):
                    payload = payload.split("\n", 1)[1].rsplit("```", 1)[0]
                candidate = pd.DataFrame(json.loads(payload), index=frame.index)
                return self._validate(frame, candidate)
            except Exception:  # API, transport, JSON, shape, and value failures all retry.
                if attempt + 1 < self.max_retries:
                    time.sleep(self.retry_base_seconds * (2**attempt))
        self.fallback_batches_ += 1
        return self._fallback(frame)

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "fallback_values_"):
            raise RuntimeError("fit must be called before transform")
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if len(data) > self.max_total_rows and not self.allow_large_dataset:
            estimated = int(np.ceil(len(data) / self.max_rows_per_request) *
                            np.ceil(len(self.feature_columns_) / self.max_columns_per_request))
            raise ValueError(
                f"refusing {len(data):,} rows (about {estimated:,} API calls); "
                f"sample at most {self.max_total_rows:,} rows or explicitly set allow_large_dataset=True"
            )
        result = data.copy()
        for row_start in range(0, len(result), self.max_rows_per_request):
            rows = result.index[row_start:row_start + self.max_rows_per_request]
            for col_start in range(0, len(self.feature_columns_), self.max_columns_per_request):
                columns = self.feature_columns_[col_start:col_start + self.max_columns_per_request]
                result.loc[rows, columns] = self._impute_batch(result.loc[rows, columns])
        return result

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.fit(data).transform(data)

    @property
    def fallback_rate_(self) -> float:
        return self.fallback_batches_ / self.total_batches_ if self.total_batches_ else 0.0
