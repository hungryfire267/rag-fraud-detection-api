"""Model-selectable LLM imputation with validation, retries, and fallback."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    model: str
    base_url: str | None = None


class LLMImputation:
    """Impute tabular batches with a selectable chat model.

    Pass either an OpenAI-compatible ``client`` or a ``completion_fn`` with the
    signature ``(prompt, model, temperature) -> str``. ``available_models`` lets
    callers inspect the endpoint before choosing a model.
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
        temperature: float = 0.1,
        max_rows: int = 40,
        max_columns: int = 10,
        max_retries: int = 3,
        retry_base_seconds: float = 1.0,
        dataset_name: str = "Unknown",
    ) -> None:
        if client is None and completion_fn is None:
            raise ValueError("provide an OpenAI-compatible client or completion_fn")
        self.model = model
        self.client = client
        self.completion_fn = completion_fn
        self.temperature = temperature
        self.max_rows = max_rows
        self.max_columns = max_columns
        self.max_retries = max_retries
        self.retry_base_seconds = retry_base_seconds
        self.dataset_name = dataset_name
        self.fallback_count_ = 0

    @classmethod
    def paper_models(cls) -> list[dict[str, str | None]]:
        """Return the models evaluated by the referenced paper."""
        return [spec.__dict__.copy() for spec in cls.PAPER_MODELS]

    def available_models(self) -> list[str]:
        """List model IDs exposed by the configured client endpoint."""
        if self.client is None or not hasattr(self.client, "models"):
            return [spec.model for spec in self.PAPER_MODELS]
        response = self.client.models.list()
        return sorted(item.id for item in response.data)

    def set_model(self, model: str, *, validate: bool = False):
        if validate and model not in self.available_models():
            raise ValueError(f"model is not available: {model}")
        self.model = model
        return self

    @staticmethod
    def _fallback(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        for column in result.columns:
            if not result[column].isna().any():
                continue
            if pd.api.types.is_numeric_dtype(result[column]):
                value = result[column].mean()
            else:
                modes = result[column].mode(dropna=True)
                value = modes.iloc[0] if not modes.empty else "missing"
            result[column] = result[column].fillna(value)
        return result

    def _prompt(self, frame: pd.DataFrame) -> str:
        records = frame.where(frame.notna(), None).to_dict(orient="records")
        return f"""You are a careful data analyst performing missing-data imputation.
I am providing a subset of the {self.dataset_name} dataset.

TASK
Replace every null value using the row context, column semantics, and dataset context.

CONSTRAINTS
- Do not execute or describe an imputation algorithm.
- Do not add, remove, rename, or reorder rows or columns.
- Preserve every non-null value exactly.
- Never output NaN, null, '?', explanations, or Markdown.

OUTPUT FORMAT
Return only a valid JSON array of objects with exactly this shape and row order:
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
    def _validate(original: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
        if candidate.shape != original.shape or list(candidate.columns) != list(original.columns):
            raise ValueError("LLM response has an unexpected shape or columns")
        if candidate.isna().any().any():
            raise ValueError("LLM response still contains missing values")
        observed = original.notna()
        for column in original.columns:
            if not candidate.loc[observed[column], column].astype(str).equals(
                original.loc[observed[column], column].astype(str)
            ):
                raise ValueError(f"LLM changed observed values in {column!r}")
        return candidate

    def _impute_batch(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not frame.isna().any().any():
            return frame.copy()
        for attempt in range(self.max_retries):
            try:
                payload = self._complete(self._prompt(frame)).strip()
                if payload.startswith("```"):
                    payload = payload.split("\n", 1)[1].rsplit("```", 1)[0]
                candidate = pd.DataFrame(json.loads(payload), index=frame.index)
                return self._validate(frame, candidate)
            except (ValueError, TypeError, KeyError, json.JSONDecodeError):
                if attempt + 1 < self.max_retries:
                    time.sleep(self.retry_base_seconds * (2**attempt))
        self.fallback_count_ += 1
        return self._fallback(frame)

    def fit(self, data: pd.DataFrame):
        if not isinstance(data, pd.DataFrame) or data.empty:
            raise TypeError("data must be a non-empty pandas DataFrame")
        self.columns_ = data.columns.copy()
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "columns_"):
            raise RuntimeError("fit must be called before transform")
        if list(data.columns) != list(self.columns_):
            raise ValueError("transform columns differ from fit columns")
        result = data.copy()
        # Paper-aligned sliding batches: at most 40 rows by 10 columns/request.
        for row_start in range(0, len(result), self.max_rows):
            rows = result.index[row_start : row_start + self.max_rows]
            for col_start in range(0, result.shape[1], self.max_columns):
                columns = result.columns[col_start : col_start + self.max_columns]
                result.loc[rows, columns] = self._impute_batch(result.loc[rows, columns])
        return result

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.fit(data).transform(data)

    @property
    def fallback_rate_(self) -> float:
        """Number of failed batches; retained explicitly for hallucination audits."""
        return float(self.fallback_count_)
