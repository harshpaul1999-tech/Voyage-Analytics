"""Validated loaders for the three travel-capstone datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from src.config import data_path


SCHEMAS = {
    "flights": {
        "travelCode", "userCode", "from", "to", "flightType", "price",
        "time", "distance", "agency", "date",
    },
    "hotels": {
        "travelCode", "userCode", "name", "place", "days", "price", "total", "date",
    },
    "users": {"code", "company", "name", "gender", "age"},
}


def _load(name: str, path: Optional[Path] = None) -> pd.DataFrame:
    """Load a CSV and fail early if its expected columns are absent."""
    csv_path = path or data_path(f"{name}.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {name} dataset: {csv_path}")
    frame = pd.read_csv(csv_path)
    missing = SCHEMAS[name] - set(frame.columns)
    if missing:
        raise ValueError(f"{csv_path.name} is missing columns: {sorted(missing)}")
    return frame


def load_flights(path: Optional[Path] = None) -> pd.DataFrame:
    """Load flights with a parsed booking date."""
    frame = _load("flights", path)
    frame["date"] = pd.to_datetime(frame["date"], format="%m/%d/%Y", errors="coerce")
    return frame


def load_hotels(path: Optional[Path] = None) -> pd.DataFrame:
    """Load hotels with a parsed booking date."""
    frame = _load("hotels", path)
    frame["date"] = pd.to_datetime(frame["date"], format="%m/%d/%Y", errors="coerce")
    return frame


def load_users(path: Optional[Path] = None) -> pd.DataFrame:
    """Load users and normalize text fields without changing label semantics."""
    frame = _load("users", path)
    for column in ("company", "name", "gender"):
        frame[column] = frame[column].astype(str).str.strip()
    return frame


def missing_columns(frame: pd.DataFrame, required: Iterable[str]) -> list[str]:
    """Return required column names that do not occur in ``frame``."""
    return sorted(set(required) - set(frame.columns))
