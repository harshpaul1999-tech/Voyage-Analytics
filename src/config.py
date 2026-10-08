"""Shared, repository-relative paths for local, container, and CI execution."""

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"


def data_path(filename: str) -> Path:
    """Return the raw-data path, with a fallback for flat data/ layouts."""
    raw_path = RAW_DATA_DIR / filename
    return raw_path if raw_path.exists() else DATA_DIR / filename


def ensure_project_dirs() -> None:
    """Create runtime output directories without changing source data."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
