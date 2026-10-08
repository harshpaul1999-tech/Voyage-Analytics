"""Interaction-matrix helpers for hotel recommendation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def hotel_key(frame: pd.DataFrame) -> pd.Series:
    """Create a stable item key because hotel names can occur in multiple places."""
    return frame["name"].astype(str).str.strip() + " | " + frame["place"].astype(str).str.strip()


def make_interaction_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate historical stays to a non-negative user--hotel affinity matrix."""
    interactions = frame.copy()
    interactions["hotel_key"] = hotel_key(interactions)
    # Frequency is the primary preference signal; stay length lightly breaks ties.
    interactions["affinity"] = 1.0 + 0.10 * np.clip(interactions["days"].astype(float) - 1, 0, 14)
    return interactions.pivot_table(
        index="userCode", columns="hotel_key", values="affinity", aggfunc="sum", fill_value=0.0
    ).sort_index().sort_index(axis=1)
