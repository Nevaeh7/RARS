"""Maximum catalog compatibility below each SID prefix."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class PrefixAssignment:
    key_to_row: dict[tuple[int, ...], int]
    item_to_row: np.ndarray


def build_prefix_assignments(sid_token_ids: np.ndarray) -> list[PrefixAssignment]:
    result = []
    for depth in range(1, sid_token_ids.shape[1] + 1):
        key_to_row = {}
        assignments = np.empty(sid_token_ids.shape[0], dtype=np.int32)
        for item_index, values in enumerate(sid_token_ids[:, :depth]):
            key = tuple(int(value) for value in values)
            row = key_to_row.get(key)
            if row is None:
                row = len(key_to_row)
                key_to_row[key] = row
            assignments[item_index] = row
        result.append(PrefixAssignment(key_to_row, assignments))
    return result


def aggregate_prefix_feasibility(
    item_scores: np.ndarray, assignments: Sequence[PrefixAssignment]
) -> list[np.ndarray]:
    result = []
    for assignment in assignments:
        values = np.full(len(assignment.key_to_row), -np.inf, dtype=np.float32)
        np.maximum.at(values, assignment.item_to_row, item_scores)
        values[~np.isfinite(values)] = 0.0
        result.append(values)
    return result
