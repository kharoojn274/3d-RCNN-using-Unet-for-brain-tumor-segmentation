"""Deterministic patient-level train/validation splitting helpers."""
from __future__ import annotations

import random
from typing import Sequence, Tuple, List


def split_subject_ids(
    subject_ids: Sequence[str],
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """Return disjoint train and validation subject IDs with reproducible order.

    The split is at patient/case level, never at slice or patch level. For small
    datasets this helper requires at least two subjects and ensures both splits
    are non-empty.
    """
    ids = sorted(str(value) for value in subject_ids)
    if len(ids) < 2:
        raise ValueError("At least two subjects are required for a train/validation split")
    if len(set(ids)) != len(ids):
        raise ValueError("subject_ids contains duplicate IDs")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be strictly between 0 and 1")

    shuffled = ids.copy()
    random.Random(seed).shuffle(shuffled)
    n_val = min(len(ids) - 1, max(1, round(len(ids) * validation_fraction)))
    validation_ids = sorted(shuffled[:n_val])
    validation_set = set(validation_ids)
    train_ids = sorted(subject_id for subject_id in ids if subject_id not in validation_set)

    if set(train_ids) & set(validation_ids):
        raise RuntimeError("Train/validation leakage detected")
    if set(train_ids) | set(validation_ids) != set(ids):
        raise RuntimeError("Split omitted or introduced subject IDs")
    return train_ids, validation_ids
