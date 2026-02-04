from __future__ import annotations

import numpy as np

from ddp_downsampling.segmentation import enforce_min_lengths


def test_extend_short_high_segment_steals_from_low_neighbors() -> None:
    base = np.array([0, 0, 2, 2, 0, 0], dtype=int)
    fixed = enforce_min_lengths(base, min_lengths={2: 4, 1: 1, 0: 1}, short_segment_policy="extend")
    assert fixed.tolist() == [0, 2, 2, 2, 2, 0]


def test_discard_short_high_segment_merges_into_neighbor() -> None:
    base = np.array([0, 0, 2, 2, 0, 0], dtype=int)
    fixed = enforce_min_lengths(base, min_lengths={2: 4, 1: 1, 0: 1}, short_segment_policy="discard")
    assert fixed.tolist() == [0, 0, 0, 0, 0, 0]

