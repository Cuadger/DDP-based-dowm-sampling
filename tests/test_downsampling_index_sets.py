from __future__ import annotations

import numpy as np

from ddp_downsampling.config import PipelineConfig
from ddp_downsampling.downsampling import build_index_sets


def test_overlap_mode_share_allows_overlap_between_high_and_low() -> None:
    base = np.array([0, 0, 0, 0, 0, 0], dtype=int)
    fixed = np.array([2, 2, 2, 0, 0, 0], dtype=int)
    cfg = PipelineConfig(
        output_interval_s=1.0,
        sampling_interval_high_s=1.0,
        sampling_interval_mid_s=1.0,
        sampling_interval_low_s=1.0,
        overlap_mode="share",
    )
    sets = build_index_sets(base, fixed, dt=1.0, cfg=cfg)
    assert set(sets["high"].tolist()) == {0, 1, 2}
    # base grades are LOW everywhere -> share mode keeps all points in low set.
    assert set(sets["low"].tolist()) == {0, 1, 2, 3, 4, 5}


def test_overlap_mode_exclusive_removes_stolen_points_from_low() -> None:
    base = np.array([0, 0, 0, 0, 0, 0], dtype=int)
    fixed = np.array([2, 2, 2, 0, 0, 0], dtype=int)
    cfg = PipelineConfig(
        output_interval_s=1.0,
        sampling_interval_high_s=1.0,
        sampling_interval_mid_s=1.0,
        sampling_interval_low_s=1.0,
        overlap_mode="exclusive",
    )
    sets = build_index_sets(base, fixed, dt=1.0, cfg=cfg)
    assert set(sets["high"].tolist()) == {0, 1, 2}
    assert set(sets["low"].tolist()) == {3, 4, 5}

