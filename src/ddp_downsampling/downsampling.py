from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .ddp_metrics import compute_vdd_cdd
from .segmentation import enforce_min_lengths


TIME_COL_CANDIDATES = ("time_s", "orig_idx")


@dataclass(frozen=True)
class DownsampleResult:
    downsampled_df: pd.DataFrame
    vdd: np.ndarray
    cdd: np.ndarray
    base_grades: np.ndarray
    fixed_grades: np.ndarray
    index_sets: Dict[str, np.ndarray]


def _infer_time_seconds(df: pd.DataFrame, dt: float) -> np.ndarray:
    for col in TIME_COL_CANDIDATES:
        if col in df.columns:
            return df[col].to_numpy(dtype=float)
    return (np.arange(len(df), dtype=float) * float(dt)).astype(float)


def _window_bounds(length: int, pos: int, win_size: int, anchor: str) -> Tuple[int, int]:
    if length <= 0:
        return 0, 0
    win_size = max(1, int(win_size))
    anchor = (anchor or "center").strip().lower()
    if anchor == "left":
        start = pos
        end = start + win_size - 1
    else:
        half = win_size // 2
        start = pos - half
        end = start + win_size - 1

    if start < 0:
        start = 0
        end = start + win_size - 1
    if end >= length:
        end = length - 1
        start = max(0, end - win_size + 1)
    return int(start), int(end)


def compute_grades(vol: np.ndarray, cur: np.ndarray, cfg: PipelineConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute VDD/CDD series and point-wise grades (2/1/0)."""
    vol = np.asarray(vol, dtype=float)
    cur = np.asarray(cur, dtype=float).reshape(-1)
    length = int(cur.size)
    if length == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0, dtype=int)

    win_size = max(1, int(cfg.ddp_window_size))
    step = max(1, int(cfg.ddp_step_size))

    vdd_values = np.zeros(length, dtype=float)
    cdd_values = np.zeros(length, dtype=float)
    grades = np.zeros(length, dtype=int)

    cur_range = float(cur.max() - cur.min())
    if cur_range < 1e-6:
        cur_range = float(np.max(np.abs(cur)))
    cur_range = max(cur_range, 1e-6)

    for pos in range(0, length, step):
        start, end = _window_bounds(length, pos, win_size, cfg.ddp_window_anchor)
        vdd, cdd = compute_vdd_cdd(
            vol[start : end + 1, :],
            cur[start : end + 1],
            alpha=float(cfg.ddp_alpha),
            beta=float(cfg.ddp_beta),
            use_square_wave_vdd=bool(cfg.enable_square_wave_vdd),
            voltage_square_wave_amplitude=float(cfg.voltage_square_wave_amplitude),
            use_square_wave_cdd=bool(cfg.enable_square_wave_cdd),
            current_square_wave_amplitude=float(cfg.current_square_wave_amplitude),
            current_range_floor=cur_range,
        )
        vdd_values[pos] = float(vdd)
        cdd_values[pos] = float(cdd)

        if vdd > cfg.vdd_thres and cdd > cfg.cdd_thres:
            grades[pos] = 2
        elif vdd < cfg.vdd_thres and cdd < cfg.cdd_thres:
            grades[pos] = 0
        else:
            grades[pos] = 1

    # Forward-fill for step > 1 (keep piecewise constant).
    if step > 1:
        last = 0
        for i in range(length):
            if i % step == 0:
                last = i
                continue
            vdd_values[i] = vdd_values[last]
            cdd_values[i] = cdd_values[last]
            grades[i] = grades[last]

    return vdd_values, cdd_values, grades


def _mask_segments(mask: np.ndarray) -> List[Tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    segments: List[Tuple[int, int]] = []
    n = int(mask.size)
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        start = i
        while i + 1 < n and mask[i + 1]:
            i += 1
        segments.append((int(start), int(i)))
        i += 1
    return segments


def _sample_segments(segments: Sequence[Tuple[int, int]], step: int) -> List[int]:
    step = max(1, int(step))
    out: List[int] = []
    for start, end in segments:
        out.extend(range(int(start), int(end) + 1, step))
    return out


def build_index_sets(
    base_grades: np.ndarray,
    fixed_grades: np.ndarray,
    dt: float,
    cfg: PipelineConfig,
) -> Dict[str, np.ndarray]:
    """Build sampling index sets for high/mid/low + their union.

    overlap_mode:
      - share: allow overlap between frequency sets (similar to the reference pipeline).
      - exclusive: disjoint sets derived from fixed_grades.
    """
    dt = float(max(dt, 1e-9))
    step_high = max(1, int(round(cfg.sampling_interval_high_s / dt)))
    step_mid = max(1, int(round(cfg.sampling_interval_mid_s / dt)))
    step_low = max(1, int(round(cfg.sampling_interval_low_s / dt)))

    overlap = (cfg.overlap_mode or "share").strip().lower()

    base_grades = np.asarray(base_grades, dtype=int).reshape(-1)
    fixed_grades = np.asarray(fixed_grades, dtype=int).reshape(-1)

    if overlap == "exclusive":
        high_mask = fixed_grades == 2
        mid_mask = fixed_grades == 1
        low_mask = fixed_grades == 0
    else:
        # "share" mode:
        # - high: fixed HIGH
        # - mid : base MID OR fixed MID (covers MID even if swallowed or expanded)
        # - low : base LOW OR fixed LOW (covers LOW even if swallowed from MID/HIGH)
        high_mask = fixed_grades == 2
        mid_mask = (base_grades == 1) | (fixed_grades == 1)
        low_mask = (base_grades == 0) | (fixed_grades == 0)

    idx_high = sorted(set(_sample_segments(_mask_segments(high_mask), step_high)))
    idx_mid = sorted(set(_sample_segments(_mask_segments(mid_mask), step_mid)))
    idx_low = sorted(set(_sample_segments(_mask_segments(low_mask), step_low)))
    all_idx = np.array(sorted(set(idx_high) | set(idx_mid) | set(idx_low)), dtype=int)

    return {
        "all": all_idx,
        "high": np.array(idx_high, dtype=int),
        "mid": np.array(idx_mid, dtype=int),
        "low": np.array(idx_low, dtype=int),
    }


def downsample_dataframe(df: pd.DataFrame, cfg: PipelineConfig) -> DownsampleResult:
    def _sort_cols(prefix: str) -> List[str]:
        cols = [c for c in df.columns if c.startswith(prefix)]
        def _key(name: str) -> Tuple[int, str]:
            try:
                return int(name[len(prefix) :]), name
            except ValueError:
                return 10**9, name
        return sorted(cols, key=_key)

    vol_cols = _sort_cols("vol_")
    if not vol_cols:
        raise ValueError("No voltage columns found. Expected columns like vol_1 ... vol_N.")
    if "cur" not in df.columns:
        raise ValueError("No current column found. Expected a `cur` column.")

    vol = df[vol_cols].to_numpy(dtype=float)
    cur = df["cur"].to_numpy(dtype=float)

    vdd, cdd, base_grades = compute_grades(vol, cur, cfg)

    min_lengths = {2: int(cfg.min_length_high), 1: int(cfg.min_length_mid), 0: int(cfg.min_length_low)}
    fixed_grades = enforce_min_lengths(
        base_grades,
        min_lengths=min_lengths,
        short_segment_policy=str(cfg.short_segment_policy),
    )

    dt = float(cfg.output_interval_s)
    index_sets = build_index_sets(base_grades, fixed_grades, dt=dt, cfg=cfg)
    indices = index_sets["all"]

    time_s = _infer_time_seconds(df, dt=dt)
    if indices.size == 0:
        raise ValueError("No indices selected by the downsampling policy. Check thresholds/intervals.")

    out = pd.DataFrame(
        {
            "orig_idx": np.round(time_s[indices].astype(float), 6),
            "orig_row": indices.astype(int),
        }
    )

    sel_high = set(index_sets["high"].tolist())
    sel_mid = set(index_sets["mid"].tolist())
    sel_low = set(index_sets["low"].tolist())
    out["sel_high"] = [1 if int(i) in sel_high else 0 for i in indices]
    out["sel_mid"] = [1 if int(i) in sel_mid else 0 for i in indices]
    out["sel_low"] = [1 if int(i) in sel_low else 0 for i in indices]

    # Preserve original columns (excluding time columns to avoid duplicates).
    exclude = set(TIME_COL_CANDIDATES) | {"orig_idx"}
    for col in df.columns:
        if col in exclude:
            continue
        out[col] = df[col].to_numpy()[indices]

    out["vdd"] = np.round(vdd[indices], 6)
    out["cdd"] = np.round(cdd[indices], 6)

    return DownsampleResult(
        downsampled_df=out,
        vdd=vdd,
        cdd=cdd,
        base_grades=base_grades,
        fixed_grades=fixed_grades,
        index_sets=index_sets,
    )


def downsample_csv_file(input_path: Path, output_path: Path, cfg: PipelineConfig) -> DownsampleResult:
    df = pd.read_csv(input_path)
    result = downsample_dataframe(df, cfg)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.downsampled_df.to_csv(output_path, index=False, float_format="%.6f")
    return result


def find_csv_files(input_dir: Path) -> List[Path]:
    return sorted([p for p in input_dir.rglob("*.csv") if p.is_file() and p.name != "manifest.csv"])


def mirror_output_path(input_path: Path, input_root: Path, output_root: Path) -> Path:
    rel = input_path.relative_to(input_root)
    return output_root / rel
