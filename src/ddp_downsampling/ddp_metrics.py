from __future__ import annotations

from typing import List, Tuple

import numpy as np


def select_representative_cell_indices(vol_window: np.ndarray) -> List[int]:
    """Select representative cell indices for a voltage window.

    This follows Supplemental Methods S5:
    - max instantaneous voltage
    - min instantaneous voltage
    - max voltage range (max-min within the window)
    - max variance
    - min variance
    """
    vol_window = np.asarray(vol_window, dtype=float)
    if vol_window.ndim != 2 or vol_window.shape[1] == 0:
        return []

    cell_max = vol_window.max(axis=0)
    cell_min = vol_window.min(axis=0)
    cell_range = cell_max - cell_min
    cell_var = vol_window.var(axis=0)

    idx_max = int(np.argmax(cell_max))
    idx_min = int(np.argmin(cell_min))
    idx_range = int(np.argmax(cell_range))
    idx_var_max = int(np.argmax(cell_var))
    idx_var_min = int(np.argmin(cell_var))

    indices: List[int] = []
    for idx in [idx_max, idx_min, idx_range, idx_var_max, idx_var_min]:
        if idx not in indices:
            indices.append(idx)

    # Ensure at least 2 indices if possible (so divergence is defined).
    if len(indices) < 2 and vol_window.shape[1] > 1:
        for idx in range(vol_window.shape[1]):
            if idx not in indices:
                indices.append(idx)
            if len(indices) >= 2:
                break
    return indices


def _apply_square_wave(seq: np.ndarray, amplitude: float) -> np.ndarray:
    if amplitude == 0.0:
        return seq
    signs = np.where((np.arange(seq.size) % 2) == 0, -1.0, 1.0)
    return seq + amplitude * signs


def divergence_metric(x1: np.ndarray, x2: np.ndarray, alpha: float, beta: float, eps: float = 1e-12) -> float:
    """Custom divergence in Supplemental Methods S5."""
    x1 = np.asarray(x1, dtype=float).reshape(-1)
    x2 = np.asarray(x2, dtype=float).reshape(-1)
    if x1.size == 0 or x2.size == 0:
        return 0.0

    x1 = x1 - x1.mean()
    x2 = x2 - x2.mean()

    prod = x1 * x2
    neg_mask = prod < 0
    if np.any(neg_mask):
        prod = prod.copy()
        prod[neg_mask] = -alpha * (np.abs(prod[neg_mask]) ** beta)

    denom = float(x1.std() * x2.std())
    if denom < eps:
        return 0.0
    return 1.0 - float(prod.mean() / denom)


def compute_vdd(
    vol_window: np.ndarray,
    alpha: float,
    beta: float,
    *,
    use_square_wave: bool,
    square_wave_amplitude: float,
    eps: float = 1e-12,
) -> float:
    indices = select_representative_cell_indices(vol_window)
    if len(indices) < 2:
        return 0.0

    amp = float(square_wave_amplitude) if use_square_wave else 0.0
    seqs = [_apply_square_wave(np.asarray(vol_window[:, idx], dtype=float), amp) for idx in indices]

    max_div = 0.0
    for i in range(len(seqs)):
        for j in range(i + 1, len(seqs)):
            div = divergence_metric(seqs[i], seqs[j], alpha=alpha, beta=beta, eps=eps)
            if div > max_div:
                max_div = float(div)
    return float(max(max_div, 0.0))


def compute_cdd(
    vol_window: np.ndarray,
    cur_window: np.ndarray,
    *,
    use_square_wave: bool,
    square_wave_amplitude: float,
    current_range_floor: float,
    eps: float = 1e-12,
) -> float:
    vol_window = np.asarray(vol_window, dtype=float)
    cur = np.asarray(cur_window, dtype=float).reshape(-1)
    if vol_window.size == 0 or cur.size == 0:
        return 0.0

    amp = float(square_wave_amplitude) if use_square_wave else 0.0
    cur_adj = _apply_square_wave(cur, amp) if amp != 0.0 else cur

    denom = float(cur_adj.max() - cur_adj.min())
    denom = max(denom, float(current_range_floor))
    if denom < eps:
        return 0.0

    v_range = float(vol_window.max() - vol_window.min())
    return float(v_range / denom)


def compute_vdd_cdd(
    vol_window: np.ndarray,
    cur_window: np.ndarray,
    *,
    alpha: float,
    beta: float,
    use_square_wave_vdd: bool,
    voltage_square_wave_amplitude: float,
    use_square_wave_cdd: bool,
    current_square_wave_amplitude: float,
    current_range_floor: float,
) -> Tuple[float, float]:
    vdd = compute_vdd(
        vol_window,
        alpha=alpha,
        beta=beta,
        use_square_wave=use_square_wave_vdd,
        square_wave_amplitude=voltage_square_wave_amplitude,
    )
    cdd = compute_cdd(
        vol_window,
        cur_window,
        use_square_wave=use_square_wave_cdd,
        square_wave_amplitude=current_square_wave_amplitude,
        current_range_floor=current_range_floor,
    )
    return float(vdd), float(cdd)

