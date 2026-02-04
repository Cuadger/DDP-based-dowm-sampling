from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


def load_kv_csv(path: Path) -> Dict[str, str]:
    """Load a key/value CSV as a dict.

    Expected header: key,value (header is optional; extra columns are ignored).
    Empty keys are ignored. Values are stored as raw strings.
    """
    cfg: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 2:
                continue
            key = str(row[0]).strip()
            if not key or key.lower() == "key":
                continue
            cfg[key] = str(row[1]).strip()
    return cfg


def get_str(cfg: Dict[str, str], key: str, default: str) -> str:
    raw = cfg.get(key)
    return default if raw is None else str(raw)


def get_float(cfg: Dict[str, str], key: str, default: float) -> float:
    raw = cfg.get(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def get_int(cfg: Dict[str, str], key: str, default: int) -> int:
    raw = cfg.get(key)
    if raw is None:
        return default
    try:
        return int(float(raw))
    except ValueError:
        return default


def get_bool(cfg: Dict[str, str], key: str, default: bool) -> bool:
    raw = cfg.get(key)
    if raw is None:
        return default
    s = str(raw).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return default


@dataclass(frozen=True)
class PipelineConfig:
    """Typed view for frequently used parameters.

    The raw CSV config remains the source of truth. This class only covers the
    keys used by this repo, with reasonable defaults.
    """

    output_interval_s: float = 1.0

    ddp_window_size: int = 50
    ddp_step_size: int = 1
    ddp_window_anchor: str = "center"  # "center" | "left"
    ddp_alpha: float = 0.3
    ddp_beta: float = 2.0
    vdd_thres: float = 0.415
    cdd_thres: float = 0.007

    enable_square_wave_vdd: bool = True
    enable_square_wave_cdd: bool = True
    voltage_square_wave_amplitude: float = 1e-4
    current_square_wave_amplitude: float = 1e-3

    sampling_interval_high_s: float = 1.0
    sampling_interval_mid_s: float = 5.0
    sampling_interval_low_s: float = 30.0

    min_length_high: int = 20
    min_length_mid: int = 10
    min_length_low: int = 50

    short_segment_policy: str = "extend"  # "extend" | "discard"
    overlap_mode: str = "share"  # "share" | "exclusive"

    @staticmethod
    def from_dict(cfg: Dict[str, str]) -> "PipelineConfig":
        return PipelineConfig(
            output_interval_s=get_float(cfg, "output_interval", 1.0),
            ddp_window_size=max(1, get_int(cfg, "ddp_window_size", 50)),
            ddp_step_size=max(1, get_int(cfg, "ddp_step_size", 1)),
            ddp_window_anchor=get_str(cfg, "ddp_window_anchor", "center"),
            ddp_alpha=get_float(cfg, "ddp_alpha", 0.3),
            ddp_beta=get_float(cfg, "ddp_beta", 2.0),
            vdd_thres=get_float(cfg, "vdd_thres", 0.415),
            cdd_thres=get_float(cfg, "cdd_thres", 0.007),
            enable_square_wave_vdd=get_bool(cfg, "ddp_enable_square_wave_vdd", True),
            enable_square_wave_cdd=get_bool(cfg, "ddp_enable_square_wave_cdd", True),
            voltage_square_wave_amplitude=get_float(cfg, "ddp_voltage_square_wave_amplitude", 1e-4),
            current_square_wave_amplitude=get_float(cfg, "ddp_current_square_wave_amplitude", 1e-3),
            sampling_interval_high_s=get_float(cfg, "sampling_interval_high", 1.0),
            sampling_interval_mid_s=get_float(cfg, "sampling_interval_mid", 5.0),
            sampling_interval_low_s=get_float(cfg, "sampling_interval_low", 30.0),
            min_length_high=max(1, get_int(cfg, "min_length_high", 20)),
            min_length_mid=max(1, get_int(cfg, "min_length_mid", 10)),
            min_length_low=max(1, get_int(cfg, "min_length_low", 50)),
            short_segment_policy=get_str(cfg, "short_segment_policy", "extend").strip().lower(),
            overlap_mode=get_str(cfg, "overlap_mode", "share").strip().lower(),
        )

