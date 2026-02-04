from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .config import PipelineConfig, load_kv_csv
from .downsampling import find_csv_files, mirror_output_path, downsample_csv_file


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DDP-guided adaptive downsampling (VDD/CDD).")
    p.add_argument("--input-dir", type=Path, required=True, help="Directory containing raw CSV files.")
    p.add_argument("--output-dir", type=Path, required=True, help="Directory to write downsampled CSV files.")
    p.add_argument("--config", type=Path, required=True, help="Path to pipeline_params.csv.")
    p.add_argument(
        "--stats",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to write compression stats (raw_len/down_len) and print a report (default path: output_dir/downsampling_stats.csv).",
    )
    p.add_argument(
        "--stats-csv",
        type=Path,
        default=None,
        help="Optional path to write a downsampling summary CSV (implies --stats).",
    )
    return p


def _print_compression_report(df: pd.DataFrame) -> None:
    if df.empty:
        return
    raw_total = int(df["raw_len"].sum())
    down_total = int(df["down_len"].sum())
    ratio_total = (raw_total / down_total) if down_total > 0 else float("inf")
    keep_rate = (down_total / raw_total) if raw_total > 0 else 0.0

    ratios = df["compression_ratio"].dropna()
    ratio_mean = float(ratios.mean()) if not ratios.empty else 0.0
    ratio_min = float(ratios.min()) if not ratios.empty else 0.0
    ratio_max = float(ratios.max()) if not ratios.empty else 0.0

    print("Compression report (raw_len/down_len)")
    print(f"- files: {len(df)}")
    print(f"- total: raw={raw_total} down={down_total} keep={keep_rate:.2%} ratio={ratio_total:.4f}")
    print(f"- per-file ratio: mean={ratio_mean:.4f} min={ratio_min:.4f} max={ratio_max:.4f}")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    cfg_dict = load_kv_csv(args.config)
    cfg = PipelineConfig.from_dict(cfg_dict)

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    files = find_csv_files(input_dir)
    if not files:
        raise SystemExit(f"No CSV files found under: {input_dir}")

    stats_enabled = bool(args.stats or args.stats_csv)
    stats: List[Dict[str, object]] = []
    for src in files:
        dst = mirror_output_path(src, input_root=input_dir, output_root=output_dir)
        result = downsample_csv_file(src, dst, cfg)
        if stats_enabled:
            stats.append(
                {
                    "file": str(src.relative_to(input_dir)),
                    "raw_len": int(result.vdd.size),
                    "down_len": int(len(result.downsampled_df)),
                    "overlap_mode": cfg.overlap_mode,
                    "short_segment_policy": cfg.short_segment_policy,
                }
            )

    if not stats_enabled:
        return

    stats_path = args.stats_csv or (output_dir / "downsampling_stats.csv")
    df = pd.DataFrame(stats)
    if df.empty:
        return
    df["compression_ratio"] = df["raw_len"] / df["down_len"].replace(0, pd.NA)
    df.to_csv(stats_path, index=False, encoding="utf-8")
    _print_compression_report(df)
    print(f"- stats_csv: {stats_path}")


if __name__ == "__main__":
    main()
