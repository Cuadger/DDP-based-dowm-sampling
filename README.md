# DDP-Guided Adaptive Downsampling

Adaptive downsampling tool for battery time-series data based on DDP (VDD/CDD) metrics.

## Overview

This tool segments time-series data into HIGH/MID/LOW regions using VDD (Voltage Divergence Degree) and CDD (Current Divergence Degree) metrics, then applies different sampling intervals to each region for adaptive downsampling.

## Input Data Format (CSV)

Each input CSV file must contain:

- `cur`: Pack current (float)
- `vol_1 ... vol_N`: Cell voltages (float)

Optional columns:
- `temp_1 ... temp_N`: Cell temperatures (float)
- `time_s` or `orig_idx`: Timestamps in seconds

## Quick Start

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
```

### Run Downsampling

```bash
ddp-downsample \
  --input-dir /path/to/raw_csv \
  --output-dir /path/to/output \
  --config config/pipeline_params.csv \
  --stats
```

When `--stats` is enabled, a `downsampling_stats.csv` file will be generated in the output directory along with a compression report.

## Configuration (`config/pipeline_params.csv`)

Configuration file uses key/value CSV format.

### Sampling Period and Target Intervals

| Parameter | Description | Default |
|-----------|-------------|---------|
| `output_interval` | Original sampling period (seconds) | 1.0 |
| `sampling_interval_high` | HIGH region sampling interval (seconds) | 1.0 |
| `sampling_interval_mid` | MID region sampling interval (seconds) | 5.0 |
| `sampling_interval_low` | LOW region sampling interval (seconds) | 30.0 |

### VDD/CDD and DDP Grading

| Parameter | Description | Default |
|-----------|-------------|---------|
| `ddp_window_size` | Window length (samples) | 50 |
| `ddp_step_size` | Computation step (samples) | 1 |
| `ddp_window_anchor` | Window anchor (`center` / `left`) | center |
| `ddp_alpha` | VDD hyperparameter | 0.3 |
| `ddp_beta` | VDD hyperparameter | 2.0 |
| `vdd_thres` | VDD threshold (HIGH/MID boundary) | 0.415 |
| `cdd_thres` | CDD threshold (MID/LOW boundary) | 0.007 |

### Square Wave Injection

| Parameter | Description | Default |
|-----------|-------------|---------|
| `ddp_enable_square_wave_vdd` | Enable voltage square wave injection | 1 |
| `ddp_enable_square_wave_cdd` | Enable current square wave injection | 1 |
| `ddp_voltage_square_wave_amplitude` | Voltage square wave amplitude | 1e-4 |
| `ddp_current_square_wave_amplitude` | Current square wave amplitude | 1e-3 |

### Minimum Segment Length and Short Segment Policy

| Parameter | Description | Default |
|-----------|-------------|---------|
| `min_length_high` | Minimum HIGH segment length (samples) | 20 |
| `min_length_mid` | Minimum MID segment length (samples) | 10 |
| `min_length_low` | Minimum LOW segment length (samples) | 50 |
| `short_segment_policy` | Short segment handling (`extend` / `discard`) | extend |

### Frequency Set Overlap Mode

| Parameter | Description | Default |
|-----------|-------------|---------|
| `overlap_mode` | Overlap mode (`share` / `exclusive`) | share |

- `share`: Allow overlap between frequency sets
- `exclusive`: Strictly disjoint; each timestamp belongs to only one frequency set

## Output Format

Downsampled CSV contains:

| Column | Description |
|--------|-------------|
| `orig_idx` | Original timestamp (seconds) |
| `orig_row` | Original row index |
| `sel_high` | HIGH frequency set membership (0/1) |
| `sel_mid` | MID frequency set membership (0/1) |
| `sel_low` | LOW frequency set membership (0/1) |
| `vol_*`, `temp_*`, `cur` | Original signal columns |
| `vdd`, `cdd` | DDP metric values for each row |

## Tuning Guide

**To increase compression ratio:**
- Increase `sampling_interval_mid` / `sampling_interval_low`
- Raise `vdd_thres` / `cdd_thres` to classify more regions as LOW

**To preserve more transient details:**
- Decrease `sampling_interval_mid` / `sampling_interval_low`
- Lower `vdd_thres` / `cdd_thres` to classify more regions as HIGH

## Dependencies

- Python >= 3.9
- numpy >= 1.23
- pandas >= 2.0
