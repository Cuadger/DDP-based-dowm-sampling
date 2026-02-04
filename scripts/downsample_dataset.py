from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_src_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    if src.as_posix() not in sys.path:
        sys.path.insert(0, src.as_posix())


def main() -> None:
    _bootstrap_src_path()
    from ddp_downsampling.cli_downsample import main as _main

    _main()


if __name__ == "__main__":
    main()

