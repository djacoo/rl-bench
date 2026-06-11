"""Add gitignored vendor/macura-upstream to sys.path for mbrl imports."""

import sys
from pathlib import Path

_VENDOR = Path(__file__).resolve().parents[2] / "vendor" / "macura-upstream"


def ensure_upstream() -> Path:
    if not _VENDOR.is_dir():
        raise RuntimeError(
            "Missing vendor/macura-upstream. Run: bash scripts/sync_macura_upstream.sh"
        )
    root = str(_VENDOR)
    if root not in sys.path:
        sys.path.insert(0, root)
    return _VENDOR
