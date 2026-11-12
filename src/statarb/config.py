"""Project-wide conventions and constants.

Conventions (also stated in the README):

* Daily close-to-close data, ``TRADING_DAYS = 252`` trading days per year.
* Returns are stored as decimals (0.01 means +1%); costs are quoted in basis
  points (1 bp = 0.01% = 1e-4).
* A signal computed with data up to day t is traded at day t+1.
"""

import os
from pathlib import Path

TRADING_DAYS: int = 252
"""Trading days per year, used to annualise daily statistics."""

BP: float = 1e-4
"""One basis point as a decimal."""

SEED: int = 42
"""Default random seed. Pass seeds explicitly; this is only the fallback."""

DEFAULT_START: str = "2005-01-01"
DEFAULT_END: str = "2025-12-31"

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = Path(os.environ.get("STARB_DATA_DIR", PROJECT_ROOT / "data"))
"""Root of the git-ignored data cache; override with the ``STARB_DATA_DIR`` env var."""

RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"


def bps_to_decimal(bps: float) -> float:
    """Convert basis points to a decimal fraction (10 bp -> 0.001)."""
    return bps * BP
