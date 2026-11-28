"""Ticker lists and point-in-time S&P 500 membership.

Project universes (see the README for their known biases):

* **U1** (`U1`): SPY, the sector ETFs and the classic pairs. Low survivorship risk.
* **U2** (`LARGE_CAP_100`): about 100 *current* large caps. Survivorship-biased by
  construction: they are large today because they did well.
* **U3**: current S&P 500 members plus sector ETFs, optionally filtered by point-in-time
  membership with `sp500_members_asof`.

Point-in-time membership is rebuilt from two Wikipedia pages: today's constituent list and the
table of historical changes. Both are pinned to fixed revisions so the reconstruction is
reproducible, and the parsed tables are cached under ``config.RAW_DIR``.
"""

from collections.abc import Iterable
from io import StringIO
from pathlib import Path

import pandas as pd

from statarb import config

SECTOR_ETFS: dict[str, str] = {
    "XLB": "Materials",
    "XLC": "Communication Services",  # launched June 2018
    "XLE": "Energy",
    "XLF": "Financials",
    "XLI": "Industrials",
    "XLK": "Information Technology",
    "XLP": "Consumer Staples",
    "XLRE": "Real Estate",  # launched October 2015
    "XLU": "Utilities",
    "XLV": "Health Care",
    "XLY": "Consumer Discretionary",
}
"""The SPDR Select Sector ETFs, one per GICS sector."""

CLASSIC_PAIRS: list[tuple[str, str]] = [
    ("KO", "PEP"),  # soft drinks
    ("XOM", "CVX"),  # integrated oil
    ("EWA", "EWC"),  # Australia vs Canada country ETFs (commodity economies)
    ("GLD", "GDX"),  # gold vs gold miners
    ("V", "MA"),  # card networks
    ("HD", "LOW"),  # home improvement retail
]
"""Textbook pairs with an economic reason to co-move."""

U1: list[str] = list(
    dict.fromkeys(["SPY", *SECTOR_ETFS, *(t for pair in CLASSIC_PAIRS for t in pair)])
)
"""Universe U1: SPY, sector ETFs and classic-pair legs (order preserved, no duplicates)."""

LARGE_CAP_100_BY_SECTOR: dict[str, list[str]] = {
    "Information Technology": [
        "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CSCO", "ADBE", "CRM", "ACN", "IBM", "INTC",
        "QCOM",
    ],
    "Communication Services": ["GOOGL", "META", "NFLX", "DIS", "CMCSA", "VZ", "T", "TMUS"],
    "Consumer Discretionary": ["AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "BKNG", "TJX"],
    "Consumer Staples": ["WMT", "PG", "KO", "PEP", "COST", "PM", "MO", "MDLZ", "CL"],
    "Health Care": [
        "LLY", "UNH", "JNJ", "ABBV", "MRK", "PFE", "TMO", "ABT", "DHR", "AMGN", "BMY", "GILD",
    ],
    "Financials": [
        "BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "AXP", "C", "BLK", "SCHW",
    ],
    "Industrials": ["GE", "CAT", "HON", "UNP", "RTX", "LMT", "DE", "UPS", "BA", "MMM", "ETN"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "PSX", "MPC", "OXY"],
    "Utilities": ["NEE", "DUK", "SO", "D", "AEP", "EXC"],
    "Real Estate": ["PLD", "AMT", "EQIX", "SPG", "PSA", "O"],
    "Materials": ["LIN", "SHW", "APD", "ECL", "FCX", "NEM", "NUE"],
}  # fmt: skip
"""Universe U2 by GICS sector: 100 large S&P 500 members as of November 2026."""

LARGE_CAP_100: list[str] = [t for tickers in LARGE_CAP_100_BY_SECTOR.values() for t in tickers]

SP500_LIST_URL = "https://en.wikipedia.org/w/index.php?title=List_of_S%26P_500_companies"
SP500_CHANGES_URL = (
    "https://en.wikipedia.org/w/index.php?title=Historical_components_of_the_S%26P_500"
)
SP500_LIST_REVISION = 1384572480  # 2026-11-08
SP500_CHANGES_REVISION = 1383679373  # 2026-11-04


def to_yahoo_ticker(ticker: str) -> str:
    """Wikipedia and exchanges write share classes with a dot (BRK.B); Yahoo uses a dash."""
    return ticker.strip().upper().replace(".", "-")


def parse_current_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Tidy Wikipedia's constituent table: ticker, security, sector, sub_industry, date_added."""
    current = pd.DataFrame(
        {
            "ticker": raw["Symbol"].map(to_yahoo_ticker),
            "security": raw["Security"],
            "sector": raw["GICS Sector"],
            "sub_industry": raw["GICS Sub-Industry"],
            "date_added": pd.to_datetime(raw["Date added"], errors="coerce"),
        }
    )
    return current.sort_values("ticker").reset_index(drop=True)


def parse_changes_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Tidy Wikipedia's changes table: one row per change with date, added, removed, reason.

    The raw table has two header rows (e.g. ``("Added", "Ticker")``). A row may have only an
    addition or only a removal; the missing side is NaN.
    """
    if isinstance(raw.columns, pd.MultiIndex):
        raw = raw.copy()
        raw.columns = [" ".join(dict.fromkeys(col)).strip() for col in raw.columns]

    def ticker(value) -> str | None:
        return to_yahoo_ticker(value) if isinstance(value, str) and value.strip() else None

    changes = pd.DataFrame(
        {
            "date": pd.to_datetime(raw["Effective Date"], format="mixed", errors="coerce"),
            "added": raw["Added Ticker"].map(ticker),
            "removed": raw["Removed Ticker"].map(ticker),
            "reason": raw["Reason"],
        }
    )
    changes = changes.dropna(subset=["date"])
    changes = changes[changes["added"].notna() | changes["removed"].notna()]
    return changes.sort_values("date", ascending=False, kind="stable").reset_index(drop=True)


def _fetch_html(url: str, revision: int) -> str:
    import requests

    response = requests.get(
        url,
        params={"oldid": revision},
        headers={"User-Agent": "starb-learning/0.1 (educational project)"},
        timeout=30,
    )
    response.raise_for_status()
    return response.text


def load_sp500_tables(
    cache: bool = True, cache_dir: Path | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Current S&P 500 constituents and the historical changes table (pinned revisions)."""
    cache_dir = Path(cache_dir or config.RAW_DIR)
    current_file = cache_dir / f"sp500_current_{SP500_LIST_REVISION}.parquet"
    changes_file = cache_dir / f"sp500_changes_{SP500_CHANGES_REVISION}.parquet"
    if cache and current_file.exists() and changes_file.exists():
        return pd.read_parquet(current_file), pd.read_parquet(changes_file)

    list_tables = pd.read_html(StringIO(_fetch_html(SP500_LIST_URL, SP500_LIST_REVISION)))
    change_tables = pd.read_html(StringIO(_fetch_html(SP500_CHANGES_URL, SP500_CHANGES_REVISION)))
    current = parse_current_table(list_tables[0])
    changes = parse_changes_table(change_tables[0])
    if cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        current.to_parquet(current_file)
        changes.to_parquet(changes_file)
    return current, changes


def members_asof(
    date: str | pd.Timestamp, current: Iterable[str], changes: pd.DataFrame
) -> list[str]:
    """Index members on ``date``, rebuilt backwards from today's list.

    Start from the current members and undo every change effective *after* ``date``: drop
    the ticker it added and restore the ticker it removed. A change effective on ``date``
    itself is kept, so the result is the membership at the close of ``date``.
    """
    date = pd.Timestamp(date)
    members = set(current)
    later = changes[changes["date"] > date].sort_values("date", ascending=False, kind="stable")
    for added, removed in zip(later["added"], later["removed"], strict=True):
        if pd.notna(added):
            members.discard(added)
        if pd.notna(removed):
            members.add(removed)
    return sorted(members)


def membership_matrix(
    dates: Iterable[pd.Timestamp], current: Iterable[str], changes: pd.DataFrame
) -> pd.DataFrame:
    """Boolean table, dates x tickers: True where the ticker is a member on that date."""
    dates = pd.DatetimeIndex(dates)
    rows = {d: set(members_asof(d, current, changes)) for d in dates}
    tickers = sorted(set().union(*rows.values()))
    return pd.DataFrame(
        [[t in rows[d] for t in tickers] for d in dates], index=dates, columns=tickers
    )


def sp500_members_asof(
    date: str | pd.Timestamp,
    current: pd.DataFrame | None = None,
    changes: pd.DataFrame | None = None,
) -> list[str]:
    """S&P 500 members (Yahoo tickers) on ``date``, from the pinned Wikipedia tables.

    Accuracy is limited by the changes table: older entries are incomplete, and recycled or
    renamed tickers are not tracked. Check ``len(result)`` against ~500.
    """
    if current is None or changes is None:
        current, changes = load_sp500_tables()
    return members_asof(date, current["ticker"], changes)
