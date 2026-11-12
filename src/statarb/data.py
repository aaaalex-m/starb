"""Download, cache and load daily price bars.

Bars come from Yahoo Finance via ``yfinance`` and are cached as one parquet file per ticker
under ``config.RAW_DIR`` (git-ignored), with a small JSON sidecar recording the requested date
range and the download time. Yahoo revises history, so the download time matters.

Column conventions of a standardised bar table (index: tz-naive daily ``date``):

* ``open, high, low, close``: Yahoo's prices. Note that Yahoo's ``close`` is already
  *split-adjusted* but *not dividend-adjusted*.
* ``adj_close``: split- and dividend-adjusted close. Use this for returns.
* ``volume``: shares traded (split-adjusted).
* ``dividends``: cash dividend per share on its ex-dividend date, else 0.
* ``splits``: split ratio on the split date (4.0 for a 4-for-1 split), else 0.
"""

import json
import re
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from statarb import config

BAR_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume", "dividends", "splits"]

_YAHOO_COLUMNS = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
    "Dividends": "dividends",
    "Stock Splits": "splits",
}

Downloader = Callable[[str, pd.Timestamp, pd.Timestamp], pd.DataFrame]
"""``downloader(ticker, start, end)`` returns raw bars for ``start <= date <= end``."""


def standardise_bars(raw: pd.DataFrame) -> pd.DataFrame:
    """Rename yfinance columns to snake case and make the index a tz-naive daily date."""
    bars = raw.rename(columns=_YAHOO_COLUMNS)
    missing = set(BAR_COLUMNS) - set(bars.columns)
    if missing:
        raise ValueError(f"bars are missing columns {sorted(missing)}")
    bars = bars[BAR_COLUMNS].astype(float)
    index = pd.DatetimeIndex(bars.index)
    if index.tz is not None:
        index = index.tz_localize(None)
    bars.index = index.normalize().rename("date")
    bars = bars[~bars.index.duplicated(keep="last")].sort_index()
    return bars


def yahoo_download(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Download daily bars for one ticker from Yahoo Finance (``end`` inclusive)."""
    import yfinance as yf

    # auto_adjust is set explicitly because its default has changed between yfinance versions.
    # False keeps both the raw close and the adjusted close.
    raw = yf.Ticker(ticker).history(
        start=start, end=end + pd.Timedelta(days=1), auto_adjust=False, actions=True
    )
    if raw.empty:
        raise ValueError(f"Yahoo returned no data for {ticker!r} between {start} and {end}")
    return standardise_bars(raw)


def cache_path(ticker: str, cache_dir: Path | None = None) -> Path:
    """Parquet file holding the cached bars for ``ticker``."""
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", ticker)
    return Path(cache_dir or config.RAW_DIR) / f"{safe}.parquet"


def cache_info(ticker: str, cache_dir: Path | None = None) -> dict | None:
    """Metadata of the cached download (date range, download time), or None if not cached."""
    meta_file = cache_path(ticker, cache_dir).with_suffix(".json")
    if not meta_file.exists():
        return None
    return json.loads(meta_file.read_text())


def load_bars(
    ticker: str,
    start: str | pd.Timestamp = config.DEFAULT_START,
    end: str | pd.Timestamp = config.DEFAULT_END,
    cache: bool = True,
    cache_dir: Path | None = None,
    downloader: Downloader | None = None,
) -> pd.DataFrame:
    """Daily OHLCV bars plus dividends and splits for one ticker, ``start``/``end`` inclusive.

    With ``cache=True`` the bars are read from the parquet cache when it already covers the
    requested range; otherwise the union of the cached and requested ranges is downloaded and
    the cache is overwritten, so it only ever grows.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start > end:
        raise ValueError(f"start {start.date()} is after end {end.date()}")
    downloader = downloader or yahoo_download

    if not cache:
        return downloader(ticker, start, end).loc[start:end]

    path = cache_path(ticker, cache_dir)
    meta = cache_info(ticker, cache_dir)
    if meta and pd.Timestamp(meta["start"]) <= start and pd.Timestamp(meta["end"]) >= end:
        return pd.read_parquet(path).loc[start:end]

    if meta:
        start_dl = min(start, pd.Timestamp(meta["start"]))
        end_dl = max(end, pd.Timestamp(meta["end"]))
    else:
        start_dl, end_dl = start, end
    bars = downloader(ticker, start_dl, end_dl)
    path.parent.mkdir(parents=True, exist_ok=True)
    bars.to_parquet(path)
    meta = {
        "ticker": ticker,
        "start": str(start_dl.date()),
        "end": str(end_dl.date()),
        "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": getattr(downloader, "__name__", repr(downloader)),
        "auto_adjust": False,
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return bars.loc[start:end]


def load_prices(
    tickers: str | Iterable[str],
    start: str | pd.Timestamp = config.DEFAULT_START,
    end: str | pd.Timestamp = config.DEFAULT_END,
    field: str = "adj_close",
    cache: bool = True,
    cache_dir: Path | None = None,
    downloader: Downloader | None = None,
) -> pd.DataFrame:
    """Wide table of one bar field (default: adjusted close), dates x tickers.

    Dates are the union over tickers; a ticker that did not trade on a date (for example,
    before it was listed) is NaN there. Cleaning is left to the caller (Module 2).
    """
    if field not in BAR_COLUMNS:
        raise ValueError(f"field must be one of {BAR_COLUMNS}, got {field!r}")
    tickers = [tickers] if isinstance(tickers, str) else list(tickers)
    columns = {
        t: load_bars(t, start, end, cache=cache, cache_dir=cache_dir, downloader=downloader)[field]
        for t in tickers
    }
    prices = pd.DataFrame(columns)
    prices.columns.name = "ticker"
    return prices


def unadjust_splits(close: pd.Series, splits: pd.Series) -> pd.Series:
    """Rebuild the as-traded price from a split-adjusted price.

    ``splits`` holds the split ratio on each split date (0 or NaN elsewhere) and must include
    every split *after* the last date of ``close`` up to the download date, since the
    provider adjusts for all of them. The as-traded price on day t is the adjusted price times
    the product of the ratios of all splits strictly after t.
    """
    ratios = splits.replace(0.0, np.nan).dropna()
    factor = pd.Series(1.0, index=close.index)
    for date, ratio in ratios.items():
        factor[close.index < date] *= ratio
    return close * factor
