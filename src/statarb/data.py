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


def clean_prices(prices: pd.DataFrame, max_ffill: int = 3) -> pd.DataFrame:
    """Align and lightly repair a wide price table (dates x tickers).

    * sorts the dates and drops duplicate dates (keeping the last);
    * treats non-positive prices as missing;
    * drops dates on which no ticker has a price (holidays on every calendar involved);
    * forward-fills at most ``max_ffill`` consecutive missing prices. Nothing is filled before
      a ticker's first price; longer gaps stay NaN after the first ``max_ffill`` days.

    Forward-filling is causal (it only uses the past), unlike interpolation or back-filling,
    which would leak future prices. Restricting the fill to gaps *between* two prices
    (pandas' ``limit_area="inside"``) would also leak: on the day a price goes missing you
    can't know whether another one will ever come (a halt or a delisting). So the last price
    of a delisted ticker is carried for up to ``max_ffill`` days, as a live system would see
    it. Each filled day is a zero return, so keep the limit small.
    """
    cleaned = prices.sort_index()
    cleaned = cleaned[~cleaned.index.duplicated(keep="last")]
    cleaned = cleaned.where(cleaned > 0)
    cleaned = cleaned.dropna(how="all")
    return cleaned.ffill(limit=max_ffill)


def _longest_run(mask: pd.Series) -> int:
    """Length of the longest run of consecutive True values."""
    if not mask.any():
        return 0
    run_id = (~mask).cumsum()
    return int(mask.groupby(run_id).sum().max())


def flag_returns(prices: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Daily moves larger than ``threshold`` in absolute value (default 50%), largest first.

    Returns are computed between consecutive *available* prices of each ticker, so a gap in
    the data doesn't hide a jump. A flagged move is not necessarily wrong, but each one should
    be explained (bad tick, missed split, takeover, bankruptcy) before the data is trusted.
    """
    rows = []
    for ticker in prices.columns:
        series = prices[ticker].dropna()
        ret = series / series.shift(1) - 1
        for date in ret.index[ret.abs() > threshold]:
            loc = series.index.get_loc(date)
            rows.append((date, ticker, ret[date], series.iloc[loc - 1], series.iloc[loc]))
    flags = pd.DataFrame(rows, columns=["date", "ticker", "return", "prev_price", "price"])
    return flags.sort_values("return", key=abs, ascending=False, ignore_index=True)


def sanity_report(
    prices: pd.DataFrame, max_abs_return: float = 0.5, volume: pd.DataFrame | None = None
) -> pd.DataFrame:
    """One row per ticker summarising data quality.

    Columns: first and last date with a price, number of prices, missing prices inside the
    ticker's life, longest such gap, longest run of unchanged prices (a sign of stale data),
    number of non-positive prices, the largest absolute daily return, and the number of daily
    moves beyond ``max_abs_return``. If a matching ``volume`` table is given, also the number
    of priced days with zero volume (the price did not come from a trade that day).
    """
    rows = {}
    for ticker in prices.columns:
        series = prices[ticker]
        first, last = series.first_valid_index(), series.last_valid_index()
        if first is None:
            rows[ticker] = {"n_prices": 0}
            continue
        life = series.loc[first:last]
        valid = life.dropna()
        ret = valid / valid.shift(1) - 1
        rows[ticker] = {
            "first_date": first,
            "last_date": last,
            "n_prices": len(valid),
            "missing_inside": int(life.isna().sum()),
            "longest_gap": _longest_run(life.isna()),
            "longest_flat_run": _longest_run(valid.diff() == 0),
            "n_nonpositive": int((valid <= 0).sum()),
            "max_abs_return": float(ret.abs().max()) if len(ret) > 1 else np.nan,
            "n_big_moves": int((ret.abs() > max_abs_return).sum()),
        }
        if volume is not None and ticker in volume:
            rows[ticker]["zero_volume_days"] = int((volume[ticker].reindex(valid.index) == 0).sum())
    report = pd.DataFrame.from_dict(rows, orient="index")
    report.index.name = "ticker"
    return report
