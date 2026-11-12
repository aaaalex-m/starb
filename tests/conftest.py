import zlib

import numpy as np
import pandas as pd
import pytest


def synthetic_bars(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Deterministic fake bars: geometric Brownian motion on weekdays, seeded by the ticker.

    Prices are generated from a fixed origin date so that overlapping requests agree, as real
    data would.
    """
    origin = pd.Timestamp("2000-01-03")
    dates = pd.bdate_range(origin, end)
    rng = np.random.default_rng(zlib.crc32(ticker.encode()))
    log_ret = rng.normal(0.0003, 0.01, len(dates))
    close = 100 * np.exp(np.cumsum(log_ret))
    bars = pd.DataFrame(
        {
            "open": close * (1 + rng.normal(0, 0.002, len(dates))),
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "adj_close": close * 0.9,
            "volume": rng.integers(1_000_000, 5_000_000, len(dates)).astype(float),
            "dividends": 0.0,
            "splits": 0.0,
        },
        index=pd.DatetimeIndex(dates, name="date"),
    )
    return bars.loc[start:end]


class CountingDownloader:
    """Wraps ``synthetic_bars`` and records every call, to check what the cache does."""

    __name__ = "synthetic"

    def __init__(self):
        self.calls: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []

    def __call__(self, ticker, start, end):
        self.calls.append((ticker, start, end))
        return synthetic_bars(ticker, start, end)


@pytest.fixture
def downloader():
    return CountingDownloader()


@pytest.fixture
def gbm_prices():
    """Three synthetic price series, dates x tickers."""
    dates = pd.bdate_range("2020-01-01", periods=500)
    rng = np.random.default_rng(0)
    log_ret = rng.normal(0.0002, 0.01, size=(len(dates), 3))
    return pd.DataFrame(100 * np.exp(np.cumsum(log_ret, axis=0)), index=dates, columns=list("ABC"))
