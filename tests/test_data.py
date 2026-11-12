import numpy as np
import pandas as pd
import pytest

from statarb import data


def test_first_load_downloads_second_reads_cache(tmp_path, downloader):
    first = data.load_bars(
        "AAA", "2020-01-01", "2020-06-30", cache_dir=tmp_path, downloader=downloader
    )
    second = data.load_bars(
        "AAA", "2020-01-01", "2020-06-30", cache_dir=tmp_path, downloader=downloader
    )
    assert len(downloader.calls) == 1
    pd.testing.assert_frame_equal(first, second, check_freq=False)
    assert data.cache_path("AAA", tmp_path).exists()


def test_sub_range_is_served_from_cache(tmp_path, downloader):
    data.load_bars("AAA", "2020-01-01", "2020-12-31", cache_dir=tmp_path, downloader=downloader)
    sub = data.load_bars(
        "AAA", "2020-03-01", "2020-03-31", cache_dir=tmp_path, downloader=downloader
    )
    assert len(downloader.calls) == 1
    assert sub.index.min() >= pd.Timestamp("2020-03-01")
    assert sub.index.max() <= pd.Timestamp("2020-03-31")


def test_wider_range_redownloads_union_and_cache_grows(tmp_path, downloader):
    data.load_bars("AAA", "2020-03-01", "2020-06-30", cache_dir=tmp_path, downloader=downloader)
    data.load_bars("AAA", "2020-01-01", "2020-04-30", cache_dir=tmp_path, downloader=downloader)
    assert len(downloader.calls) == 2
    _, start, end = downloader.calls[-1]
    assert (start, end) == (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-06-30"))
    info = data.cache_info("AAA", tmp_path)
    assert (info["start"], info["end"]) == ("2020-01-01", "2020-06-30")


def test_cache_records_download_time_and_settings(tmp_path, downloader):
    data.load_bars("AAA", "2020-01-01", "2020-01-31", cache_dir=tmp_path, downloader=downloader)
    info = data.cache_info("AAA", tmp_path)
    assert info["ticker"] == "AAA"
    assert info["auto_adjust"] is False
    assert pd.Timestamp(info["downloaded_at"]).tzinfo is not None


def test_no_cache_always_downloads_and_writes_nothing(tmp_path, downloader):
    for _ in range(2):
        data.load_bars(
            "AAA",
            "2020-01-01",
            "2020-01-31",
            cache=False,
            cache_dir=tmp_path,
            downloader=downloader,
        )
    assert len(downloader.calls) == 2
    assert not any(tmp_path.iterdir())


def test_load_prices_is_wide_and_selects_field(tmp_path, downloader):
    tickers = ["AAA", "BBB", "CCC"]
    prices = data.load_prices(
        tickers, "2020-01-01", "2020-03-31", cache_dir=tmp_path, downloader=downloader
    )
    assert list(prices.columns) == tickers
    bars = data.load_bars(
        "BBB", "2020-01-01", "2020-03-31", cache_dir=tmp_path, downloader=downloader
    )
    np.testing.assert_allclose(prices["BBB"], bars["adj_close"])
    close = data.load_prices(
        "BBB", "2020-01-01", "2020-03-31", field="close", cache_dir=tmp_path, downloader=downloader
    )
    np.testing.assert_allclose(close["BBB"], bars["close"])


def test_load_prices_rejects_unknown_field(tmp_path, downloader):
    with pytest.raises(ValueError, match="field"):
        data.load_prices("AAA", field="Adj Close", cache_dir=tmp_path, downloader=downloader)


def test_start_after_end_raises(tmp_path, downloader):
    with pytest.raises(ValueError, match="after"):
        data.load_bars("AAA", "2021-01-01", "2020-01-01", cache_dir=tmp_path, downloader=downloader)


def test_standardise_bars_renames_and_strips_timezone():
    idx = pd.DatetimeIndex(["2020-08-28 00:00", "2020-08-31 00:00"], tz="America/New_York")
    raw = pd.DataFrame(
        {
            "Open": [1.0, 2.0],
            "High": [1.0, 2.0],
            "Low": [1.0, 2.0],
            "Close": [1.0, 2.0],
            "Adj Close": [0.9, 1.8],
            "Volume": [10, 20],
            "Dividends": [0.0, 0.0],
            "Stock Splits": [0.0, 4.0],
        },
        index=idx,
    )
    bars = data.standardise_bars(raw)
    assert list(bars.columns) == data.BAR_COLUMNS
    assert bars.index.tz is None
    assert bars.index.name == "date"
    assert bars.loc["2020-08-31", "splits"] == 4.0


def test_unadjust_splits_restores_as_traded_price():
    dates = pd.bdate_range("2020-08-26", periods=5)  # Wed .. Tue; split on Mon 2020-08-31
    traded = pd.Series([500.0, 504.0, 508.0, 128.0, 130.0], index=dates)
    adjusted = traded / pd.Series([4, 4, 4, 1, 1], index=dates)
    splits = pd.Series([0, 0, 0, 4.0, 0], index=dates)
    pd.testing.assert_series_equal(data.unadjust_splits(adjusted, splits), traded)


def test_unadjust_splits_compounds_multiple_splits():
    dates = pd.bdate_range("2020-01-01", periods=4)
    adjusted = pd.Series(10.0, index=dates)
    splits = pd.Series([0, 2.0, 0, 3.0], index=dates)
    np.testing.assert_allclose(data.unadjust_splits(adjusted, splits), [60, 30, 30, 10])


@pytest.mark.network
def test_yahoo_download_smoke(tmp_path):
    prices = data.load_prices(["SPY"], "2024-01-02", "2024-01-31", cache_dir=tmp_path)
    assert prices["SPY"].notna().all()
    assert 15 <= len(prices) <= 22
