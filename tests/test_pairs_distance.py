import numpy as np
import pandas as pd
import pytest

from statarb.pairs import distance


def make_prices(n: int = 1000, seed: int = 0, phi: float = 0.9, spread_sd: float = 0.01):
    """A and B share a random-walk trend plus a mean-reverting (AR(1)) log spread; C is an
    independent random walk; D is A's trend plus independent noise of its own."""
    rng = np.random.default_rng(seed)
    trend = np.cumsum(rng.normal(0.0003, 0.01, n))
    u = np.zeros(n)
    for t in range(1, n):
        u[t] = phi * u[t - 1] + rng.normal(0, spread_sd)
    other = np.cumsum(rng.normal(0.0003, 0.01, n))
    drift = np.cumsum(rng.normal(0, 0.01, n))
    index = pd.bdate_range("2010-01-01", periods=n)
    return pd.DataFrame(
        {
            "A": 100 * np.exp(trend + u / 2),
            "B": 50 * np.exp(trend - u / 2),
            "C": 80 * np.exp(other),
            "D": 30 * np.exp(trend + drift),
        },
        index=index,
    )


def test_normalise_starts_at_one_and_keeps_ratios():
    prices = make_prices(50)
    norm = distance.normalise(prices)
    np.testing.assert_allclose(norm.iloc[0], 1.0)
    np.testing.assert_allclose(norm["A"] / norm["A"].shift(), prices["A"] / prices["A"].shift())
    series = pd.Series([np.nan, 4.0, 8.0])
    np.testing.assert_allclose(distance.normalise(series), [np.nan, 1.0, 2.0])


def test_ssd_matrix_values_and_symmetry():
    norm = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 1.0, 1.0], "z": [1.0, 2.0, 4.0]})
    ssd = distance.ssd_matrix(norm)
    assert ssd.loc["x", "y"] == pytest.approx(0 + 1 + 4)
    assert ssd.loc["x", "z"] == pytest.approx(1)
    pd.testing.assert_frame_equal(ssd, ssd.T)
    assert np.isnan(np.diag(ssd)).all()
    with pytest.raises(ValueError, match="NaN"):
        distance.ssd_matrix(norm.where(norm > 1.5))


def test_select_pairs_finds_the_true_pair_and_drops_incomplete_tickers():
    prices = make_prices(300)
    prices["E"] = prices["A"] * 1.001
    prices.loc[prices.index[5], "E"] = np.nan  # incomplete: must be dropped
    pairs = distance.select_pairs(prices, n=2)
    assert (pairs.loc[0, "a"], pairs.loc[0, "b"]) == ("A", "B")
    assert "E" not in set(pairs["a"]) | set(pairs["b"])
    assert pairs["ssd"].is_monotonic_increasing
    assert len(distance.select_pairs(prices, n=100)) == 6  # 4 complete tickers -> 6 pairs


def test_spread_is_difference_of_normalised_prices():
    prices = make_prices(20)
    s = distance.spread(prices, "A", "B")
    expected = prices["A"] / prices["A"].iloc[0] - prices["B"] / prices["B"].iloc[0]
    np.testing.assert_allclose(s, expected)
    assert s.iloc[0] == 0


def test_pair_returns_timing_and_sign():
    index = pd.bdate_range("2020-01-01", periods=5)
    position = pd.Series([0.0, 1.0, 1.0, -1.0, 0.0], index=index)
    ra = pd.Series([0.0, 0.01, 0.02, 0.03, 0.04], index=index)
    rb = pd.Series([0.0, 0.0, 0.01, 0.01, 0.01], index=index)
    diff = ra - rb
    np.testing.assert_allclose(
        distance.pair_returns(position, ra, rb, lag=1),
        [0, 0, diff.iloc[2], diff.iloc[3], -diff.iloc[4]],
    )
    np.testing.assert_allclose(
        distance.pair_returns(position, ra, rb, lag=2), [0, 0, 0, diff.iloc[3], diff.iloc[4]]
    )


@pytest.fixture(scope="module")
def result():
    return distance.ggr_backtest(make_prices(1000, seed=1), formation=200, trading=100, n_pairs=1)


def test_backtest_selects_the_cointegrated_pair_and_makes_money(result):
    assert set(zip(result.pairs["a"], result.pairs["b"], strict=True)) == {("A", "B")}
    assert len(result.trades) > 5
    assert result.committed.mean() > 0
    assert (result.trades["return"] > 0).mean() > 0.7


def test_committed_and_employed_returns_are_consistent(result):
    # with a single pair, employed = committed on days with an open position, else 0
    open_days = result.open_pairs > 0
    np.testing.assert_allclose(result.employed[open_days], result.committed[open_days])
    assert (result.employed[~open_days] == 0).all()
    assert result.trades["days"].sum() == open_days.sum()


def test_trade_returns_compound_the_daily_returns(result):
    trade = result.trades.iloc[0]
    held = result.committed.loc[result.committed.index > trade["open"]].iloc[: trade["days"]]
    assert trade["return"] == pytest.approx((1 + held).prod() - 1)


def test_waiting_one_day_shifts_every_trade(result):
    waited = distance.ggr_backtest(
        make_prices(1000, seed=1), formation=200, trading=100, n_pairs=1, lag=2
    )
    pd.testing.assert_frame_equal(
        waited.trades[["open", "close", "direction"]], result.trades[["open", "close", "direction"]]
    )
    assert not np.allclose(waited.committed, result.committed)


def test_backtest_has_no_look_ahead():
    """Changing prices after a date must not change any return before it."""
    prices = make_prices(1000, seed=2)
    cut = prices.index[650]
    perturbed = prices.copy()
    rng = np.random.default_rng(3)
    after = perturbed.index >= cut
    perturbed.loc[after] *= np.exp(np.cumsum(rng.normal(0, 0.02, (after.sum(), 4)), axis=0))
    kwargs = {"formation": 200, "trading": 100, "n_pairs": 3}
    base = distance.ggr_backtest(prices, **kwargs).committed
    moved = distance.ggr_backtest(perturbed, **kwargs).committed
    pd.testing.assert_series_equal(base.loc[base.index < cut], moved.loc[moved.index < cut])
    assert not np.allclose(base.loc[base.index > cut], moved.loc[moved.index > cut])
