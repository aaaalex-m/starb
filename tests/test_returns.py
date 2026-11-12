import numpy as np
import pandas as pd
import pytest

from statarb import returns
from statarb.config import TRADING_DAYS


@pytest.fixture
def three_prices():
    return pd.Series([100.0, 110.0, 99.0], index=pd.bdate_range("2024-01-01", periods=3))


def test_simple_and_log_returns_known_values(three_prices):
    np.testing.assert_allclose(returns.simple_returns(three_prices), [0.10, -0.10])
    np.testing.assert_allclose(returns.log_returns(three_prices), np.log([1.1, 0.9]))


def test_first_row_dropped_and_index_aligned(gbm_prices):
    R = returns.simple_returns(gbm_prices)
    assert len(R) == len(gbm_prices) - 1
    assert R.index[0] == gbm_prices.index[1]
    assert not R.isna().any().any()


def test_log_returns_add_over_time(gbm_prices):
    r = returns.log_returns(gbm_prices)
    total = np.log(gbm_prices.iloc[-1] / gbm_prices.iloc[0])
    np.testing.assert_allclose(r.sum(), total)


def test_simple_returns_add_across_assets(gbm_prices):
    """A one-period portfolio return is the weighted sum of simple (not log) returns."""
    w = np.array([0.5, 0.3, 0.2])
    p0, p1 = gbm_prices.iloc[0].to_numpy(), gbm_prices.iloc[1].to_numpy()
    shares = w / p0  # one dollar split by weight
    portfolio_return = shares @ p1 - 1
    R = returns.simple_returns(gbm_prices).iloc[0].to_numpy()
    r = returns.log_returns(gbm_prices).iloc[0].to_numpy()
    assert portfolio_return == pytest.approx(w @ R, rel=1e-12)
    assert np.log1p(portfolio_return) != pytest.approx(w @ r, rel=1e-12)


def test_simple_log_conversion_round_trip(gbm_prices):
    R = returns.simple_returns(gbm_prices)
    pd.testing.assert_frame_equal(returns.log_to_simple(returns.simple_to_log(R)), R)
    pd.testing.assert_frame_equal(returns.simple_to_log(R), returns.log_returns(gbm_prices))


@pytest.mark.parametrize("kind", ["simple", "log"])
def test_cumulative_matches_price_ratio(gbm_prices, kind):
    rets = (
        returns.simple_returns(gbm_prices) if kind == "simple" else returns.log_returns(gbm_prices)
    )
    cum = returns.cumulative(rets, kind=kind)
    np.testing.assert_allclose(cum.iloc[-1], gbm_prices.iloc[-1] / gbm_prices.iloc[0] - 1)


def test_cumulative_rejects_unknown_kind(gbm_prices):
    with pytest.raises(ValueError, match="kind"):
        returns.cumulative(returns.simple_returns(gbm_prices), kind="arithmetic")


def test_annualise_scalars():
    ann = returns.annualise(0.0004, 0.01)
    assert ann.mean == pytest.approx(0.0004 * 252)
    assert ann.vol == pytest.approx(0.01 * np.sqrt(252))


def test_square_root_of_time_for_iid_returns():
    """Variance of a T-day sum of i.i.d. daily returns is T times the daily variance."""
    rng = np.random.default_rng(1)
    daily_vol = 0.01
    paths = rng.normal(0, daily_vol, size=(20_000, TRADING_DAYS))
    annual_vol = paths.sum(axis=1).std()
    assert annual_vol == pytest.approx(returns.annualise(0, daily_vol).vol, rel=0.02)


def test_rolling_volatility_constant_input():
    alternating = pd.Series([0.01, -0.01] * 50)
    vol = returns.rolling_volatility(alternating, window=20)
    assert vol.iloc[:19].isna().all()
    expected = np.std([0.01, -0.01] * 10, ddof=1) * np.sqrt(252)
    np.testing.assert_allclose(vol.iloc[19:], expected)


def test_summary_gaussian_vs_fat_tailed():
    rng = np.random.default_rng(2)
    n = 400_000
    df = pd.DataFrame(
        {
            "gauss": rng.normal(0, 0.01, n),
            # Student t with 10 degrees of freedom has excess kurtosis 6 / (10 - 4) = 1.
            "student_t10": rng.standard_t(10, n) * 0.01,
        }
    )
    s = returns.summary(df)
    assert s.loc["gauss", "excess_kurtosis"] == pytest.approx(0, abs=0.05)
    assert s.loc["student_t10", "excess_kurtosis"] == pytest.approx(1.0, abs=0.25)
    assert s.loc["gauss", "ann_vol"] == pytest.approx(0.01 * np.sqrt(252), rel=0.01)
    assert s.loc["gauss", "n_obs"] == n


def test_summary_accepts_series(gbm_prices):
    s = returns.summary(returns.simple_returns(gbm_prices["A"]))
    assert s.shape[0] == 1
