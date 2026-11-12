import numpy as np
import pandas as pd
import pytest

from statarb import checks


@pytest.fixture
def walk():
    rng = np.random.default_rng(3)
    index = pd.bdate_range("2020-01-01", periods=300)
    return pd.Series(100 + rng.normal(0, 1, len(index)).cumsum(), index=index)


CAUSAL = {
    "rolling mean": lambda x: x.rolling(20).mean(),
    "rolling z-score": lambda x: (x - x.rolling(20).mean()) / x.rolling(20).std(),
    "expanding mean": lambda x: x.expanding().mean(),
    "lagged return": lambda x: x.pct_change().shift(1),
    "forward fill": lambda x: x.where(x.index.day_of_week != 2).ffill(),
}

LEAKY = {
    "full-sample z-score": lambda x: (x - x.mean()) / x.std(),
    "centred rolling mean": lambda x: x.rolling(21, center=True).mean(),
    "next-day return": lambda x: x.pct_change().shift(-1),
    "back fill": lambda x: x.where(x.index.day_of_week != 2).bfill(),
    "interpolation": lambda x: x.where(x.index.day_of_week != 2).interpolate(),
}


@pytest.mark.parametrize("name", CAUSAL)
def test_causal_signals_pass(walk, name):
    assert checks.lookahead_violations(CAUSAL[name], walk) == []


@pytest.mark.parametrize("name", LEAKY)
def test_leaky_signals_are_caught(walk, name):
    assert checks.lookahead_violations(LEAKY[name], walk, n_checks=20)


def test_works_on_dataframes(walk):
    frame = pd.DataFrame({"a": walk, "b": walk * 2})
    assert checks.lookahead_violations(lambda df: df.rolling(5).mean(), frame) == []
    assert checks.lookahead_violations(lambda df: df - df.mean(), frame)


def test_assert_no_lookahead_raises(walk):
    checks.assert_no_lookahead(CAUSAL["rolling mean"], walk)
    with pytest.raises(AssertionError, match="past values"):
        checks.assert_no_lookahead(LEAKY["full-sample z-score"], walk)
