r"""Simple and log returns, compounding, annualisation and summary statistics.

Notation: price :math:`P_t`; simple return :math:`R_t = P_t/P_{t-1} - 1`; log return
:math:`r_t = \ln(P_t/P_{t-1}) = \ln(1 + R_t)`.

* Log returns add over **time**: :math:`\sum_t r_t = \ln(P_T/P_0)`.
* Simple returns add across **assets**: a portfolio with weights :math:`w_i` earns
  :math:`\sum_i w_i R_{i,t}`.

All functions take pandas Series (one asset) or DataFrames (one column per asset). Returns
are decimals; the first row, which has no previous price, is dropped.
"""

from typing import NamedTuple

import numpy as np
import pandas as pd

from statarb.config import TRADING_DAYS

PandasData = pd.Series | pd.DataFrame


def simple_returns(prices: PandasData) -> PandasData:
    """:math:`R_t = P_t / P_{t-1} - 1`."""
    return (prices / prices.shift(1) - 1).iloc[1:]


def log_returns(prices: PandasData) -> PandasData:
    r""":math:`r_t = \ln(P_t / P_{t-1})`."""
    return np.log(prices / prices.shift(1)).iloc[1:]


def simple_to_log(simple: PandasData) -> PandasData:
    r""":math:`r = \ln(1 + R)`."""
    return np.log1p(simple)


def log_to_simple(log: PandasData) -> PandasData:
    """:math:`R = e^r - 1`."""
    return np.expm1(log)


def cumulative(returns: PandasData, kind: str = "simple") -> PandasData:
    """Cumulative simple return since the start, :math:`P_t/P_0 - 1`.

    ``kind`` says whether ``returns`` are ``"simple"`` (compounded with a product) or
    ``"log"`` (summed, then exponentiated). Both give the same answer for the same prices.
    Add 1 to get the growth of one dollar (the "equity curve").
    """
    if kind == "simple":
        return (1 + returns).cumprod() - 1
    if kind == "log":
        return np.expm1(returns.cumsum())
    raise ValueError(f"kind must be 'simple' or 'log', got {kind!r}")


class Annualised(NamedTuple):
    mean: float | pd.Series
    vol: float | pd.Series


def annualise(
    mean: float | pd.Series, vol: float | pd.Series, periods_per_year: int = TRADING_DAYS
) -> Annualised:
    r"""Scale a per-period mean and volatility to annual units.

    The mean scales with time, :math:`\mu_{\text{ann}} = N\mu`, and the volatility with
    its square root, :math:`\sigma_{\text{ann}} = \sqrt{N}\sigma`. The square-root rule
    assumes returns are uncorrelated from one period to the next (variances then add, as in
    diffusion). The mean rule is exact for log returns and an approximation for simple ones.
    """
    return Annualised(mean * periods_per_year, vol * np.sqrt(periods_per_year))


def rolling_volatility(
    returns: PandasData, window: int = 21, periods_per_year: int = TRADING_DAYS
) -> PandasData:
    """Annualised rolling standard deviation over ``window`` periods (21 days ~ one month)."""
    return returns.rolling(window).std() * np.sqrt(periods_per_year)


def summary(returns: PandasData, periods_per_year: int = TRADING_DAYS) -> pd.DataFrame:
    """Per-asset table: annualised mean and volatility, skewness, excess kurtosis, extremes.

    Excess kurtosis is kurtosis minus 3, so it is 0 for a Gaussian; positive values mean fat
    tails. Pandas' ``kurt`` already returns the excess (bias-corrected) value.
    """
    frame = returns.to_frame() if isinstance(returns, pd.Series) else returns
    ann = annualise(frame.mean(), frame.std(), periods_per_year)
    return pd.DataFrame(
        {
            "ann_mean": ann.mean,
            "ann_vol": ann.vol,
            "skew": frame.skew(),
            "excess_kurtosis": frame.kurt(),
            "min": frame.min(),
            "max": frame.max(),
            "n_obs": frame.count(),
        }
    )
