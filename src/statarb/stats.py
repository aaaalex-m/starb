r"""Statistics toolkit: ordinary least squares and autocorrelation.

``ols`` is a thin wrapper around ``statsmodels`` that returns only the numbers this project
uses, with named coefficients, and makes the choice of standard errors explicit:
``cov_type="nonrobust"`` (the textbook formula, valid for uncorrelated, constant-variance
errors) or ``cov_type="HAC"`` (Newey-West: robust to heteroskedasticity and autocorrelation).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

ArrayLike = np.ndarray | pd.Series | pd.DataFrame


@dataclass(frozen=True)
class OLSResult:
    """Coefficients and inference from :func:`ols`. Series are indexed by regressor name."""

    params: pd.Series
    bse: pd.Series
    tvalues: pd.Series
    pvalues: pd.Series
    rsquared: float
    resid: np.ndarray
    sigma2: float
    nobs: int
    df_resid: int
    cov_type: str

    def table(self) -> pd.DataFrame:
        """Coefficient table: estimate, standard error, t-statistic, p-value."""
        return pd.DataFrame(
            {"coef": self.params, "std_err": self.bse, "t": self.tvalues, "p_value": self.pvalues}
        )


def newey_west_lags(nobs: int) -> int:
    r"""Rule-of-thumb HAC bandwidth :math:`\lfloor 4 (T/100)^{2/9} \rfloor` (Newey & West, 1994)."""
    return int(np.floor(4 * (nobs / 100) ** (2 / 9)))


def ols(
    y: ArrayLike,
    X: ArrayLike,
    add_constant: bool = True,
    cov_type: str = "nonrobust",
    maxlags: int | None = None,
) -> OLSResult:
    r"""Regress ``y`` on the columns of ``X`` by ordinary least squares.

    Estimates :math:`\hat\beta = (X^\top X)^{-1} X^\top y`. With ``add_constant`` an intercept
    named ``const`` is prepended. ``cov_type`` is ``"nonrobust"`` or ``"HAC"``; for HAC,
    ``maxlags`` defaults to :func:`newey_west_lags`.
    """
    y = np.asarray(y, dtype=float)
    if isinstance(X, pd.Series):
        X = X.to_frame()
    if isinstance(X, pd.DataFrame):
        exog = X.astype(float)
    else:
        X = np.asarray(X, dtype=float)
        X = X[:, None] if X.ndim == 1 else X
        names = ["x"] if X.shape[1] == 1 else [f"x{i + 1}" for i in range(X.shape[1])]
        exog = pd.DataFrame(X, columns=names)
    if add_constant:
        exog = sm.add_constant(exog, has_constant="add")
    exog = exog.reset_index(drop=True)

    if cov_type == "nonrobust":
        fit = sm.OLS(y, exog).fit()
    elif cov_type == "HAC":
        lags = newey_west_lags(len(y)) if maxlags is None else maxlags
        fit = sm.OLS(y, exog).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    else:
        raise ValueError(f"cov_type must be 'nonrobust' or 'HAC', got {cov_type!r}")

    return OLSResult(
        params=pd.Series(fit.params, index=exog.columns),
        bse=pd.Series(fit.bse, index=exog.columns),
        tvalues=pd.Series(fit.tvalues, index=exog.columns),
        pvalues=pd.Series(fit.pvalues, index=exog.columns),
        rsquared=float(fit.rsquared),
        resid=np.asarray(fit.resid),
        sigma2=float(fit.scale),
        nobs=int(fit.nobs),
        df_resid=int(fit.df_resid),
        cov_type=cov_type,
    )


def acf(x: ArrayLike, nlags: int = 20) -> pd.Series:
    r"""Sample autocorrelation function :math:`\hat\rho_k` for lags :math:`0, \dots, n_{lags}`.

    .. math:: \hat\rho_k = \frac{\sum_{t=k+1}^{T} (x_t - \bar x)(x_{t-k} - \bar x)}
                              {\sum_{t=1}^{T} (x_t - \bar x)^2}

    This is the standard (biased, always positive semi-definite) estimator used by
    ``statsmodels``. NaNs are dropped first. For white noise each :math:`\hat\rho_k`
    (:math:`k \ge 1`) is roughly :math:`\mathcal N(0, 1/T)`, hence the usual
    :math:`\pm 1.96/\sqrt T` band.
    """
    values = np.asarray(pd.Series(np.ravel(x)).dropna(), dtype=float)
    if nlags >= len(values):
        raise ValueError(f"nlags={nlags} needs more than {len(values)} observations")
    dev = values - values.mean()
    denom = dev @ dev
    rho = [1.0] + [dev[k:] @ dev[:-k] / denom for k in range(1, nlags + 1)]
    return pd.Series(rho, index=pd.RangeIndex(nlags + 1, name="lag"), name="acf")


def acf_band(nobs: int, level: float = 0.95) -> float:
    """Half-width of the white-noise confidence band for a sample ACF (1.96/sqrt(T) at 95%)."""
    return float(norm.ppf(0.5 + level / 2) / np.sqrt(nobs))
