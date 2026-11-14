import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps
from statsmodels.tsa.stattools import acf as sm_acf

from statarb import simulate, stats


@pytest.fixture
def linear_data():
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    y = 1.0 + 2.0 * x + rng.normal(0, 0.5, 500)
    return x, y


def test_ols_matches_textbook_formulas(linear_data):
    x, y = linear_data
    res = stats.ols(y, x)
    X = np.column_stack([np.ones_like(x), x])
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    resid = y - X @ beta
    n, k = X.shape
    sigma2 = resid @ resid / (n - k)
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    t = beta / se
    p = 2 * sps.t.sf(np.abs(t), df=n - k)
    r2 = 1 - resid @ resid / np.sum((y - y.mean()) ** 2)

    np.testing.assert_allclose(res.params, beta)
    np.testing.assert_allclose(res.bse, se)
    np.testing.assert_allclose(res.tvalues, t)
    np.testing.assert_allclose(res.pvalues, p, atol=1e-12)
    assert res.rsquared == pytest.approx(r2)
    assert res.sigma2 == pytest.approx(sigma2)
    assert (res.nobs, res.df_resid) == (n, n - k)
    np.testing.assert_allclose(res.resid, resid)


def test_ols_recovers_true_coefficients(linear_data):
    x, y = linear_data
    res = stats.ols(y, x)
    assert list(res.params.index) == ["const", "x"]
    assert abs(res.params["const"] - 1.0) < 3 * res.bse["const"]
    assert abs(res.params["x"] - 2.0) < 3 * res.bse["x"]
    assert list(res.table().columns) == ["coef", "std_err", "t", "p_value"]


def test_ols_names_and_no_constant():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, 2))
    y = X @ [1.0, -1.0] + rng.normal(0, 0.1, 100)
    assert list(stats.ols(y, X).params.index) == ["const", "x1", "x2"]
    frame = pd.DataFrame(X, columns=["mkt", "sector"], index=pd.bdate_range("2020", periods=100))
    assert list(stats.ols(pd.Series(y), frame, add_constant=False).params.index) == [
        "mkt",
        "sector",
    ]


def test_hac_widens_error_bars_with_autocorrelated_errors():
    """With persistent regressor and errors, naive standard errors are too small."""
    naive_reject, hac_reject = 0, 0
    for seed in range(200):
        rng = np.random.default_rng(seed)
        x = simulate.ar1(300, 0.9, seed=rng)
        y = simulate.ar1(300, 0.9, seed=rng)  # true slope is zero
        naive_reject += stats.ols(y, x).pvalues["x"] < 0.05
        hac_reject += stats.ols(y, x, cov_type="HAC").pvalues["x"] < 0.05
    assert naive_reject / 200 > 0.25  # far above the nominal 5%
    assert hac_reject < naive_reject


def test_ols_rejects_unknown_cov_type(linear_data):
    with pytest.raises(ValueError, match="cov_type"):
        stats.ols(*linear_data[::-1], cov_type="robust")


def test_newey_west_lags():
    assert stats.newey_west_lags(100) == 4
    assert stats.newey_west_lags(1000) == 6


def test_acf_matches_statsmodels():
    x = simulate.ar1(1000, 0.6, seed=5)
    np.testing.assert_allclose(stats.acf(x, 15), sm_acf(x, nlags=15))
    assert stats.acf(x, 5).index.name == "lag"


def test_acf_of_ar1_decays_geometrically():
    phi = 0.8
    rho = stats.acf(simulate.ar1(200_000, phi, seed=8), nlags=5)
    np.testing.assert_allclose(rho, phi ** np.arange(6), atol=0.01)


def test_white_noise_acf_mostly_inside_band():
    n = 2000
    rho = stats.acf(simulate.white_noise(n, seed=3), nlags=100).iloc[1:]
    inside = (rho.abs() < stats.acf_band(n)).mean()
    assert inside > 0.9
    assert stats.acf_band(100) == pytest.approx(0.196, abs=1e-3)


def test_acf_drops_nans_and_validates_nlags():
    x = pd.Series([1.0, 2.0, np.nan, 3.0, 2.0, 1.0])
    assert len(stats.acf(x, 2)) == 3
    with pytest.raises(ValueError, match="nlags"):
        stats.acf([1.0, 2.0, 3.0], nlags=3)
