# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Module 3: Statistics toolkit — random walks, stationarity, regression
#
# Statistical arbitrage rests on one question: **is this spread stationary?** Before we can
# test that (Module 5), we need the vocabulary (random walk, AR(1), stationarity, unit root,
# autocorrelation), the workhorse estimator (ordinary least squares, OLS) with its error bars,
# and the main trap: **spurious regression**.
#
# Everything is simulated first, where the truth is known; real data appears only in
# Section 5.
#
# 1. Three basic processes and their autocorrelation functions
# 2. Ensembles: diffusion vs a particle in a trap
# 3. OLS by hand, hypothesis tests, and robust (HAC) error bars
# 4. Spurious regression
# 5. Real data: SPY returns and volatility clustering
# 6. Exercises
#
# Lecture notes, glossary and interview questions: [`lecture.md`](lecture.md).
# New code: `statarb.simulate`, `statarb.stats`.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sps
from statsmodels.tsa.stattools import pacf

from statarb import data, returns, simulate, stats

pd.set_option("display.precision", 4)
plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3})
rng = np.random.default_rng(2026)


def plot_acf(ax, rho: pd.Series, nobs: int, title: str) -> None:
    ax.bar(rho.index[1:], rho.iloc[1:], width=0.6)
    band = stats.acf_band(nobs)
    ax.axhspan(-band, band, color="C1", alpha=0.2, label="±1.96/√T")
    ax.set_ylim(-0.3, 1.05)
    ax.set_xlabel("lag k")
    ax.set_title(title)


# %% [markdown]
# ## 1. Three basic processes
#
# With i.i.d. shocks $\varepsilon_t \sim \mathcal N(0, \sigma^2)$:
#
# - **White noise:** $x_t = \varepsilon_t$. No memory at all.
# - **AR(1)** (autoregressive of order 1): $x_t = c + \phi x_{t-1} + \varepsilon_t$. Each
#   value remembers a fraction $\phi$ of the previous one. Stationary when $|\phi| < 1$.
# - **Random walk:** $x_t = x_{t-1} + \varepsilon_t$, the AR(1) with $\phi = 1$ (a **unit
#   root**). Shocks never decay.
#
# The **autocorrelation function** (ACF) $\rho_k = \operatorname{Corr}(x_t, x_{t-k})$ is
# the fingerprint of each. For AR(1), $\rho_k = \phi^k$. The shaded band, $\pm 1.96/\sqrt T$,
# is where 95% of sample autocorrelations of *white noise* fall.

# %%
n = 500
series = {
    "white noise": simulate.white_noise(n, seed=rng),
    "AR(1), φ = 0.9": simulate.ar1(n, 0.9, seed=rng),
    "random walk": simulate.random_walk(n, seed=rng),
}

fig, axes = plt.subplots(3, 2, figsize=(11, 7.5), gridspec_kw={"width_ratios": [2, 1]})
for (name, x), (ax_ts, ax_acf) in zip(series.items(), axes, strict=True):
    ax_ts.plot(x, lw=0.8)
    ax_ts.set_title(name)
    plot_acf(ax_acf, stats.acf(x, 40), n, f"ACF: {name}")
    if name.startswith("AR"):
        ax_acf.plot(np.arange(41), 0.9 ** np.arange(41), "k--", lw=1, label="φ^k")
    ax_acf.legend(fontsize=8)
fig.tight_layout()

# %% [markdown]
# - White noise: every autocorrelation is inside the band (about 1 in 20 pokes out by
#   chance, which is what a 95% band means).
# - AR(1): geometric decay, $0.9^k$, with a **correlation time** $\tau = -1/\ln\phi \approx
#   9.5$ steps. The sample ACF is noisy around the theoretical curve.
# - Random walk: the sample ACF starts near 1 and decays only slowly and roughly linearly.
#   That slow decay is the visual signature of a unit root. (The theoretical ACF of a
#   random walk isn't even defined, because its variance isn't constant.)
#
# The **partial autocorrelation function** (PACF) at lag $k$ is the correlation between
# $x_t$ and $x_{t-k}$ after removing the effect of the lags in between. For an AR(1), only
# lag 1 matters directly, so its PACF cuts off after lag 1:

# %%
ar = series["AR(1), φ = 0.9"]
pd.DataFrame({"ACF": stats.acf(ar, 5), "PACF": pacf(ar, nlags=5)}).T

# %% [markdown]
# ## 2. Ensembles: diffusion vs a particle in a trap
#
# One path can mislead. Simulate an **ensemble** of 2,000 independent paths and look at the
# variance across paths at each time, $\operatorname{Var}(x_t)$.
#
# For the random walk, $x_t = \sum_{s \le t} \varepsilon_s$, so $\operatorname{Var}(x_t) =
# t\sigma^2$, growing without bound: free diffusion, $\langle x^2\rangle = 2Dt$ with
# $2D = \sigma^2$. For the AR(1), the pull back towards the mean balances the shocks, and the
# variance saturates: a particle in a harmonic trap. (Here both start at 0, so the AR(1) needs
# a few correlation times to relax to equilibrium.)

# %%
n_paths, n_steps = 2000, 300
rw_ens = simulate.random_walk(n_steps, n_paths=n_paths, seed=rng)
ar_ens = simulate.ar1(n_steps, 0.9, x0=0.0, n_paths=n_paths, seed=rng)
t = np.arange(1, n_steps + 1)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
axes[0].plot(rw_ens[:, :30], lw=0.5, color="C0", alpha=0.6)
axes[0].plot(ar_ens[:, :30], lw=0.5, color="C1", alpha=0.6)
axes[0].set_title("30 sample paths: random walk (blue), AR(1) φ=0.9 (orange)")
axes[1].plot(t, rw_ens.var(axis=1), label="random walk")
axes[1].plot(t, t, "k--", lw=0.8, label=r"$t\sigma^2$")
axes[1].plot(t, ar_ens.var(axis=1), label="AR(1), φ = 0.9")
axes[1].set_xlabel("t")
axes[1].set_title("Variance across the ensemble")
axes[1].legend()
fig.tight_layout()

# %% [markdown]
# This is the distinction everything in stat arb hinges on:
#
# - A **stationary** series (AR(1) with $|\phi| < 1$) has a constant mean and variance and
#   keeps coming back. A spread like this can be traded: when it is far from its mean, it is
#   likely to return.
# - A **non-stationary** series (random walk) has no fixed level to return to. Its variance
#   grows with time; "far from the mean" means nothing.
#
# **Order of integration.** A series is $I(1)$ if it must be differenced once to become
# stationary, $I(0)$ if it is stationary already. Differencing a random walk gives back the
# white-noise shocks, so a random walk is $I(1)$. Log prices are approximately $I(1)$, and
# returns (their first differences) approximately $I(0)$.
#
# ## 3. Ordinary least squares
#
# ### 3a. By hand vs `statsmodels`
#
# Model: $y = X\beta + \varepsilon$, with $X$ an $n \times k$ matrix whose first column is
# ones (the intercept). OLS minimises the sum of squared residuals, which gives
#
# $$\hat\beta = (X^\top X)^{-1}X^\top y, \qquad \hat\sigma^2 = \frac{e^\top e}{n-k}, \qquad
# \widehat{\operatorname{Var}}(\hat\beta) = \hat\sigma^2 (X^\top X)^{-1},$$
#
# with residuals $e = y - X\hat\beta$. Simulate $y = 1 + 2x + \varepsilon$ and compute it
# all with NumPy:

# %%
n = 100
x = rng.normal(size=n)
y = 1.0 + 2.0 * x + rng.normal(0, 1.5, n)

X = np.column_stack([np.ones(n), x])
k = X.shape[1]
beta_hat = np.linalg.solve(X.T @ X, X.T @ y)
resid = y - X @ beta_hat
sigma2_hat = resid @ resid / (n - k)
se = np.sqrt(np.diag(sigma2_hat * np.linalg.inv(X.T @ X)))
t_stat = beta_hat / se
p_value = 2 * sps.t.sf(np.abs(t_stat), df=n - k)
r2 = 1 - resid @ resid / np.sum((y - y.mean()) ** 2)

by_hand = pd.DataFrame(
    {"coef": beta_hat, "std_err": se, "t": t_stat, "p_value": p_value}, index=["const", "x"]
)
print(f"R² = {r2:.4f}")
by_hand

# %%
print(sm.OLS(y, sm.add_constant(x)).fit().summary())

# %% [markdown]
# The numbers match. What each line of the summary means:
#
# | Item | Meaning |
# |---|---|
# | `coef` | $\hat\beta$: the fitted intercept and slope (true values 1 and 2) |
# | `std err` | Standard error: the estimated standard deviation of $\hat\beta$ across hypothetical repeated samples |
# | `t` | $t = \hat\beta / \text{SE}$: how many standard errors the estimate is from 0 |
# | `P>\|t\|` | p-value: probability of a $\lvert t\rvert$ at least this large *if the true coefficient were 0* |
# | `[0.025 0.975]` | 95% confidence interval, $\hat\beta \pm t_{0.975, n-k}\,\text{SE}$ |
# | `R-squared` | Fraction of the variance of $y$ explained by the fit |
# | `Adj. R-squared` | $R^2$ penalised for the number of regressors |
# | `F-statistic` | Test that all slopes are zero together |
# | `Log-Likelihood`, `AIC`, `BIC` | Gaussian likelihood of the fit and information criteria for comparing models (Module 10) |
# | `Durbin-Watson` | Checks residual autocorrelation: about 2 if none, near 0 if strongly positive |
# | `Omnibus`, `Jarque-Bera`, `Skew`, `Kurtosis` | Tests whether the residuals look Gaussian |
# | `Cond. No.` | Condition number of $X$: large values warn of nearly collinear regressors |
#
# `statarb.stats.ols` wraps this and returns just what the project needs:

# %%
fit = stats.ols(y, x)
print(f"R² = {fit.rsquared:.4f}, residual variance = {fit.sigma2:.3f}")
fit.table()

# %% [markdown]
# ### 3b. What a hypothesis test promises
#
# A test of $H_0: \beta = 0$ at **significance level** 5% rejects when $p < 0.05$. If the
# assumptions hold, then *when $H_0$ is true* it rejects 5% of the time: that is the **type I
# error** rate (false positive). When $H_0$ is false, the probability of rejecting is the
# **power**; failing to reject is a **type II error** (false negative). Check both by
# simulation:


# %%
def rejection_rate(beta: float, n: int, trials: int = 2000) -> float:
    rejections = 0
    for _ in range(trials):
        x = rng.normal(size=n)
        y = beta * x + rng.normal(size=n)
        rejections += stats.ols(y, x).pvalues["x"] < 0.05
    return rejections / trials


tests = pd.DataFrame(
    {
        "n = 50": [rejection_rate(0.0, 50), rejection_rate(0.2, 50)],
        "n = 500": [rejection_rate(0.0, 500), rejection_rate(0.2, 500)],
    },
    index=["true β = 0 (type I error rate)", "true β = 0.2 (power)"],
)
tests

# %% [markdown]
# The false-positive rate is about 5% whatever the sample size, as promised. Power grows
# with $n$: a small true effect is usually missed with 50 observations and almost always
# found with 500.
#
# ### 3c. When the error bars lie: autocorrelation and HAC
#
# The textbook standard error assumes errors that are uncorrelated and have constant
# variance (**homoskedastic**). Financial time series violate both: errors are often
# autocorrelated, and their variance changes over time (**heteroskedasticity**).
#
# Repeat the type I error experiment with a persistent regressor and persistent errors, both
# AR(1) with $\phi = 0.9$ and independent of each other, so the true slope is still 0:

# %%
trials, n = 1000, 300
naive, hac = 0, 0
for _ in range(trials):
    x = simulate.ar1(n, 0.9, seed=rng)
    y = simulate.ar1(n, 0.9, seed=rng)
    naive += stats.ols(y, x).pvalues["x"] < 0.05
    hac += stats.ols(y, x, cov_type="HAC").pvalues["x"] < 0.05
print(
    f"false-positive rate at a nominal 5%: textbook SE {naive / trials:.0%}, HAC SE {hac / trials:.0%}"
)

# %% [markdown]
# With autocorrelated data, 300 observations carry much less information than 300
# independent ones. The textbook formula doesn't know that, so its error bars are too small
# and it "finds" a relationship about half the time. **HAC** (heteroskedasticity- and
# autocorrelation-consistent, or Newey–West) standard errors add the covariance between
# nearby residuals back in, and cut the false-positive rate by more than half. They don't
# reach 5% here: the rule-of-thumb window (`stats.newey_west_lags(300)` = 5 lags) is short
# compared with the correlation time of about 10 steps, and HAC is only correct as
# $n \to \infty$. Persistent data needs long samples, and some humility about error bars.
#
# ## 4. Spurious regression
#
# Now the main trap. Take two *independent* random walks and regress one on the other.
# There is no relationship, so a 5% test should reject 5% of the time.

# %%
n, trials = 500, 1000
t_vals, r2_vals, t_hac = [], [], []
for _ in range(trials):
    x = simulate.random_walk(n, seed=rng)
    y = simulate.random_walk(n, seed=rng)
    fit = stats.ols(y, x)
    t_vals.append(fit.tvalues["x"])
    r2_vals.append(fit.rsquared)
    t_hac.append(stats.ols(y, x, cov_type="HAC").tvalues["x"])
t_vals, r2_vals, t_hac = map(np.asarray, (t_vals, r2_vals, t_hac))

fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
axes[0].plot(x, label="x")
axes[0].plot(y, label="y")
axes[0].set_title(f"One pair: R² = {r2_vals[-1]:.2f}, t = {t_vals[-1]:.1f}")
axes[0].legend()
grid = np.linspace(-40, 40, 400)
axes[1].hist(t_vals, bins=60, density=True, alpha=0.6, label="spurious regressions")
axes[1].plot(grid, sps.norm.pdf(grid), "C3", label="N(0,1): what the test assumes")
for side in (-1.96, 1.96):
    axes[1].axvline(side, color="k", lw=0.8)
axes[1].set_title("t-statistic of the slope")
axes[1].legend(fontsize=8)
axes[2].hist(r2_vals, bins=40)
axes[2].set_title("R²")
fig.tight_layout()

print(f"share with |t| > 1.96: {(np.abs(t_vals) > 1.96).mean():.0%} (should be 5%)")
print(f"with HAC errors:       {(np.abs(t_hac) > 1.96).mean():.0%}")
print(
    f"median |t| = {np.median(np.abs(t_vals)):.1f};  share with R² > 0.5: {(r2_vals > 0.5).mean():.0%}"
)

# %% [markdown]
# About 90% of regressions between unrelated random walks are "significant", with typical
# t-statistics around 10 and often a large $R^2$. Granger and Newbold (1974) showed this by
# simulation. Phillips (1986) proved why: when $y$ and $x$ are independent $I(1)$ series, the
# residual is itself a random walk, every OLS assumption about the errors fails, $\hat\beta$
# converges to a *random variable* instead of 0, and the t-statistic grows like $\sqrt T$.
# More data makes it worse:

# %%
by_length = {}
for T in (100, 500, 2000):
    tt = [
        stats.ols(simulate.random_walk(T, seed=rng), simulate.random_walk(T, seed=rng)).tvalues["x"]
        for _ in range(300)
    ]
    by_length[T] = {
        "share |t| > 1.96": np.mean(np.abs(tt) > 1.96),
        "median |t|": np.median(np.abs(tt)),
    }
pd.DataFrame(by_length).T.rename_axis("T")

# %% [markdown]
# HAC error bars help only a little: the problem is not the size of the error bars but that
# the model itself is wrong. *Physics analogy:* two independent diffusing particles. In any
# single run both wander far from the origin, and over a finite window their trajectories
# look correlated (both drifted up, say). The sample correlation of two independent random
# walks does not shrink to 0 as the window grows; its distribution stays broad.
#
# **Why this matters for stat arb:** regressing one price on another and trusting the
# t-statistic or $R^2$ is exactly this mistake. Two cures, coming next: difference the data
# (regress returns on returns, Exercise 2), or test whether the *residual* is stationary,
# which is cointegration (Module 5).
#
# ## 5. Real data: SPY returns and volatility clustering
#
# Twenty years of SPY daily data. Compare the ACF of the log price, of daily log returns, and
# of squared returns.

# %%
prices = data.load_prices("SPY", "2005-01-01", "2024-12-31")["SPY"]
log_price = np.log(prices)
r = returns.log_returns(prices)
T = len(r)

fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
plot_acf(axes[0], stats.acf(log_price, 60), T, "ACF: log price")
plot_acf(axes[1], stats.acf(r, 60), T, "ACF: daily log return")
plot_acf(axes[2], stats.acf(r**2, 60), T, "ACF: squared return")
axes[1].set_ylim(-0.15, 0.15)
axes[2].set_ylim(-0.15, 0.6)
fig.tight_layout()

summary = pd.DataFrame(
    {
        "log price": stats.acf(log_price, 5),
        "return": stats.acf(r, 5),
        "squared return": stats.acf(r**2, 5),
    }
).iloc[1:]
print(f"T = {T}, white-noise band ±{stats.acf_band(T):.3f}")
summary.T

# %% [markdown]
# - **Log price:** ACF near 1 at every lag shown, the unit-root signature. Prices are
#   approximately $I(1)$.
# - **Returns:** autocorrelations are small. Lag 1 is about −0.1, outside the band (a little
#   short-term reversal, which we met in Module 1 as monthly volatility below daily), but
#   there is almost no linear predictability. Returns are approximately $I(0)$.
# - **Squared returns:** large and persistent positive autocorrelation, lasting weeks to
#   months. Big moves follow big moves: **volatility clustering**. Returns are uncorrelated
#   but *not independent*; their variance is predictable. Models of this (ARCH and GARCH)
#   are beyond this project, but its consequences are everywhere: rolling volatility,
#   heteroskedastic regression errors, and the HAC correction above.
#
# One caution: the white-noise band assumes i.i.d. data. With volatility clustering, the
# sampling error of the return ACF is larger than $1/\sqrt T$, so a few bars just outside
# the band are weaker evidence than they look.
#
# ## 6. Exercises
#
# 1. **AR(1) stationary variance.** Derive $\operatorname{Var}(x_t) = \sigma^2/(1-\phi^2)$ for
#    a stationary AR(1), and check it on a simulated ensemble.
# 2. **Differencing cures spurious regression.** Regress the *differences* of two independent
#    random walks on each other (1,000 pairs) and show the t-statistics now follow the
#    nominal distribution, with about 5% beyond ±1.96.
# 3. **Estimating $\phi$ near a unit root.** For $\phi \in \{0.5, 0.9, 0.99, 1.0\}$ and 250
#    observations, estimate $\phi$ by regressing $x_t$ on $x_{t-1}$, over many simulations.
#    How biased and how noisy is $\hat\phi$ as $\phi \to 1$? Why does this matter for
#    deciding whether a spread is mean-reverting?

# %%
# Exercise 1


# %%
# Exercise 2


# %%
# Exercise 3
