# Module 3 — Statistics toolkit: random walks, stationarity, regression

**Why this module:** stat arb rests on one question, *is this spread stationary?* This module builds the vocabulary to ask it (random walk, AR(1), stationarity, unit root, autocorrelation), the main estimator (ordinary least squares) with honest error bars, and the main trap: regressing one random walk on another gives "significant" nonsense.

Notebook: [`notebook.ipynb`](notebook.ipynb). Code: [`statarb/simulate.py`](../../src/statarb/simulate.py), [`statarb/stats.py`](../../src/statarb/stats.py).

---

## 1. Three building blocks

Let $\varepsilon_t$ be i.i.d. shocks with mean 0 and variance $\sigma^2$.

| Process | Definition | Memory | Physics picture |
|---|---|---|---|
| **White noise** | $x_t = \varepsilon_t$ | none | thermal noise |
| **AR(1)** | $x_t = c + \phi x_{t-1} + \varepsilon_t$ | decays as $\phi^k$ | particle in a harmonic trap (discrete Langevin equation) |
| **Random walk** | $x_t = x_{t-1} + \mu + \varepsilon_t$ | permanent | free diffusion |

The constant $\mu$ in the random walk is the **drift**: a steady trend added each step.

**Random walk.** Summing the shocks, $x_t = x_0 + \mu t + \sum_{s=1}^{t}\varepsilon_s$, so

```math
\mathbb E[x_t] = x_0 + \mu t, \qquad \operatorname{Var}(x_t) = t\sigma^2 .
```

The variance grows without bound, like $\langle x^2\rangle = 2Dt$ for a diffusing particle. There is no level the series returns to.

**AR(1).** For $|\phi| < 1$ the pull towards the mean balances the shocks and the process settles into an equilibrium distribution with

```math
\mathbb E[x_t] = \frac{c}{1-\phi}, \qquad \operatorname{Var}(x_t) = \frac{\sigma^2}{1-\phi^2}, \qquad \rho_k = \phi^{k}.
```

(Deriving the variance is Exercise 1.) The autocorrelation decays exponentially with **correlation time** $\tau = -1/\ln\phi$, about 9.5 steps for $\phi = 0.9$. Module 6 shows that AR(1) is the exact discretisation of the Ornstein–Uhlenbeck process, the overdamped Langevin equation $dX = -\theta X\,dt + \sigma\,dW$ with $\phi = e^{-\theta\,\Delta t}$.

At $\phi = 1$ the restoring force vanishes and AR(1) becomes a random walk. That boundary case is called a **unit root**, because the characteristic equation $1 - \phi z = 0$ has its root at $z = 1$.

## 2. Stationarity and integration

A process is **strictly stationary** if its joint distributions don't change when shifted in time. In practice we use **weak (covariance) stationarity**:

```math
\mathbb E[x_t] = \mu, \qquad \operatorname{Var}(x_t) = \gamma_0, \qquad \operatorname{Cov}(x_t, x_{t-k}) = \gamma_k \quad\text{(depends only on the lag } k\text{)} .
```

*Physics analogy:* a system in equilibrium vs one still diffusing away. White noise and AR(1) with $|\phi|<1$ are stationary; a random walk is not, because its variance depends on $t$.

**Order of integration.** $x$ is $I(d)$ if it must be differenced $d$ times to become stationary. Stationary series are $I(0)$. A random walk is $I(1)$: its differences $\Delta x_t = x_t - x_{t-1} = \mu + \varepsilon_t$ are white noise. Empirically, **log prices are roughly $I(1)$ and returns roughly $I(0)$**.

**Why it matters for trading.** "The spread is two standard deviations above its mean, so it will probably come back" only makes sense if the spread *has* a stable mean and standard deviation, i.e. is stationary. Most prices aren't; the hope of stat arb is that some *combinations* of prices are (cointegration, Module 5).

## 3. Autocorrelation

The **autocorrelation function** (ACF) is $\rho_k = \gamma_k/\gamma_0$, the correlation of the series with itself $k$ steps earlier. Its sample estimate is

```math
\hat\rho_k = \frac{\sum_{t=k+1}^{T}(x_t-\bar x)(x_{t-k}-\bar x)}{\sum_{t=1}^{T}(x_t-\bar x)^2}.
```

For i.i.d. data each $\hat\rho_k$ is roughly $\mathcal N(0, 1/T)$, which gives the usual **$\pm 1.96/\sqrt T$ band**. About 1 lag in 20 falls outside it by chance.

The **partial autocorrelation function** (PACF) at lag $k$ is the correlation between $x_t$ and $x_{t-k}$ after removing the linear effect of the lags in between. An AR(1)'s PACF cuts off after lag 1, while its ACF decays slowly.

*Physics analogy:* the time-correlation function $\langle A(t)A(0)\rangle$ of statistical mechanics.

What the ACFs look like (sketch them from memory):

- **white noise:** a spike of 1 at lag 0, then nothing;
- **AR(1):** geometric decay $\phi^k$;
- **random walk:** sample ACF near 1, decaying slowly and almost linearly (in a finite sample);
- **SPY returns:** almost nothing (lag 1 ≈ −0.10);
- **SPY squared returns:** 0.25–0.5 at short lags, still positive after two months.

That last pair is **volatility clustering**: returns are nearly uncorrelated but not independent, because their *variance* is predictable. Time-varying variance is called **heteroskedasticity** (constant variance: **homoskedasticity**).

## 4. Ordinary least squares (OLS)

**Model:** $y = X\beta + \varepsilon$, with $X$ an $n\times k$ matrix of **regressors** (first column all ones for the intercept). OLS chooses $\hat\beta$ to minimise the sum of squared **residuals** $e = y - X\hat\beta$. Setting the gradient to zero gives the normal equations $X^\top X\hat\beta = X^\top y$, so

```math
\hat\beta = (X^\top X)^{-1}X^\top y, \qquad \hat\sigma^2 = \frac{e^\top e}{n-k}, \qquad \widehat{\operatorname{Var}}(\hat\beta) = \hat\sigma^2(X^\top X)^{-1}.
```

*Physics analogy:* a $\chi^2$ fit with equal, uncorrelated error bars. The standard errors come from the curvature of $\chi^2$ around its minimum, as in any least-squares fit.

- **Standard error (SE):** the square root of a diagonal entry of $\widehat{\operatorname{Var}}(\hat\beta)$, i.e. the error bar on one coefficient.
- **t-statistic:** $t_j = \hat\beta_j / \mathrm{SE}_j$, the number of error bars between the estimate and 0.
- **$R^2$** $= 1 - \sum e_t^2 / \sum (y_t - \bar y)^2$: the fraction of the variance of $y$ explained by the fit.

**Assumptions behind the textbook error bars:** errors have mean zero given $X$ (**exogeneity**: no omitted variable or reverse causation correlated with $X$); errors are uncorrelated with each other and have constant variance; and, for exact small-sample t-distributions, they are Gaussian. Under the first two, OLS is the best linear unbiased estimator (Gauss–Markov theorem).

## 5. Hypothesis testing

| Term | Meaning |
|---|---|
| **Null hypothesis** $H_0$ | The default claim being tested, e.g. $\beta = 0$ (no effect) |
| **Alternative** $H_1$ | What you accept if you reject $H_0$, e.g. $\beta \ne 0$ |
| **Test statistic** | A number computed from data whose distribution under $H_0$ is known (here $t$) |
| **p-value** | Probability, *assuming $H_0$ is true*, of a statistic at least as extreme as observed |
| **Significance level** $\alpha$ | The threshold, e.g. 5%: reject $H_0$ if $p < \alpha$ |
| **Type I error** | Rejecting a true $H_0$ (false positive); happens with probability $\alpha$ if the test is valid |
| **Type II error** | Not rejecting a false $H_0$ (false negative) |
| **Power** | Probability of rejecting a false $H_0$ = 1 − P(type II) |

A p-value of 0.03 does **not** mean "a 3% chance the null is true", and $1 - p$ is not the probability the effect is real. It says that data this extreme would occur 3% of the time if there were no effect. In the notebook a valid test rejects a true null about 5% of the time at any sample size, while its power to detect $\beta = 0.2$ rises from 27% ($n=50$) to 99% ($n=500$).

## 6. When error bars lie: HAC standard errors

With autocorrelated or heteroskedastic errors, $\hat\beta$ is still sensible, but $\hat\sigma^2(X^\top X)^{-1}$ is wrong, usually too small. Autocorrelated data carries less information than its sample size suggests. The **HAC** (heteroskedasticity- and autocorrelation-consistent, **Newey–West**) estimator uses the "sandwich"

```math
\widehat{\operatorname{Var}}(\hat\beta) = (X^\top X)^{-1}\,\hat S\,(X^\top X)^{-1}, \qquad \hat S = \sum_{|j|\le L} w_j \sum_t e_t e_{t-j}\, x_t x_{t-j}^\top ,
```

where the weights $w_j = 1 - |j|/(L+1)$ fade out the covariance between residuals up to $L$ lags apart. The project's default is $L = \lfloor 4(T/100)^{2/9}\rfloor$.

In the notebook, with a persistent regressor and errors ($\phi = 0.9$, $T = 300$) and no true relationship, textbook errors give 54% false positives at a nominal 5%; HAC gives 24%. That is better but still far off, because HAC is only correct for large $T$ and a long enough window. **Use HAC by default for time-series regressions, and still distrust p-values from very persistent data.**

## 7. Spurious regression

Regress one *independent* random walk on another (Granger & Newbold, 1974). In the notebook, with $T = 500$:

- about 90% of regressions have $|t| > 1.96$ (it should be 5%);
- the median $|t|$ is about 10, and $R^2$ is often large;
- it gets *worse* with more data: the median $|t|$ grows like $\sqrt T$ (about 4.6, 10 and 20 for $T$ = 100, 500 and 2,000).

**Why** (Phillips, 1986): under $H_0$ the residual $y_t - \beta x_t$ is itself a random walk, so every assumption about the errors fails. $\hat\beta$ converges to a random variable, not to 0, and $t$ diverges. HAC errors barely help (still about 76% rejections), because the problem is the model, not the error-bar formula.

*Physics analogy:* two independent diffusing particles. Over any finite window both wander far, so their trajectories look correlated, and the sample correlation of two independent random walks has a broad distribution that doesn't shrink as the window grows.

**The cures:**

1. **Difference first:** regress returns on returns ($I(0)$ on $I(0)$). That is Exercise 2.
2. **Test the residual:** if some combination $y - \beta x$ *is* stationary, the regression is meaningful. That is cointegration, Module 5, and it needs special critical values precisely because of this module's lesson.

---

## Glossary

| Term | Meaning |
|---|---|
| White noise | Uncorrelated shocks with mean 0 and constant variance |
| Random walk | $x_t = x_{t-1} + \varepsilon_t$; shocks accumulate forever |
| Drift | A constant added each step, giving a trend |
| AR(1) | Autoregressive process of order 1: $x_t = c + \phi x_{t-1} + \varepsilon_t$ |
| Stationarity (weak) | Constant mean and variance; autocovariance depends only on the lag |
| Unit root | $\phi = 1$ in an AR model: no pull back, the series is a random walk |
| Integration order, $I(0)$ / $I(1)$ | Number of differences needed to make a series stationary |
| ACF / PACF | Autocorrelation function / partial autocorrelation function (correlation at lag $k$ net of the lags in between) |
| Correlation time | $-1/\ln\phi$ for AR(1): how long memory lasts |
| Heteroskedasticity | Error or return variance that changes over time (opposite: homoskedasticity) |
| Volatility clustering | Big moves follow big moves; squared returns are autocorrelated |
| OLS | Ordinary least squares: minimise the sum of squared residuals |
| Regressor | An explanatory variable (a column of $X$) |
| Residual | Observed minus fitted value, $e = y - X\hat\beta$ |
| Standard error | The estimated standard deviation of an estimate (its error bar) |
| $R^2$ | Fraction of the variance of $y$ explained by the regression |
| t-statistic | Estimate divided by its standard error |
| Null hypothesis | The default claim a test tries to reject |
| p-value | Probability under the null of a result at least as extreme as observed |
| Significance level | The rejection threshold for the p-value, e.g. 5% |
| Type I / type II error | False positive / false negative |
| Power | Probability of correctly rejecting a false null |
| HAC / Newey–West standard errors | Error bars robust to heteroskedasticity and autocorrelation |
| Spurious regression | A "significant" regression between unrelated non-stationary series |
| Ensemble | Many independent simulated paths of the same process |

## Interview questions

Write your own answer under each question first, then compare with the key points.

**1. What is stationarity, and why does it matter for trading?**

<details><summary>Key points</summary>

Weak stationarity: constant mean and variance, and autocovariance depending only on the lag. Its statistics estimated from the past apply to the future, so "far from the mean" implies an expected return towards it. A mean-reversion strategy needs a stationary spread; a non-stationary one can drift away forever. Physics: equilibrium vs diffusion.
</details>

**2. What is a unit root? Are prices or returns stationary?**

<details><summary>Key points</summary>

$\phi = 1$ in $x_t = \phi x_{t-1} + \varepsilon_t$: shocks never decay, so the variance grows like $t$ (a random walk, $I(1)$). Log prices are approximately $I(1)$ (ACF ≈ 1 at all lags); returns are approximately $I(0)$ (ACF ≈ 0), though their volatility clusters. Testing for a unit root is hard near $\phi = 1$ (Exercise 3; the ADF test in Module 5).
</details>

**3. Explain spurious regression. Why can't you regress price on price and trust the t-stat?**

<details><summary>Key points</summary>

Independent $I(1)$ series regressed on each other give $|t| > 1.96$ about 90% of the time (500 points), with high $R^2$, getting worse as $T$ grows (t ∝ √T). The residual is non-stationary, so the OLS error assumptions fail and $\hat\beta$ doesn't converge to the truth (Granger–Newbold 1974; Phillips 1986). Fix: difference the data, or test the residual for stationarity (cointegration).
</details>

**4. What does a p-value of 0.03 mean, and what does it *not* mean?**

<details><summary>Key points</summary>

If the null were true (and the model assumptions hold), data at least this extreme would occur 3% of the time. It is not the probability the null is true, not the probability the result is a fluke, and says nothing about effect size. It is only valid if the test's assumptions hold (spurious regression gives tiny p-values for nothing), and running many tests inflates false positives (Module 8).
</details>

**5. What is volatility clustering, and how would you detect it?**

<details><summary>Key points</summary>

Periods of high volatility persist: big moves follow big moves. Returns themselves are nearly uncorrelated, but squared or absolute returns have large, slowly decaying autocorrelation (SPY: 0.25–0.5 at short lags, still positive after 60 days). Detect with the ACF of squared returns, a Ljung–Box test on squared returns, or rolling volatility; model it with GARCH. Consequences: heteroskedastic regression errors (use HAC), time-varying risk, wider error bars than i.i.d. formulas suggest.
</details>

## References

- Hyndman & Athanasopoulos, *Forecasting: Principles and Practice* (3rd ed.), Chapters 2 and 9.1–9.3 (free online).
- Wooldridge, *Introductory Econometrics*, Chapters 2–4 (OLS and inference) and 10–11 (time-series regression).
- Granger & Newbold, "Spurious regressions in econometrics", *Journal of Econometrics* 2(2) (1974).
- Phillips, "Understanding spurious regressions in econometrics", *Journal of Econometrics* 33(3) (1986).
- Newey & West, "A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix", *Econometrica* 55(3) (1987).
