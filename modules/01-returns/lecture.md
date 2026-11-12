# Module 1 — Markets, prices and returns

**Why this module:** every later model works with returns, not prices. You need to know what is being traded, what a price series really contains, and the handful of facts about returns (how they add up, how volatility scales, how fat the tails are) that every interviewer assumes.

Notebook: [`notebook.ipynb`](notebook.ipynb). Code: [`statarb/data.py`](../../src/statarb/data.py), [`statarb/returns.py`](../../src/statarb/returns.py).

---

## 1. What is traded

- A **stock** (also **share** or **equity**) is a unit of ownership in a company. It is identified by a **ticker**, a short symbol such as KO (Coca-Cola).
- An **index** is a rule-based basket used as a benchmark. The S&P 500 holds about 500 large US companies, weighted by **market capitalisation** (share price × number of shares). An index is a number, not something you can buy.
- An **ETF** (exchange-traded fund) is a fund that holds a basket and trades like a single stock. SPY tracks the S&P 500; XLE holds US energy stocks; XLP holds consumer-staples stocks such as KO and PEP.
- Trades happen on an exchange against standing orders. The **bid** is the best price someone will pay you; the **ask** (or offer) is the best price someone will sell to you. The **spread** (ask − bid) is a cost you pay every time you trade. The **mid** is their average.
- **Liquidity** is how easily you can trade without moving the price. For a large cap such as AAPL the spread is usually one cent on a price of a few hundred dollars, well under 1 bp; for small caps it is often tens of basis points.

### The daily bar (OHLCV)

One row per trading day: **open**, **high**, **low** and **close** prices (the first, highest, lowest and last trade, with the close set by a closing auction) plus **volume** (shares traded). We use **close-to-close** data: each daily return runs from one close to the next.

## 2. Corporate actions and the adjusted close

**Corporate actions** are company events that change the price without changing the shareholder's wealth.

- **Stock split** ($k$-for-1): each share becomes $k$ shares, so the price drops by a factor $k$. Apple split 4-for-1 on 31 August 2020; its traded price went from about 500 to about 125 overnight.
- **Cash dividend** $D$ per share: paid to whoever holds the stock before the **ex-dividend date**. On that date the price falls by roughly $D$, since buyers no longer receive it.

An **adjusted close** is the close rewritten backwards in time so that its ratios give the investor's total return. On each event date $t$, every earlier price is multiplied by a factor:

```math
\text{split } k:\quad \times \frac{1}{k}, \qquad\qquad \text{dividend } D:\quad \times\Big(1 - \frac{D}{P_{t-1}}\Big).
```

Then $P^{\text{adj}}_t / P^{\text{adj}}_{t-1} - 1$ is the true one-day return, including the dividend.

**Yahoo's conventions** (what `statarb.data` stores):

| column | splits | dividends | use |
|---|---|---|---|
| `close` | adjusted | not adjusted | display; **price return** |
| `adj_close` | adjusted | adjusted | **all return calculations** (**total return**) |

Two consequences. First, the *as-traded* price isn't in either column; `data.unadjust_splits` rebuilds it. Second, adjusted history is rewritten after every new dividend or split, so the same download done next year gives different numbers. That is why the cache records its download date.

**What goes wrong with unadjusted prices:** a split shows up as a fake −75% day (a fantastic fake profit if you were short); every ex-dividend date shows a small fake loss. Over 2010–2024 dividends account for roughly half (KO, PEP, XLP) to two-thirds (XOM, CVX, XLE) of the total return. For a long/short pair it's worse: the long leg *receives* dividends and the short leg *pays* them.

## 3. Returns

```math
R_t = \frac{P_t}{P_{t-1}} - 1 \quad\text{(simple)}, \qquad r_t = \ln\frac{P_t}{P_{t-1}} = \ln(1+R_t) \quad\text{(log)}.
```

Expanding, $r = R - R^2/2 + R^3/3 - \dots$, so the two agree to first order and $r < R$ whenever $R \neq 0$. On a 1% day the gap is half a basis point; on a 10% day it is 50 bp.

**Log returns add over time.** The total growth telescopes:

```math
\ln\frac{P_T}{P_0} = \sum_{t=1}^{T} r_t .
```

This makes log returns convenient for multi-period statistics and for modelling (a sum of many small shocks).

**Simple returns add across assets.** With $n_i$ shares, portfolio value $V_t = \sum_i n_i P_{i,t}$ and weights $w_i = n_i P_{i,t-1} / V_{t-1}$:

```math
R^{\text{port}}_t = \frac{V_t}{V_{t-1}} - 1 = \sum_i w_i R_{i,t} .
```

The same is *not* true of log returns, since $\ln$ of a weighted average is not the weighted average of $\ln$ (Jensen's inequality).

| Task | Use |
|---|---|
| Portfolio P&L, backtests, weights | simple returns |
| Aggregating over time, statistics, models of price dynamics | log returns |

**Volatility drag.** The average simple return overstates the growth rate. For returns with mean $\mu$ and variance $\sigma^2$, taking expectations of the Taylor expansion,

```math
\mathbb E[r] \approx \mu - \tfrac{1}{2}\sigma^2 .
```

Example: +50% then −50% averages 0% but loses 25%. Physicists will recognise this from geometric Brownian motion, $dP/P = \mu\,dt + \sigma\,dW$: Itô's lemma gives $d\ln P = (\mu - \sigma^2/2)\,dt + \sigma\,dW$, because $(dW)^2 = dt$ makes the second-order term survive.

## 4. Volatility and annualisation

**Volatility** $\sigma$ is the standard deviation of returns. Statistics are quoted per year, using 252 trading days:

```math
\mu_{\text{ann}} = 252\,\mu_{\text{daily}}, \qquad \sigma_{\text{ann}} = \sqrt{252}\,\sigma_{\text{daily}} .
```

Why $\sqrt{T}$: the variance of a sum is the sum of the variances when the terms are uncorrelated. In general, with $\rho_k$ the correlation between returns $k$ days apart,

```math
\operatorname{Var}\Big(\sum_{t=1}^{T} r_t\Big) = T\sigma^2\Big(1 + 2\sum_{k=1}^{T-1}\big(1-\tfrac{k}{T}\big)\rho_k\Big).
```

*Physics analogy:* the log price is a diffusing particle, $\langle x^2\rangle = 2Dt$, so its spread grows as $\sqrt{t}$.

The $\sqrt{252}$ rule therefore assumes:

1. **No autocorrelation** ($\rho_k = 0$). Returns need not be Gaussian or even independent, just uncorrelated.
2. **Constant variance** over the period. In reality volatility moves around a lot (**volatility clustering**: turbulent days come in bunches, e.g. March 2020).

In the notebook, monthly-based volatility comes out 10–25% below daily-based volatility because daily returns are slightly *negatively* autocorrelated. For a mean-reverting spread, the thing stat arb trades, the effect is extreme: variance stops growing at all (a particle in a confining potential, Module 6).

## 5. Distributions and fat tails

For a random variable $x$ with mean $\mu$ and standard deviation $\sigma$:

```math
\text{skewness} = \frac{\mathbb E[(x-\mu)^3]}{\sigma^3}, \qquad \text{kurtosis} = \frac{\mathbb E[(x-\mu)^4]}{\sigma^4}, \qquad \text{excess kurtosis} = \text{kurtosis} - 3 .
```

A Gaussian has skewness 0 and excess kurtosis 0. Daily equity returns from 2010–2024 have excess kurtosis between 5 and 27 and negative skew: the distribution is more peaked in the middle, with **fat tails** and bigger crashes than rallies.

How fat? A Gaussian puts a 5σ day at probability $5.7 \times 10^{-7}$, about once every 7,000 years of trading. SPY had nine such days in 2010–2024 (one in August 2011, eight in spring 2020).

*Physics link:* the tails of stock returns decay roughly as a power law, $P(|r| > x) \propto x^{-\alpha}$ with $\alpha \approx 3$, the "inverse cubic law" (Gopikrishnan et al., 1998). With $\alpha < 4$ the fourth moment formally diverges, so sample kurtosis is unstable: it is dominated by the few largest days and keeps growing as the sample grows. That is why CVX shows 27 and AAPL 5.

**Why fat tails matter for stat arb:** a long/short book is hedged against *typical* co-movement. On extreme days correlations change, hedges break, and positions that are individually small can lose together. Gaussian risk estimates understate the probability of such days by orders of magnitude.

## 6. Positions: long, short and neutral

- **Long:** you own the asset and profit if it rises.
- **Short:** you borrow shares (through a broker, from a lender), sell them, and later buy them back to return them. You profit if the price falls. You pay any dividends to the lender and a **borrow fee**; your loss is unbounded if the price rises; the lender can **recall** the shares. Details in Module 8.
- **Benchmark:** a reference portfolio, usually the market (SPY here).

**Beta and alpha.** Regress an asset's excess return on the market's:

```math
R_{i,t} - r_f = \alpha_i + \beta_i\,(R_{m,t} - r_f) + \varepsilon_{i,t}, \qquad \beta_i = \frac{\operatorname{Cov}(R_i, R_m)}{\operatorname{Var}(R_m)} .
```

**Beta** $\beta$ is the sensitivity to the market (KO ≈ 0.6, JPM ≈ 1.2 in our sample). **Alpha** $\alpha$ is the average return not explained by market exposure. The **risk-free rate** $r_f$ is the return on short-term government debt; **excess return** is return minus $r_f$.

**Neutrality.** Write $w_i$ for the dollar weight of position $i$ (negative for shorts).

- **Dollar-neutral:** long and short notional (dollar size) are equal, $\sum_i w_i = 0$.
- **Beta-neutral** (market-neutral): net market sensitivity is zero, $\sum_i w_i \beta_i = 0$.

They are not the same. One dollar long XOM (β ≈ 0.88) and one dollar short KO (β ≈ 0.60) is dollar-neutral but has a net beta of +0.28; to be beta-neutral you would short about 1.47 dollars of KO per dollar of XOM.

**Why stat arb aims to be market-neutral:** the bet is on the *relative* price of related assets, not on the market's direction. *Physics analogy:* a differential measurement. Subtracting two signals with a large common-mode component (the market) leaves the small relative signal, as in a differential amplifier. The analogy breaks down because the "common-mode gain" (beta) is estimated with error and drifts, and jumps in crises.

---

## Glossary

| Term | Meaning |
|---|---|
| Stock / share / equity | A unit of ownership in a company |
| Ticker | The symbol identifying a security (KO, SPY) |
| Index | A rule-based basket used as a benchmark (S&P 500); not directly tradable |
| ETF | Exchange-traded fund: a basket that trades like a single stock |
| Market capitalisation | Share price × number of shares outstanding |
| Liquidity | How easily you can trade without moving the price |
| Bid / ask / spread / mid | Best buying price / best selling price / their gap / their average |
| OHLCV | Open, high, low, close, volume: the daily bar |
| Close-to-close | Returns measured between consecutive closing prices |
| Corporate action | Company event that changes the price but not holders' wealth (split, dividend) |
| Stock split | Each share becomes $k$ shares at $1/k$ of the price |
| Dividend / ex-dividend date | Cash paid per share / first day a buyer no longer receives it |
| Adjusted close | Close back-adjusted for splits and dividends so its ratios are total returns |
| Price return / total return | Return from price changes only / including dividends |
| Simple return $R$ | $P_t/P_{t-1} - 1$; adds across assets |
| Log return $r$ | $\ln(P_t/P_{t-1})$; adds over time |
| Volatility | Standard deviation of returns, usually annualised |
| Annualise | Convert to yearly units: mean × 252, volatility × $\sqrt{252}$ |
| Volatility drag | Geometric growth falls short of the average simple return by about $\sigma^2/2$ |
| Volatility clustering | Large moves tend to follow large moves |
| Skewness / kurtosis | Third / fourth standardised moments (asymmetry / tail weight) |
| Fat tails | Extreme moves far more frequent than a Gaussian predicts (excess kurtosis > 0) |
| Long / short | Own the asset / borrow and sell it, profiting if it rises / falls |
| Notional | The dollar size of a position |
| Benchmark | Reference portfolio, usually the market |
| Beta | Sensitivity of an asset's return to the benchmark's |
| Alpha | Return not explained by benchmark exposure |
| Risk-free rate / excess return | Return on short-term government debt / return minus it |
| Dollar-neutral | Long and short dollars equal |
| Market-neutral / beta-neutral | Net sensitivity to the market is zero |
| Basis point (bp) | 0.01%, i.e. $10^{-4}$ |

## Interview questions

Write your own answer under each question first, then compare with the key points.

**1. Why use log returns for time aggregation but simple returns for portfolio aggregation?**

<details><summary>Key points</summary>

Log returns telescope: $\sum_t r_t = \ln(P_T/P_0)$. Simple returns are linear in prices, so a portfolio's simple return is the weighted sum of its assets' simple returns. Neither property holds for the other kind. For small daily moves the two agree to $O(R^2)$, but the gap is systematic.
</details>

**2. What is an adjusted close, and what goes wrong if you backtest on unadjusted prices?**

<details><summary>Key points</summary>

Close back-adjusted for splits (÷k) and dividends (×(1 − D/P)) so that ratios are total returns. Unadjusted: fake crashes on split dates, a missing dividend yield (a large share of long-run return), wrong P&L on long/short pairs where one leg pays dividends. Also know your vendor: Yahoo's `Close` is already split-adjusted. Adjusted history changes after each new event, so record the download date.
</details>

**3. How do you annualise volatility, and what assumption does $\sqrt{252}$ rely on?**

<details><summary>Key points</summary>

Multiply daily σ by $\sqrt{252}$, because variances of uncorrelated returns add (diffusion). It assumes zero autocorrelation and constant variance; it does not need Gaussianity. Autocorrelation adds covariance terms (negative autocorrelation means less long-horizon variance); volatility clustering means the daily σ itself varies.
</details>

**4. What does "market-neutral" mean, and is dollar-neutral the same as beta-neutral?**

<details><summary>Key points</summary>

Market-neutral: P&L insensitive to the market's direction, i.e. net beta ≈ 0. Dollar-neutral only equalises long and short dollars; if the legs have different betas, the book still has market exposure (e.g. long 1 XOM / short 1 KO has net β ≈ +0.28). Beta-neutral sizes legs by beta. Betas are estimates and drift, so neutrality is approximate.
</details>

**5. Why are fat tails important for a strategy that holds offsetting long and short positions?**

<details><summary>Key points</summary>

Hedges are calibrated on typical days; extreme days are far more common than Gaussian models predict (SPY: nine 5σ days in 15 years versus one expected per ~7,000 years). In crises correlations shift and hedges break, so both legs can lose together, and leverage amplifies the loss. Risk models must account for tail risk (Module 14), and kurtosis estimates are themselves unstable because the tails follow a power law.
</details>

## References

- Harris, *Trading and Exchanges* (2003), Chapters 1–4 (skim): market structure, orders, spreads.
- Tsay, *Analysis of Financial Time Series* (3rd ed., 2010), Chapter 1: returns and their distributional properties.
- Bouchaud & Potters, *Theory of Financial Risk and Derivative Pricing* (2nd ed., 2003), Chapter 1: fat tails.
- Gopikrishnan, Meyer, Amaral & Stanley, "Inverse cubic law for the distribution of stock price variations", *European Physical Journal B* 3, 139 (1998).
- Lo & MacKinlay, "Stock market prices do not follow random walks: Evidence from a simple specification test", *Review of Financial Studies* 1(1) (1988): the variance-ratio test.
