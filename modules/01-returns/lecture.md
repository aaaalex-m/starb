# Module 1, part 1 — Prices and returns

Every later model in this project works with **returns** (percentage changes), not prices. In this part you load real prices, see what a price series actually contains, turn prices into returns, and measure the facts about returns that every interviewer assumes you know: how they add up, how volatility scales with time, and how fat the tails are. Part 2 ([data pitfalls](../02-data-pitfalls/lecture.md)) covers what can go wrong with the data itself.

Notebook: [`notebook.ipynb`](notebook.ipynb). Code: [`statarb/data.py`](../../src/statarb/data.py), [`statarb/returns.py`](../../src/statarb/returns.py).

> [!TIP]
> **How to read these notes.** The numbered steps are the main path. Boxes are optional: notes and warnings are short asides; collapsed sections hold derivations and interview-level detail.

## Step 1. Look at one day of prices

```python
aapl = data.load_bars("AAPL", "2010-01-01", "2024-12-31")
aapl.loc["2020-08-26":"2020-09-01"]
```

| date | open | high | low | close | adj_close | volume | dividends | splits |
|---|---|---|---|---|---|---|---|---|
| 2020-08-28 | 126.01 | 126.44 | 124.58 | 124.81 | 120.96 | 1.9e8 | 0 | 0 |
| 2020-08-31 | 127.58 | 131.00 | 126.00 | 129.04 | 125.06 | 2.3e8 | 0 | 4 |
| 2020-09-01 | 132.76 | 134.80 | 130.53 | 134.18 | 130.04 | 1.5e8 | 0 | 0 |

Each row is a daily **bar** for one **stock** (a share of ownership in a company), identified by its **ticker** (AAPL is Apple). Open, high, low and close are the first, highest, lowest and last traded prices of the day; with **volume** (shares traded) the bar is called **OHLCV**.

**What it means:** this is the raw material for everything else. Two columns hint at complications: `splits` says something happened on 31 August, and `adj_close` differs from `close`. Steps 3 and 4 explain both.

> [!NOTE]
> Every trade crosses a **bid** (the best price a buyer will pay) or an **ask** (the best price a seller will accept). Their gap, the **spread**, is a cost on every trade; for large stocks it is well under 1 basis point.

## Step 2. Load a whole universe as one table

```python
TICKERS = ["SPY", "XLE", "XLP", "AAPL", "MSFT", "JPM", "XOM", "CVX", "KO", "PEP"]
prices = data.load_prices(TICKERS, "2010-01-01", "2024-12-31")  # 3774 days x 10 tickers
```

`load_prices` returns one column per ticker of adjusted closing prices, cached on disk after the first download.

**What it means:** a **universe** is the set of assets you study. **SPY** is an **ETF** (exchange-traded fund: a basket of stocks that trades like a single stock) that tracks the **S&P 500 index** (about 500 large US companies). **XLE** and **XLP** are ETFs of US energy and consumer-staples stocks. The rest are single stocks, including two textbook pairs: Exxon/Chevron (XOM, CVX) and Coca-Cola/PepsiCo (KO, PEP).

## Step 3. See a stock split in raw and adjusted prices

Apple split **4-for-1** on 31 August 2020: each share became four shares worth a quarter as much. Nothing economic happened, but the traded price fell from about 500 to about 125 overnight.

```
return on 2020-08-31 from as-traded prices: -74.2%
return on 2020-08-31 from adjusted prices:  +3.4%
```

**What it means:** a split is a **corporate action**, an event that changes the price but not the holders' wealth. An **adjusted close** rewrites history so that day-to-day ratios are the investor's true return: on each split date every earlier price is divided by the split ratio $k$.

> [!WARNING]
> A backtest on unadjusted prices books the split as a −74% crash, or as a fantastic fake profit if it was short. Yahoo's `close` column is *already* split-adjusted; the notebook rebuilds the as-traded price with `data.unadjust_splits` to show the jump.

## Step 4. See a dividend in raw and adjusted prices

Coca-Cola pays a cash **dividend** $D$ per share every quarter. On the **ex-dividend date** (the first day a buyer no longer receives it) the price falls by roughly $D$. On 16 March 2023 ($D = 0.46$):

```
return from close     -0.215%
return from adj_close +0.550%
difference             0.765%   vs   dividend / previous close  0.761%
```

**What it means:** the holder didn't lose money; the drop was paid in cash. The adjusted close multiplies all earlier prices by $1 - D/P_{t-1}$, so its return includes the dividend. The **price return** (from `close`) ignores dividends; the **total return** (from `adj_close`) includes them. Over 2010–2024 dividends were about half (KO, PEP, XLP) to two-thirds (XOM, CVX, XLE) of the total return.

**Rule for the rest of the project: compute returns from `adj_close`.**

<details>
<summary>Why adjusted prices change after every download</summary>

On each event date the adjustment factor ($1/k$ for a split, $1 - D/P_{t-1}$ for a dividend) is applied to all *earlier* prices, so the whole adjusted history is rewritten after every new dividend or split. The same download done next year gives different price *levels*, though the same returns; that is why the cache records its download date.

For a long/short pair (Module 3), dividends count twice: the long leg receives its dividends, and the short leg must *pay* the dividends on the shares it borrowed.

</details>

## Step 5. Turn prices into returns

```math
R_t = \frac{P_t}{P_{t-1}} - 1 \quad\text{(simple return)}, \qquad r_t = \ln\frac{P_t}{P_{t-1}} = \ln(1+R_t) \quad\text{(log return)}.
```

```python
R = returns.simple_returns(prices)
r = returns.log_returns(prices)
```

On a typical 1% day the two differ by about $R^2/2 = 5 \times 10^{-5}$ (half a basis point); on the largest days of March 2020 (about ±10%) they differ by 40–65 bp.

**What it means:** expanding the logarithm, $r = R - R^2/2 + \dots$, so the two agree for small moves and $r \lt R$ always. Both are used, because each adds up correctly in a different direction (the next two steps).

## Step 6. Add log returns over time

The total growth telescopes:

```math
\ln\frac{P_T}{P_0} = \sum_{t=1}^{T} r_t .
```

On SPY, the sum of 3,773 daily log returns and $\ln(P_{\text{end}}/P_{\text{start}})$ agree to six decimals (both 1.918802).

**What it means:** log returns are the natural unit for anything over time: multi-day returns, statistics, and models of price dynamics.

## Step 7. Add simple returns across assets

A portfolio with dollar weights $w_i$ (fractions of its value) earns

```math
R^{\text{port}}_t = \sum_i w_i R_{i,t} .
```

Averaging *log* returns instead is only approximate: for a 50/50 KO/PEP portfolio the error is tiny on a typical day (0.02 bp) but always of the same sign, up to 8 bp.

**What it means:** portfolio P&L, backtests and weights use simple returns.

<details>
<summary>Derivation, Jensen's inequality, and volatility drag</summary>

With $n_i$ shares, value $V_t = \sum_i n_i P_{i,t}$ and weights $w_i = n_i P_{i,t-1}/V_{t-1}$:

$$R_t^{\text{port}} = \frac{V_t}{V_{t-1}} - 1 = \sum_i \frac{n_i P_{i,t-1}}{V_{t-1}}\Big(\frac{P_{i,t}}{P_{i,t-1}} - 1\Big) = \sum_i w_i R_{i,t}.$$

The same fails for log returns because $\ln$ of a weighted average is not the weighted average of $\ln$. Since $\ln$ is concave, the weighted log return is always the smaller (**Jensen's inequality**), which is why the error has one sign.

**Volatility drag.** Taking expectations of $r = R - R^2/2 + \dots$ for returns with mean $\mu$ and variance $\sigma^2$ gives $\mathbb E[r] \approx \mu - \sigma^2/2$: the average simple return overstates the compound growth rate. Example: +50% then −50% averages 0% but loses 25%. Exercise 1 checks this on SPY.

</details>

## Step 8. Measure volatility and annualise it

**Volatility** $\sigma$ is the standard deviation of returns. Statistics are quoted per year:

```math
\mu_{\text{ann}} = 252\,\mu_{\text{daily}}, \qquad \sigma_{\text{ann}} = \sqrt{252}\,\sigma_{\text{daily}} .
```

```python
returns.summary(r)  # annualised mean and volatility, skewness, excess kurtosis
```

Annualised volatility ranges from about 14–17% (XLP, KO, SPY, PEP) to about 28% (XLE, JPM, AAPL).

**What it means:** the square root comes from diffusion. If daily returns are uncorrelated, their variances add, so the variance of a $T$-day return is $T$ times the daily one and its standard deviation grows like $\sqrt T$, just as $\langle x^2\rangle = 2Dt$ for a diffusing particle.

> [!NOTE]
> Volatility is not constant: a rolling 21-day estimate for SPY sits near 10–15% for long stretches, then jumps (above 90% in March 2020). Turbulent days come in bunches; this is **volatility clustering** (Module 2).

<details>
<summary>When the square-root-of-time rule fails: autocorrelated returns</summary>

With $\rho_k$ the correlation between returns $k$ days apart,

$$\operatorname{Var}\Big(\sum_{t=1}^{T} r_t\Big) = T\sigma^2\Big(1 + 2\sum_{k=1}^{T-1}\big(1-\tfrac{k}{T}\big)\rho_k\Big).$$

So the $\sqrt{252}$ rule assumes no autocorrelation (returns need not be Gaussian, only uncorrelated) and constant variance. In the notebook, volatility measured from monthly returns is 10–25% *lower* than from daily returns for most assets, because daily returns are slightly negatively autocorrelated. For a mean-reverting spread, the thing stat arb trades, the variance stops growing altogether (Module 5).

</details>

## Step 9. Check how fat the tails are

Standardise SPY's daily returns and compare them with a Gaussian. Two numbers summarise the shape:

```math
\text{skewness} = \frac{\mathbb E[(x-\mu)^3]}{\sigma^3}, \qquad \text{excess kurtosis} = \frac{\mathbb E[(x-\mu)^4]}{\sigma^4} - 3 .
```

Both are 0 for a Gaussian. For our ten assets the skewness is negative (big down days are bigger than big up days) and the excess kurtosis ranges from 5 (AAPL) to 27 (CVX).

**What it means:** returns have **fat tails**: extreme days are far more common than a Gaussian predicts. A Gaussian puts a 5σ day at probability $5.7 \times 10^{-7}$, about once in 7,000 years of trading; SPY had nine of them in 2010–2024.

> [!WARNING]
> A long/short book is hedged against *typical* co-movement. On extreme days correlations change and hedges break, so positions that are individually small can lose together.

## Step 10. Go long, go short, and stay market-neutral

Being **long** means owning an asset (you profit if it rises); being **short** means borrowing it and selling it, to buy it back later (you profit if it falls). Stat arb holds both. Measure each asset's sensitivity to the market, its **beta** (the slope of its daily return regressed on SPY's):

```python
beta = R.cov()["SPY"] / R["SPY"].var()
```

KO and PEP have betas near 0.6, XOM 0.88, JPM about 1.2. One dollar long XOM and one dollar short KO has equal dollar sizes, but a net beta of $0.88 - 0.60 = +0.28$: it still gains when the market rises.

**What it means:** with $w_i$ the dollar weight of each position (negative for shorts), a book is **dollar-neutral** if $\sum_i w_i = 0$ and **beta-neutral** (market-neutral) if $\sum_i w_i \beta_i = 0$. They are not the same: to be beta-neutral here you would short about 1.47 dollars of KO per dollar of XOM. Stat arb aims for both, so that profit comes from the *relative* move of the legs.

> [!NOTE]
> *Physics analogy:* a differential measurement. Subtracting two signals with a large common-mode component (the market) leaves the small relative signal. The analogy is imperfect because beta, the "common-mode gain", is estimated with error and drifts.

> [!NOTE]
> Shorting has costs and risks of its own: you pay the lender a **borrow fee** and any dividends, your loss is unbounded if the price rises, and the lender can recall the shares (Module 6).

## Step 11. Exercises

Do these in the empty cells at the end of the notebook. Your answers aren't written here.

1. **Volatility drag.** Show numerically that the sum of SPY's daily log returns equals the log of total growth, and that the average simple return overstates the growth rate. Compare the gap with $\sigma^2/2$.
2. **Counting tail events.** Estimate the excess kurtosis of SPY's daily returns. How many days beyond 4 standard deviations would a Gaussian predict over these 15 years, and how many occurred?
3. **A first spread.** Build a portfolio one dollar long KO and one dollar short PEP (rebalanced daily) and plot its cumulative P&L next to each leg. How do the volatility and the market beta of the spread compare with each leg's?

**Done when:** `load_prices` caches correctly, and you can explain every column of a price bar and every term in the glossary below without notes.

<details>
<summary>Interview questions (5), with key points</summary>

Write your own answer first, then open the key points.

**1. Why use log returns for time aggregation but simple returns for portfolio aggregation?**

<details>
<summary>Key points</summary>

Log returns telescope: $\sum_t r_t = \ln(P_T/P_0)$. Simple returns are linear in prices, so a portfolio's simple return is the weighted sum of its assets' simple returns. Neither property holds for the other kind. For small daily moves the two agree to $O(R^2)$, but the gap is systematic.

</details>

**2. What is an adjusted close, and what goes wrong if you backtest on unadjusted prices?**

<details>
<summary>Key points</summary>

Close back-adjusted for splits (÷k) and dividends (×(1 − D/P)) so that ratios are total returns. Unadjusted: fake crashes on split dates, a missing dividend yield (a large share of long-run return), wrong P&L on long/short pairs where one leg pays dividends. Yahoo's `Close` is already split-adjusted, and adjusted history changes after each new event, so record the download date.

</details>

**3. How do you annualise volatility, and what assumption does $\sqrt{252}$ rely on?**

<details>
<summary>Key points</summary>

Multiply daily σ by $\sqrt{252}$, because variances of uncorrelated returns add (diffusion). It assumes zero autocorrelation and constant variance, not Gaussianity. Autocorrelation adds covariance terms (negative autocorrelation means less long-horizon variance); volatility clustering means the daily σ itself varies.

</details>

**4. What does "market-neutral" mean, and is dollar-neutral the same as beta-neutral?**

<details>
<summary>Key points</summary>

Market-neutral: P&L insensitive to the market's direction, i.e. net beta ≈ 0. Dollar-neutral only equalises long and short dollars; if the legs have different betas the book still has market exposure (long 1 XOM / short 1 KO has net β ≈ +0.28). Beta-neutral sizes legs by beta. Betas are estimates and drift, so neutrality is approximate.

</details>

**5. Why are fat tails important for a strategy that holds offsetting long and short positions?**

<details>
<summary>Key points</summary>

Hedges are calibrated on typical days, and extreme days are far more common than Gaussian models predict (SPY: nine 5σ days in 15 years against one expected per ~7,000 years). In crises correlations shift and hedges break, so both legs can lose together, and leverage amplifies the loss.

</details>

</details>

<details>
<summary>Glossary of terms in this module</summary>

| Term | Meaning |
|---|---|
| Stock / share; ticker | A unit of ownership in a company; its trading symbol (KO, SPY) |
| Bar, OHLCV | One period's open, high, low, close prices and volume |
| Bid / ask / spread | Best price to sell at / to buy at / their gap |
| ETF; index | A basket that trades like a single stock; a rule-based benchmark basket (S&P 500) |
| Universe | The set of assets studied or traded |
| Corporate action; stock split | Event that changes the price but not holders' wealth; $k$ shares at $1/k$ of the price |
| Dividend / ex-dividend date | Cash paid per share / first day a buyer no longer receives it |
| Adjusted close | Close back-adjusted for splits and dividends so its ratios are total returns |
| Price return / total return | Return from price changes only / including dividends |
| Simple return $R$ / log return $r$ | $P_t/P_{t-1} - 1$, adds across assets / $\ln(P_t/P_{t-1})$, adds over time |
| Volatility; annualise | Standard deviation of returns; mean × 252, volatility × $\sqrt{252}$ |
| Volatility drag | Compound growth falls short of the average simple return by about $\sigma^2/2$ |
| Volatility clustering | Large moves tend to follow large moves |
| Skewness / excess kurtosis | Third standardised moment / fourth minus 3 |
| Fat tails | Extreme moves far more frequent than a Gaussian predicts |
| Long / short; borrow fee | Own the asset / borrow and sell it; the cost of borrowing shares |
| Beta | Sensitivity of an asset's return to the market's |
| Dollar-neutral / beta-neutral | Long and short dollars equal / net market sensitivity zero |
| Basis point (bp) | 0.01%, i.e. $10^{-4}$ |

</details>

<details>
<summary>References and further reading</summary>

- Harris, *Trading and Exchanges* (2003), Chapters 1–4 (skim): market structure, orders, spreads.
- Tsay, *Analysis of Financial Time Series* (3rd ed., 2010), Chapter 1: returns and their distributional properties.
- Bouchaud & Potters, *Theory of Financial Risk and Derivative Pricing* (2nd ed., 2003), Chapter 1: fat tails.

</details>
