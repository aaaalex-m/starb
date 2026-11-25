# Module 1 — Markets, prices and returns

Every later model in this project works with **returns** (percentage changes), not prices. In this module you load real prices, see what a price series actually contains, turn prices into returns, and measure the handful of facts about returns that every interviewer assumes you know: how they add up, how volatility scales with time, and how fat the tails are.

Notebook: [`notebook.ipynb`](notebook.ipynb). Code: [`statarb/data.py`](../../src/statarb/data.py), [`statarb/returns.py`](../../src/statarb/returns.py).

> [!TIP]
> **How to read these notes.** The numbered steps are the main path. Boxes are optional: notes and warnings are short asides; collapsed sections hold derivations, extra jargon and interview-level detail.

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

Each row is a daily **bar** for one **stock** (a share of ownership in a company), identified by its **ticker** (AAPL is Apple). The first, highest, lowest and last traded prices of the day are **open, high, low, close** (with **volume**, the number of shares traded, the bar is called **OHLCV**).

**What it means:** this is the raw material for everything else. Two columns already hint at complications: `splits` says something happened on 31 August, and `adj_close` differs from `close`. Steps 3 and 4 explain both.

<details>
<summary>Market plumbing: exchanges, bid and ask, spread, liquidity, market cap</summary>

- Trades happen on an exchange against standing orders. The **bid** is the best price someone will pay you; the **ask** (or offer) is the best price someone will sell to you. The **spread** (ask − bid) is a cost you pay on every trade; the **mid** is their average.
- **Liquidity** is how easily you can trade without moving the price. For a large stock such as AAPL the spread is one cent on a few hundred dollars, well under 1 bp; for small companies it is often tens of basis points.
- **Market capitalisation** is share price × number of shares: the market value of the whole company.
- The closing price is set by a closing auction, which is why it is the standard reference price.

</details>

## Step 2. Load a whole universe as one table

```python
TICKERS = ["SPY", "XLE", "XLP", "AAPL", "MSFT", "JPM", "XOM", "CVX", "KO", "PEP"]
prices = data.load_prices(TICKERS, "2010-01-01", "2024-12-31")  # 3774 days x 10 tickers
```

`load_prices` returns one column per ticker of adjusted closing prices, cached on disk after the first download.

**What it means:** a **universe** is the set of assets you study. Two kinds appear here. **SPY** is an **ETF** (exchange-traded fund: a basket of stocks that trades like a single stock) that tracks the **S&P 500 index** (a rule-based basket of about 500 large US companies, used as a benchmark; an index is a number, not something you can buy). **XLE** and **XLP** are ETFs holding US energy and consumer-staples stocks. The rest are single stocks, including two textbook pairs: Exxon/Chevron (XOM, CVX) and Coca-Cola/PepsiCo (KO, PEP).

## Step 3. See a stock split in raw and adjusted prices

Apple split **4-for-1** on 31 August 2020: each share became four shares worth a quarter as much. Nothing economic happened, but the traded price fell from about 500 to about 125 overnight.

```
return on 2020-08-31 from as-traded prices: -74.2%
return on 2020-08-31 from adjusted prices:  +3.4%
```

**What it means:** a split is a **corporate action**, a company event that changes the price but not the holders' wealth. An **adjusted close** rewrites history so that day-to-day ratios are the investor's true return: on each split date every earlier price is divided by the split ratio $k$. Then $P_t^{\text{adj}} / P_{t-1}^{\text{adj}} - 1$ is the real one-day return.

> [!WARNING]
> A backtest on unadjusted prices books the split as a −74% crash, or as a fantastic fake profit if it was short. Yahoo's `close` column is *already* split-adjusted; the notebook rebuilds the as-traded price with `data.unadjust_splits` to show the jump.

## Step 4. See a dividend in raw and adjusted prices

Coca-Cola pays a cash **dividend** $D$ per share every quarter. On the **ex-dividend date** (the first day a buyer no longer receives the upcoming dividend) the price falls by roughly $D$. On 16 March 2023 ($D = 0.46$):

```
return from close     -0.215%
return from adj_close +0.550%
difference             0.765%   vs   dividend / previous close  0.761%
```

**What it means:** the holder didn't lose money; the drop was paid in cash. The adjusted close multiplies all earlier prices by $1 - D/P_{t-1}$, so the return includes the dividend. The **price return** (from `close`) ignores dividends; the **total return** (from `adj_close`) includes them. Over 2010–2024 dividends were about half (KO, PEP, XLP) to two-thirds (XOM, CVX, XLE) of the total return.

**Rule for the rest of the project: compute returns from `adj_close`.**

<details>
<summary>Adjustment factors, Yahoo's conventions, and why adjusted history changes</summary>

On each event date $t$, every earlier price is multiplied by

$$\text{split } k:\quad \times \frac{1}{k}, \qquad\qquad \text{dividend } D:\quad \times\Big(1 - \frac{D}{P_{t-1}}\Big).$$

| Yahoo column | splits | dividends | use |
|---|---|---|---|
| `close` | adjusted | not adjusted | display; price return |
| `adj_close` | adjusted | adjusted | all return calculations |

Because the factors apply to all *earlier* prices, the whole adjusted history is rewritten after every new dividend or split. The same download done next year gives different price *levels* (the returns are unchanged), which is why the cache records its download date.

For a long/short pair (Module 4), dividends matter twice: the long leg receives its dividends, and the short leg must *pay* the lender the dividends on the shares it borrowed.

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

**What it means:** expanding the logarithm, $r = R - R^2/2 + R^3/3 - \dots$, so the two agree for small moves and $r \lt R$ always. Both are in use, because each adds up correctly in a different direction (next step).

## Step 6. Add returns up: over time, and across assets

**Over time, log returns add.** The total growth telescopes:

```math
\ln\frac{P_T}{P_0} = \sum_{t=1}^{T} r_t .
```

The notebook checks it on SPY: the sum of 3,773 daily log returns and $\ln(P_{\text{end}}/P_{\text{start}})$ agree to six decimals (both 1.918802).

**Across assets, simple returns add.** A portfolio with dollar weights $w_i$ (fractions of its value) earns

```math
R^{\text{port}}_t = \sum_i w_i R_{i,t} .
```

Averaging *log* returns instead is only approximate: for a 50/50 KO/PEP portfolio the error is tiny on a typical day (0.02 bp) but always of the same sign, up to 8 bp.

**What it means:**

| Task | Use |
|---|---|
| Portfolio P&L, backtests, weights | simple returns |
| Aggregating over time, statistics, models of price dynamics | log returns |

<details>
<summary>Derivations: portfolio returns, Jensen's inequality, volatility drag and Itô's lemma</summary>

**Portfolio return.** With $n_i$ shares, value $V_t = \sum_i n_i P_{i,t}$ and weights $w_i = n_i P_{i,t-1}/V_{t-1}$,

$$R_t^{\text{port}} = \frac{V_t}{V_{t-1}} - 1 = \sum_i \frac{n_i P_{i,t-1}}{V_{t-1}}\Big(\frac{P_{i,t}}{P_{i,t-1}} - 1\Big) = \sum_i w_i R_{i,t}.$$

The same fails for log returns because $\ln$ of a weighted average is not the weighted average of $\ln$. Since $\ln$ is concave, the weighted log return is always the smaller (**Jensen's inequality**), which is why the error above has one sign.

**Volatility drag.** Taking expectations of $r = R - R^2/2 + \dots$ for returns with mean $\mu$ and variance $\sigma^2$:

$$\mathbb E[r] \approx \mu - \tfrac{1}{2}\sigma^2 .$$

The average simple return overstates the compound growth rate. Example: +50% then −50% averages 0% but loses 25%.

**Physics link.** For geometric Brownian motion, $dP/P = \mu dt + \sigma dW$, Itô's lemma gives $d\ln P = (\mu - \sigma^2/2) dt + \sigma dW$. The second-order term survives because $(dW)^2 = dt$: the same $\sigma^2/2$ drag.

</details>

## Step 7. Measure volatility and annualise it

**Volatility** $\sigma$ is the standard deviation of returns. Statistics are quoted per year:

```math
\mu_{\text{ann}} = 252\,\mu_{\text{daily}}, \qquad \sigma_{\text{ann}} = \sqrt{252}\,\sigma_{\text{daily}} .
```

```python
returns.summary(r)  # annualised mean and volatility, skewness, excess kurtosis
```

Annualised volatility ranges from about 14–17% (XLP, KO, SPY, PEP) to about 28% (XLE, JPM, AAPL).

**What it means:** the square root comes from diffusion. If daily returns are uncorrelated, their variances add, so the variance of a $T$-day return is $T$ times the daily one and its standard deviation grows like $\sqrt T$. The log price behaves like a diffusing particle, $\langle x^2\rangle = 2Dt$.

> [!NOTE]
> Volatility is not constant. A rolling 21-day estimate for SPY sits near 10–15% for long stretches and then jumps: 2011, 2015, 2018, above 90% in March 2020, 2022. Turbulent days come in bunches; this is **volatility clustering** (Module 3).

<details>
<summary>When the square-root-of-time rule fails: autocorrelation and the variance ratio</summary>

With $\rho_k$ the correlation between returns $k$ days apart,

$$\operatorname{Var}\Big(\sum_{t=1}^{T} r_t\Big) = T\sigma^2\Big(1 + 2\sum_{k=1}^{T-1}\big(1-\tfrac{k}{T}\big)\rho_k\Big).$$

The $\sqrt{252}$ rule therefore assumes no autocorrelation ($\rho_k = 0$; returns need not be Gaussian or independent, only uncorrelated) and constant variance over the period.

In the notebook, volatility measured from monthly returns is 10–25% *lower* than from daily returns for most assets, because daily returns are slightly negatively autocorrelated (lag-1 autocorrelation between −0.02 and −0.14). Comparing variances across horizons like this is the **variance-ratio test** (Lo and MacKinlay, 1988). For a mean-reverting spread, the thing stat arb trades, the effect is extreme: the variance stops growing altogether (a particle in a confining potential, Module 6).

</details>

## Step 8. Check how fat the tails are

Standardise SPY's daily returns (subtract the mean, divide by the standard deviation) and compare them with a Gaussian. Two numbers summarise the shape:

```math
\text{skewness} = \frac{\mathbb E[(x-\mu)^3]}{\sigma^3}, \qquad \text{excess kurtosis} = \frac{\mathbb E[(x-\mu)^4]}{\sigma^4} - 3 .
```

Both are 0 for a Gaussian. For our ten assets the skewness is negative (big down days are bigger than big up days) and the excess kurtosis ranges from 5 (AAPL) to 27 (CVX).

**What it means:** returns have **fat tails**: extreme days are far more common than a Gaussian predicts. A Gaussian puts a 5σ day at probability $5.7 \times 10^{-7}$, about once in 7,000 years of trading; SPY had nine of them in 2010–2024 (one in August 2011, eight in spring 2020). On a QQ-plot (data quantiles against Gaussian quantiles) the points bend away from the straight line at both ends.

> [!WARNING]
> A long/short book is hedged against *typical* co-movement. On extreme days correlations change and hedges break, so positions that are individually small can lose together. Gaussian risk estimates understate how often such days come by orders of magnitude.

<details>
<summary>The inverse cubic law, and why kurtosis estimates are unstable</summary>

The tails of stock returns decay roughly as a power law, $P(\lvert r\rvert \gt x) \propto x^{-\alpha}$ with $\alpha \approx 3$: the "inverse cubic law" (Gopikrishnan et al., 1998), a classic econophysics result. With $\alpha \lt 4$ the fourth moment formally diverges, so sample kurtosis is dominated by the few largest days and keeps growing as the sample grows. That is why the estimates range from 5 to 27 across similar stocks.

</details>

## Step 9. Go long, go short, and stay market-neutral

Being **long** means owning an asset: you profit if it rises. Being **short** means borrowing it and selling it, to buy it back later: you profit if it falls. Stat arb holds both at once. To see why, measure each asset's sensitivity to the market, its **beta** (the slope of its daily return regressed on SPY's):

```python
beta = R.cov()["SPY"] / R["SPY"].var()
```

KO and PEP have betas near 0.6, XOM 0.88, JPM about 1.2. A position one dollar long XOM and one dollar short KO has equal dollar sizes, but its net beta is $0.88 - 0.60 = +0.28$: it still gains when the market rises.

**What it means:** with $w_i$ the dollar weight of each position (negative for shorts):

- **dollar-neutral:** $\sum_i w_i = 0$ (long and short dollars equal);
- **beta-neutral**, or **market-neutral:** $\sum_i w_i \beta_i = 0$ (no net sensitivity to the market).

They are not the same. To be beta-neutral here you would short about 1.47 dollars of KO per dollar of XOM. Stat arb aims for both, so that profit comes from the *relative* move of the legs, not the market's direction.

> [!NOTE]
> *Physics analogy:* a differential measurement. Subtracting two signals with a large common-mode component (the market) leaves the small relative signal, as in a differential amplifier. The analogy breaks down because the "common-mode gain" (beta) is estimated with error and drifts, especially in crises.

<details>
<summary>Short selling, alpha, the risk-free rate and excess returns</summary>

- **Short selling in practice:** you borrow shares through a broker from a lender, sell them, and later buy them back to return them. You pay the lender any dividends and a **borrow fee**; your loss is unbounded if the price rises; the lender can **recall** the shares. Costs and risks of shorting are covered in Module 8.
- **Risk-free rate** $r_f$: the return on short-term government debt. **Excess return**: return minus $r_f$.
- **Notional**: the dollar size of a position.

**Benchmark, beta, alpha.** Regress an asset's excess return on the market's:

$$R_{i,t} - r_f = \alpha_i + \beta_i (R_{m,t} - r_f) + \varepsilon_{i,t}, \qquad \beta_i = \frac{\operatorname{Cov}(R_i, R_m)}{\operatorname{Var}(R_m)} .$$

The **benchmark** is the reference portfolio (the market, here SPY). **Alpha** $\alpha_i$ is the average return not explained by market exposure.

</details>

## Step 10. Exercises

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

Close back-adjusted for splits (÷k) and dividends (×(1 − D/P)) so that ratios are total returns. Unadjusted: fake crashes on split dates, a missing dividend yield (a large share of long-run return), wrong P&L on long/short pairs where one leg pays dividends. Know your vendor: Yahoo's `Close` is already split-adjusted. Adjusted history changes after each new event, so record the download date.

</details>

**3. How do you annualise volatility, and what assumption does $\sqrt{252}$ rely on?**

<details>
<summary>Key points</summary>

Multiply daily σ by $\sqrt{252}$, because variances of uncorrelated returns add (diffusion). It assumes zero autocorrelation and constant variance; it does not need Gaussianity. Autocorrelation adds covariance terms (negative autocorrelation means less long-horizon variance); volatility clustering means the daily σ itself varies.

</details>

**4. What does "market-neutral" mean, and is dollar-neutral the same as beta-neutral?**

<details>
<summary>Key points</summary>

Market-neutral: P&L insensitive to the market's direction, i.e. net beta ≈ 0. Dollar-neutral only equalises long and short dollars; if the legs have different betas the book still has market exposure (long 1 XOM / short 1 KO has net β ≈ +0.28). Beta-neutral sizes legs by beta. Betas are estimates and drift, so neutrality is approximate.

</details>

**5. Why are fat tails important for a strategy that holds offsetting long and short positions?**

<details>
<summary>Key points</summary>

Hedges are calibrated on typical days; extreme days are far more common than Gaussian models predict (SPY: nine 5σ days in 15 years versus one expected per ~7,000 years). In crises correlations shift and hedges break, so both legs can lose together, and leverage amplifies the loss. Kurtosis estimates are themselves unstable because the tails follow a power law.

</details>

</details>

<details>
<summary>Glossary of terms in this module</summary>

| Term | Meaning |
|---|---|
| Stock / share / equity; ticker | A unit of ownership in a company; its trading symbol (KO, SPY) |
| Bar, OHLCV | One period's open, high, low, close prices and volume |
| Index | A rule-based basket used as a benchmark (S&P 500); not directly tradable |
| ETF | Exchange-traded fund: a basket that trades like a single stock |
| Universe | The set of assets studied or traded |
| Bid / ask / spread / mid | Best price to sell at / to buy at / their gap / their average |
| Liquidity | How easily you can trade without moving the price |
| Market capitalisation | Share price × number of shares outstanding |
| Corporate action | Company event that changes the price but not holders' wealth |
| Stock split | Each share becomes $k$ shares at $1/k$ of the price |
| Dividend / ex-dividend date | Cash paid per share / first day a buyer no longer receives it |
| Adjusted close | Close back-adjusted for splits and dividends so its ratios are total returns |
| Price return / total return | Return from price changes only / including dividends |
| Simple return $R$ | $P_t/P_{t-1} - 1$; adds across assets |
| Log return $r$ | $\ln(P_t/P_{t-1})$; adds over time |
| Volatility | Standard deviation of returns, usually annualised |
| Annualise | Mean × 252, volatility × $\sqrt{252}$ |
| Volatility drag | Compound growth falls short of the average simple return by about $\sigma^2/2$ |
| Volatility clustering | Large moves tend to follow large moves |
| Skewness / excess kurtosis | Third standardised moment (asymmetry) / fourth minus 3 (tail weight) |
| Fat tails | Extreme moves far more frequent than a Gaussian predicts |
| Long / short | Own the asset / borrow and sell it |
| Beta / alpha | Sensitivity to the benchmark / return not explained by it |
| Dollar-neutral | Long and short dollars equal |
| Market-neutral, beta-neutral | Net sensitivity to the market is zero |
| Basis point (bp) | 0.01%, i.e. $10^{-4}$ |

</details>

<details>
<summary>References and further reading</summary>

- Harris, *Trading and Exchanges* (2003), Chapters 1–4 (skim): market structure, orders, spreads.
- Tsay, *Analysis of Financial Time Series* (3rd ed., 2010), Chapter 1: returns and their distributional properties.
- Bouchaud & Potters, *Theory of Financial Risk and Derivative Pricing* (2nd ed., 2003), Chapter 1: fat tails.
- Gopikrishnan, Meyer, Amaral & Stanley, "Inverse cubic law for the distribution of stock price variations", *European Physical Journal B* 3, 139 (1998).
- Lo & MacKinlay, "Stock market prices do not follow random walks: Evidence from a simple specification test", *Review of Financial Studies* 1(1) (1988).

</details>
