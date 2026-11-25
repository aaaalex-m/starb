# Module 2 — Data pitfalls

Most "too good to be true" backtests are data bugs, not discoveries, and interviewers probe this hard: "your backtest has a Sharpe ratio of 4; what do you check first?" In this module you rebuild the S&P 500 as it was on past dates, measure how much a hindsight universe flatters returns, fake two profits with look-ahead bias and then catch them automatically, and clean and audit real price data. At the end you fix the project's three universes.

Notebook: [`notebook.ipynb`](notebook.ipynb). Code: [`statarb/universe.py`](../../src/statarb/universe.py), [`statarb/checks.py`](../../src/statarb/checks.py), and `clean_prices`, `sanity_report`, `flag_returns` in [`statarb/data.py`](../../src/statarb/data.py).

> [!TIP]
> **How to read these notes.** The numbered steps are the main path. Boxes are optional: notes and warnings are short asides; collapsed sections hold derivations, extra jargon and interview-level detail.

## Step 1. Rebuild the S&P 500 on any past date

```python
current, changes = universe.load_sp500_tables()      # 503 members today, 415 recorded changes
universe.sp500_members_asof("2016-01-04", current, changes)
```

The index's membership changes over time: companies are added when they grow and removed when they shrink, are acquired or fail (**index reconstitution**, about 20–30 changes a year). `members_asof` starts from today's list and undoes every later change: it drops what was added and restores what was removed.

**What it means:** this gives **point-in-time** membership, the index exactly as it was known on each date. As a built-in error check, the reconstructed count should stay near 503 (about 500 companies, some with two share classes). It does to within about 1% for the last decade, then drifts up to about 516 before 2007, where Wikipedia's changes table is incomplete.

<details>
<summary>The reconstruction formula, pinned sources, and why point-in-time data is expensive</summary>

Let $M_t$ be the member set on date $t$, and let each change $c$, effective on date $d_c$, add a ticker $a_c$ and remove a ticker $r_c$. Undoing the changes in reverse date order gives

$$M_t = \Big(M_{\text{today}} \setminus \lbrace a_c : d_c \gt t\rbrace\Big) \cup \lbrace r_c : d_c \gt t\rbrace .$$

Both Wikipedia pages (today's constituents and the historical changes) are pinned to fixed revisions, so the reconstruction is reproducible.

**Point-in-time data** in general means everything as it was known on each date: membership, prices including companies that later disappeared, and company accounts as first reported rather than later restated. It is expensive because someone had to record it at the time, every day, and keep the losers. Commercial sources: CRSP (US prices including delisted stocks), Compustat Point-in-Time, and index providers' historical constituent files.

</details>

## Step 2. Count how much the index changed

```
members today that were not members ten years ago: 145
members ten years ago that are gone today:          150
```

Of the changes in the last ten years, 97 were market-capitalisation changes, 88 acquisitions or mergers and 25 spin-offs.

**What it means:** about 30% of today's index wasn't in it ten years ago. Companies join *because* they did well. Any study that starts from today's list silently selects the winners.

## Step 3. Simulate survivorship bias, where the truth is known

Simulate 2,000 stocks for ten years as random walks with **zero expected return**. A stock that falls below 20% of its starting price is **delisted** (removed from the exchange, think bankruptcy) and frozen there.

| | stocks | annualised return |
|---|---|---|
| all stocks | 2000 | ≈ 0.1% |
| survivors only | 1418 | ≈ 3.0% |

**What it means:** 29% of stocks were delisted, and the survivors show +3% a year, purely because the losers were removed from the sample. This is **survivorship bias**: studying only assets that still exist today drops the ones that failed, and survival is correlated with past returns.

> [!NOTE]
> *Physics analogy:* Malmquist bias. A flux-limited survey only sees stars bright enough to detect, so the sample's mean luminosity overstates the population's. Any detector with an acceptance cut has the same problem.

<details>
<summary>The maths of a selection effect: conditioning on survival</summary>

If survival $S$ is more likely after high returns, then $\mathbb E[R \mid S] \gt \mathbb E[R]$. A clean special case: if returns are Gaussian, $R \sim \mathcal N(\mu, \sigma^2)$, and only assets with $R \gt c$ survive, then

$$\mathbb E[R \mid R \gt c] = \mu + \sigma \frac{\varphi(\alpha)}{1 - \Phi(\alpha)}, \qquad \alpha = \frac{c - \mu}{\sigma},$$

where $\varphi$ and $\Phi$ are the standard normal density and distribution function. The bias grows with the volatility $\sigma$ and with how harsh the cut is. The general term is **selection bias**: any distortion from a sample chosen in a way correlated with the outcome.

</details>

## Step 4. Measure survivorship bias in real data

Download every ticker that was an S&P 500 member at any month-start in 2016–2025 (724 tickers) and build three **equal-weight** portfolios (the same amount in every stock, rebalanced daily):

| Portfolio, 2016–2025 | Annualised return |
|---|---|
| today's 503 members, held for the whole decade | 18.1% |
| point-in-time members (as of the previous close) that have Yahoo data | 13.0% |
| RSP, the real S&P 500 equal-weight ETF | 11.6% |

**What it means:** today's members beat the real, investable index by about 6 percentage points a year, but nobody could have bought that portfolio in 2016. Point-in-time membership removes most of the gap. The rest remains because 123 of the 724 tickers return no data at all: about 20% of the 2016 members are missing, and they are exactly the companies that later disappeared.

> [!WARNING]
> Survivorship bias is a form of look-ahead: the universe was chosen with information from the future. The honest benchmark is always something investable at the time, here RSP or SPY.

> [!NOTE]
> The **delisting return** (from the last trade to what shareholders finally received) is often missing from databases. For delistings caused by poor performance it averages around −30% (Shumway, 1997).

## Step 5. Check for recycled tickers

Exchanges reuse ticker symbols. The notebook asks: did a ticker's Yahoo price history start only *after* its S&P 500 membership ended?

```
STI  member until 2019-12   Yahoo history starts 2022-05   max daily move +294%
SE   member until 2017-02   Yahoo history starts 2017-10
ADT  member until 2016-05   Yahoo history starts 2018-01
```

**What it means:** these are **recycled tickers**: an old symbol now belongs to a different company. STI was SunTrust Banks (merged into Truist in 2019); today's STI is an unrelated small stock. SE was Spectra Energy and is now Sea Ltd. A careless "download every ticker ever in the index" study would mix them in.

<details>
<summary>Related biases: backfill, ticker changes, recycled tickers</summary>

- **Backfill bias:** when a series enters a database, its earlier history is often added retroactively, and only for series that survived long enough to be added. Common in hedge-fund databases; Amcor's pre-listing price history on Yahoo is a stock-market example (Step 10).
- **Ticker change:** the same company under a new symbol (FB became META in 2022). A naive join by ticker splits one company into two, or loses its history.
- **Recycled ticker:** a different company under an old symbol. Checking that a ticker's price history overlaps the dates you think it was trading is a cheap defence.

</details>

## Step 6. Fake a profit by trading at the price you decided on

Rule: go long SPY after an up day, short after a down day. The signal uses day $t$'s close.

```
same-day (look-ahead): Sharpe  13.41
next-day (honest):     Sharpe  -0.35
```

**What it means:** trading at the same close you used to decide earns the day's absolute return every day, a Sharpe ratio no real strategy has ever had. This is **look-ahead bias**: using information that wasn't available when the decision was made. The project rule, signal at $t$, trade at $t+1$, prevents this blatant form.

> [!NOTE]
> The **Sharpe ratio** is the average return divided by its standard deviation, annualised with $\sqrt{252}$. It measures return per unit of risk; Module 7 covers it properly.

## Step 7. Fake a profit with a subtle leak

A common pattern: standardise a series, $z_t = (x_t - \bar x)/s_x$, and bet on reversion (short when $z_t \gt 0$, long when $z_t \lt 0$). Run it on 1,000 *pure random walks*, which are unpredictable by construction, trading correctly at $t+1$ in both versions:

| Normalisation | mean Sharpe | share of paths with Sharpe above 0 |
|---|---|---|
| full-sample mean and standard deviation (leaky) | 0.77 | 96% |
| expanding window, past data only (causal) | 0.02 | 51% |

**What it means:** the only difference is that $\bar x$ and $s_x$ were computed with the whole sample. The full-sample mean is a level the path is guaranteed to come back to, because it was computed from where the path went. Fitting hedge ratios or thresholds on all the data, as naive pairs trading does, leaks in the same way.

<details>
<summary>Look-ahead bias formally, and seven common forms</summary>

The position decided at time $t$ may depend only on the information set $\mathcal F_t$, everything observed up to $t$. With $x_s$ the data observed on day $s$ and $\Pi$ the profit and loss (P&L):

$$w_t = f(x_1, x_2, \dots, x_t), \qquad \Pi_{t+1} = w_t R_{t+1}.$$

*Physics analogy:* causality. A physical filter's response kernel vanishes for future times (the retarded, not the advanced, Green's function); a signal processor that peeks at future samples isn't realisable.

Common forms, from blatant to subtle:

1. **Same-bar execution:** deciding with today's close and trading at today's close (Step 6).
2. **Full-sample statistics:** normalisation, hedge ratios or thresholds fitted on all the data (Step 7).
3. **Centred windows, back-filling and interpolation:** each uses later prices to set today's value.
4. **Hindsight universes:** survivorship bias (Steps 3–4).
5. **Revised data:** vendor history that changed after the fact (restated earnings, revised prices).
6. **Timing details:** index changes are announced days before they take effect; earnings are often released after the close. Use the date the information became public.
7. **Smoothed estimates:** a Kalman or HMM smoother uses the future (Modules 9 and E1).

</details>

## Step 8. Catch look-ahead automatically

`checks.lookahead_violations` runs the **truncation test**: compute the signal on the full history and on the history cut at day $t$. For a causal signal the values up to $t$ must be identical:

```math
f(x_{1:T})_{1:t} = f(x_{1:t}) \quad \text{for all } t \le T.
```

```
full-sample z-score    LOOK-AHEAD
expanding z-score      ok
rolling 20-day mean    ok
centred rolling mean   LOOK-AHEAD
```

**What it means:** equivalently, changing future data must not change past signals. The test is cheap and generic, so every signal in this project gets it. Passing is evidence, not proof: it can't detect revised vendor data or a hindsight universe.

> [!TIP]
> It caught a real bug while this module was being written. The first `data.clean_prices` only forward-filled gaps *between* two prices (pandas' `limit_area="inside"`). On the day a price goes missing you can't know whether another will ever arrive, so that choice uses the future.

## Step 9. Align trading calendars and fill short gaps

Put SPY, Japan's Nikkei 225 and the UK's FTSE 100 on one date index. Exchanges have different holidays (a **trading calendar** is the set of days and hours an exchange is open):

| | SPY | Nikkei 225 | FTSE 100 |
|---|---|---|---|
| missing days, raw | 86 | 156 | 75 |
| after `data.clean_prices(max_ffill=3)` | 0 | 3 | 0 |

`clean_prices` drops dates on which nothing traded, then **forward-fills** gaps of up to three days (carries the last price forward).

**What it means:** forward-filling is causal, because it only uses the past. Every filled day is a zero return, so long gaps (here a few Japanese Golden Week days) are deliberately left missing rather than filled with an invented flat price.

> [!WARNING]
> Never back-fill or interpolate prices in a backtest: both use later prices to fill earlier gaps, which is look-ahead.

## Step 10. Beware prices that weren't traded

**Same date is not same time.** Tokyo closes about 14 hours before New York, so a "daily" return dated $t$ covers different hours in each market:

```
              same date   SPY day t vs index day t+1
Nikkei 225     0.17          0.42
FTSE 100       0.49          0.15
```

**Stale prices.** Amcor (AMCR) joined the US market in 2019, but Yahoo shows an earlier history with 669 zero-volume days and a 66-day run of identical prices.

**What it means:** **nonsynchronous trading** (prices recorded at different times) creates lead–lag relations that look like predictability but can't be traded: by Tokyo's next open the news is already in the price. A **stale price** (one that didn't come from a trade that day) lowers measured volatility and creates fake autocorrelation, which can look like mean reversion. `data.sanity_report` counts flat runs and zero-volume days for every ticker.

## Step 11. Explain every big move

```python
flags = data.flag_returns(adj, threshold=0.5)   # 46 daily moves beyond ±50%, in 25 tickers
```

Each flag needs an explanation before the data is trusted. In our download there are three kinds:

- **real events:** GameStop's January 2021 short squeeze, PG&E's 2019 bankruptcy filing, the oil crash of 9 March 2020;
- **a different company under a recycled ticker:** STI (Step 5);
- **a bad adjustment:** Danaher's 2016 **spin-off** of Fortive (a company distributing shares of a subsidiary to its shareholders) was recorded by Yahoo as both a dividend and a split. The raw close rose 3.6% that day; the adjusted close shows +61%.

**What it means:** a 50% daily move in a large company is far more often a data error, a recycled ticker or a takeover than a trading opportunity. Every DHR price before July 2016 is wrong by a factor of about 1.5, and any backtest holding DHR through that date books a fake 61% gain.

<details>
<summary>Yahoo Finance quirks, and the project's defences</summary>

Yahoo is free and typical of free sources:

- history is revised without notice, and `adj_close` levels change after every new dividend;
- delisted securities vanish (a sixth of the tickers we needed returned nothing);
- tickers are recycled;
- corporate actions are sometimes mis-adjusted, especially spin-offs;
- pre-listing histories can be stale;
- there are occasional bad ticks.

Defences: cache every download with its download date; run `data.sanity_report` (gaps, flat runs, zero-volume days, non-positive prices, largest moves) and `data.flag_returns` on every new universe; explain every flag before trusting the data.

</details>

## Step 12. Choose the project's universes

| Universe | Contents | Known biases | Used in |
|---|---|---|---|
| **U1** | SPY, 11 sector ETFs, classic pairs (KO/PEP, XOM/CVX, EWA/EWC, GLD/GDX, V/MA, HD/LOW) | Low survivorship risk; XLRE and XLC have short histories | Modules 4–10 |
| **U2** | `LARGE_CAP_100`: 100 large S&P 500 members as of November 2026 | Survivorship-biased by construction: chosen because they are large today | Modules 5, 11 |
| **U3** | Current S&P 500 plus sector ETFs, optionally filtered by point-in-time membership | Delisted members largely missing from Yahoo; reconstruction unreliable before about 2008 | Modules 11–13 |

**What it means:** every result on U2 and U3 must be reported with these biases stated. U1 is mostly ETFs and long-lived large companies, so it is the safest place to start trading pairs (Module 4).

## Step 13. Exercises

Do these in the empty cells at the end of the notebook. Your answers aren't written here.

1. **No-look-ahead test.** Read `statarb/checks.py` and `tests/test_checks.py`. Write a signal with a subtle leak of your own (for example a z-score using `rolling(..., center=True)`, or prices cleaned with `bfill`) and confirm the truncation test catches it. Explain in a paragraph why "changing future data must not change past signals" is equivalent to the truncation test.
2. **A ticker change and a delisting.** Find one ticker change (e.g. FB to META) and one delisting in U2's history. For each, describe exactly how it would corrupt a naive backtest.
3. **Short histories.** Quantify the missing data for XLRE and XLC (launched 2015 and 2018). Decide, and write down, how the project handles assets with short histories.

**Done when:** the README has a "Known data limitations" section, and the no-look-ahead test exists (`tests/test_checks.py`).

<details>
<summary>Interview questions (5), with key points</summary>

Write your own answer first, then open the key points.

**1. Define survivorship bias and look-ahead bias, and give a concrete example of each from your project.**

<details>
<summary>Key points</summary>

Survivorship: conditioning on survival inflates returns. Example: an equal-weight portfolio of today's S&P 500 members earned 18.1% a year over 2016–2025, against 11.6% for the real equal-weight ETF; even with point-in-time membership, about 20% of 2016 members have no free data. Look-ahead: a decision uses information from after the decision time. Example: a full-sample z-score mean-reversion rule earns a Sharpe ratio of about 0.8 on pure random walks. Survivorship is itself a form of look-ahead (a hindsight universe).

</details>

**2. What is point-in-time data, and why is it expensive?**

<details>
<summary>Key points</summary>

Data as it was known on each date: membership, prices including dead securities and delisting returns, fundamentals as first reported (not restated). It must be recorded at the time and maintained forever, losers included. It can't be fully reconstructed after the fact, as our Wikipedia reconstruction shows (good for about a decade, then it drifts). Vendors: CRSP, Compustat Point-in-Time.

</details>

**3. Your backtest has a Sharpe ratio of 4. What data problems do you check first?**

<details>
<summary>Key points</summary>

Timing: is the signal from day $t$ traded at $t+1$? Run a truncation test. Full-sample fitting or normalisation. Universe: chosen with hindsight? Delisted names included? Prices: adjusted for splits and dividends, and are the adjustments right (spin-offs)? Outliers: any 50% days, recycled tickers, stale prices creating fake mean reversion? Then costs and capacity (Module 8). Only after all that consider that it might be real.

</details>

**4. How did you handle missing prices and different trading calendars?**

<details>
<summary>Key points</summary>

Align on the union of dates; drop dates when nothing traded; forward-fill at most 3 days (causal, unlike back-fill or interpolation); leave longer gaps missing so no fake flat prices are invented. Watch for nonsynchronous closes across time zones (Nikkei vs SPY correlation 0.17 same day, 0.42 lagged), and for assets with short histories, which need an explicit rule.

</details>

**5. How would you test automatically that a strategy has no look-ahead?**

<details>
<summary>Key points</summary>

Truncation test: compute signals on the full data and on data cut at many dates $t$; values up to $t$ must match. Equivalently, perturbing future data must not change past signals. It catches full-sample statistics, centred windows, `shift(-1)`, back-filling, and subtle cases such as filling only "inside" gaps. Passing is evidence, not proof (it can't detect revised vendor data or a hindsight universe).

</details>

</details>

<details>
<summary>Glossary of terms in this module</summary>

| Term | Meaning |
|---|---|
| Universe | The set of assets a strategy may trade |
| Index reconstitution | Periodic changes to an index's membership |
| Point-in-time data | Data exactly as known on each historical date |
| Survivorship bias | Testing only on assets that still exist today, dropping those that failed or disappeared |
| Selection bias | Any distortion from a sample chosen in a way correlated with the outcome |
| Delisting / delisting return | Removal from an exchange / the return from the last trade to what holders finally received |
| Equal-weight portfolio | The same dollar amount in every asset, rebalanced back to equal weights |
| Backfill bias | History added retroactively when a series enters a database, only for survivors |
| Ticker change / recycled ticker | Same company under a new symbol / a different company under an old symbol |
| Look-ahead bias | Using information not available at decision time |
| Truncation test | A causal signal's past values don't change when the future is removed |
| Trading calendar | The days and hours an exchange is open, including holidays and half-days |
| Forward fill | Carrying the last known price forward over a gap (causal) |
| Nonsynchronous trading | Prices of different assets or markets recorded at different times |
| Stale price | A price that didn't come from a trade that day |
| Spin-off | A company distributing shares of a subsidiary to its own shareholders |
| GICS | Global Industry Classification Standard: the sector labels used by S&P and MSCI |
| Sharpe ratio | Average return over its standard deviation, annualised (Module 7) |

</details>

<details>
<summary>References and further reading</summary>

- Brown, Goetzmann, Ibbotson & Ross, "Survivorship bias in performance studies", *Review of Financial Studies* 5(4) (1992).
- Shumway, "The delisting bias in CRSP data", *Journal of Finance* 52(1) (1997).
- Lo & MacKinlay, "An econometric analysis of nonsynchronous trading", *Journal of Econometrics* 45 (1990).
- López de Prado, *Advances in Financial Machine Learning* (2018), Chapter 2, and his "seven sins of quantitative investing".

</details>
