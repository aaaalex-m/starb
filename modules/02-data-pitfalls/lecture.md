# Module 2 — Data pitfalls

**Why this module:** most "too good to be true" backtests are data bugs, not discoveries. Interviewers probe this hard: "your backtest has a Sharpe ratio of 4, what do you check first?" This module covers the three big ones (survivorship bias, look-ahead bias, bad corporate-action data) plus the plumbing (calendars, missing and stale prices) and fixes the project's universes.

Notebook: [`notebook.ipynb`](notebook.ipynb). Code: [`statarb/universe.py`](../../src/statarb/universe.py), [`statarb/checks.py`](../../src/statarb/checks.py), `clean_prices` / `sanity_report` / `flag_returns` in [`statarb/data.py`](../../src/statarb/data.py).

---

## 1. Survivorship bias: a selection effect

**Survivorship bias:** studying only the assets that still exist today silently drops the ones that went bankrupt, were acquired or were **delisted** (removed from an exchange). The surviving sample is not representative: survival is correlated with past returns.

The maths is conditioning. If survival $S$ is more likely after high returns, then

```math
\mathbb E[R \mid S] > \mathbb E[R].
```

A clean special case: if returns are Gaussian, $R \sim \mathcal N(\mu, \sigma^2)$, and only assets with $R > c$ survive, then

```math
\mathbb E[R \mid R > c] = \mu + \sigma\,\frac{\varphi(\alpha)}{1 - \Phi(\alpha)}, \qquad \alpha = \frac{c - \mu}{\sigma},
```

where $\varphi$ and $\Phi$ are the standard normal density and distribution function. The bias grows with volatility $\sigma$ and with how harsh the cut is.

*Physics analogy:* Malmquist bias. A flux-limited survey only sees stars bright enough to detect, so the sample's mean luminosity overstates the population's. Any detector with an acceptance cut has the same problem.

**How big is it?**

| Setting (notebook) | Annualised return |
|---|---|
| Simulated stocks with true expected return 0: all stocks | ≈ 0% |
| ... survivors only (29% were delisted) | ≈ +3% |
| Equal-weight S&P 500, 2016–2025, *today's* members | 18.1% |
| ... point-in-time members that still have Yahoo data | 13.0% |
| ... RSP, the real equal-weight ETF (true point-in-time) | 11.6% |

The point-in-time portfolio closes most of the gap but not all, because about 20% of the 2016 members have no Yahoo data at all. They are exactly the ones that later disappeared, often after large losses. The final **delisting return** (from the last trade to what shareholders actually received) is frequently missing from databases. For delistings caused by poor performance it averages around −30% (Shumway, 1997).

**Related biases:**

- **Index reconstitution:** index providers periodically add and remove members (about 20–30 S&P 500 changes a year). Companies join after growing and leave after shrinking or being acquired. About 30% of today's S&P 500 wasn't in it ten years ago.
- **Backfill bias:** when a series enters a database, its earlier history is often added retroactively, and only for series that survived long enough to be added (common in hedge-fund databases; Amcor's pre-listing price history on Yahoo is a stock-market example).
- **Ticker change:** the same company under a new symbol (FB → META in 2022). A naive join by ticker splits one company into two or loses its history.
- **Recycled ticker:** an old symbol reused by a different company. Asking Yahoo for STI (SunTrust Banks, gone since 2019) returns an unrelated small stock with +290% days.

## 2. Point-in-time data

**Point-in-time data** is data exactly as it was known on each historical date: index membership on that date, prices including later-delisted securities, and company fundamentals as originally reported rather than as later restated.

It is expensive because someone has to have recorded it at the time, every day, and kept the losers. Commercial sources: CRSP (US prices including delisted stocks and delisting returns), Compustat Point-in-Time, index providers' historical constituent files.

Our free substitute rebuilds S&P 500 membership from Wikipedia: start with today's list and undo each later change. Let $M_t$ be the member set on date $t$ and let each change $c$ (effective on date $d_c$) add a ticker $a_c$ and remove a ticker $r_c$. Then

```math
M_t = \Big(M_{\text{today}} \setminus \{a_c : d_c > t\}\Big) \cup \{r_c : d_c > t\} \qquad\text{(applied in reverse date order)}.
```

Both Wikipedia pages are pinned to fixed revisions for reproducibility. A built-in error check: the reconstructed count should stay near 503 (about 500 companies, some with two share classes). It does to within about 1% for the last decade, and it drifts upwards before about 2008, where the changes table is incomplete.

## 3. Look-ahead bias

**Look-ahead bias:** using information in a backtest that wasn't available at decision time. Formally, the position decided at time $t$ must depend only on the information set $\mathcal F_t$ (everything observed up to $t$):

```math
w_t = f(\text{data}_{1}, \dots, \text{data}_{t}), \qquad \text{P\&L}_{t+1} = w_t \cdot R_{t+1}.
```

*Physics analogy:* causality. A physical filter's response kernel vanishes for future times (the retarded, not the advanced, Green's function). A signal processor that peeks at future samples isn't realisable.

**Common forms, from blatant to subtle:**

1. **Same-bar execution:** deciding with today's close and trading at today's close. In the notebook, "long after an up day" scores a Sharpe ratio of 13 this way and −0.35 honestly.
2. **Full-sample statistics:** normalising with the whole sample's mean and standard deviation, or fitting hedge ratios and thresholds on all the data. A mean-reversion rule using a full-sample z-score earns a Sharpe ratio of about 0.8 on *pure random walks* (positive on 96% of paths). The full-sample mean is a level the path is guaranteed to come back to, because it is computed from where the path went.
3. **Centred windows, back-filling and interpolation:** each uses later prices to set today's value.
4. **Hindsight universes:** choosing assets using today's knowledge. Survivorship bias is a special case.
5. **Revised data:** vendor history that changed after the fact (restated earnings, Yahoo's revised prices).
6. **Timing details:** index changes are announced days before they take effect; earnings are often released after the close. Use the date the information became public.
7. **Smoothed estimates:** a Kalman smoother or HMM smoother uses the future (Modules 9 and E1).

**The truncation test** (`checks.lookahead_violations`). A signal $f$ is causal if computing it on the full history and on the history cut at any $t$ gives the same values up to $t$:

```math
f(x_{1:T})_{1:t} = f(x_{1:t}) \quad \text{for all } t \le T.
```

Equivalently, changing future data must not change past signals. The test is cheap and generic, so we run it on every signal. It caught a real bug while this module was written: forward-filling only gaps *between* two prices (pandas' `limit_area="inside"`) leaks, because on the day a price goes missing you can't know whether another will ever arrive.

## 4. Calendars, missing data and stale prices

- **Trading calendar:** the days an exchange is open. Holidays differ across markets (Martin Luther King Day closes New York, not Tokyo or London). There are also **half-days** (early closes, e.g. the day after US Thanksgiving).
- **Missing data:** after aligning assets on one date index, drop dates on which nothing traded, then **forward-fill** short gaps (carry the last price forward). Forward-filling is causal; back-filling and interpolation are not. Each filled day is a zero return, so the project caps fills at 3 days (`data.clean_prices(max_ffill=3)`).
- **Stale price:** a price that didn't come from a trade that day, typically a long run of identical prices with zero volume (Amcor before its 2019 US listing). Stale prices lower measured volatility and create fake autocorrelation.
- **Nonsynchronous trading:** "daily" returns from different markets cover different hours. Tokyo closes about 14 hours before New York, so the Nikkei's correlation with SPY is 0.17 on the same date but 0.42 against the previous US day. Stale or asynchronous prices create spurious lead–lag relations that look like predictability but can't be traded (Lo & MacKinlay, 1990).

## 5. Vendor quirks and sanity checks

Yahoo Finance (free, and typical of free sources) has known quirks:

- history is revised without notice, and `adj_close` levels change after every new dividend;
- delisted securities vanish (a sixth of the tickers we needed returned nothing);
- tickers are recycled;
- corporate actions are sometimes mis-adjusted, especially **spin-offs** (a company distributing shares of a subsidiary to its shareholders). Danaher's 2016 spin-off of Fortive was recorded as both a dividend and a split, giving a fake +61% day;
- pre-listing histories can be stale;
- occasional bad ticks.

Defences used in this project: cache every download with its download date; run `data.sanity_report` (gaps, flat runs, zero-volume days, non-positive prices, largest moves) and `data.flag_returns` (daily moves beyond ±50%) on every new universe; **explain every flag** before trusting the data. A 50% daily move in a large cap is far more often a data error, a recycled ticker or a takeover than a real trading opportunity.

## 6. The project's universes

| Universe | Contents | Known biases | Used in |
|---|---|---|---|
| **U1** | SPY, 11 sector ETFs, classic pairs (KO/PEP, XOM/CVX, EWA/EWC, GLD/GDX, V/MA, HD/LOW) | Low survivorship risk; XLRE and XLC have short histories | Modules 4–10 |
| **U2** | `LARGE_CAP_100`: 100 large S&P 500 members as of November 2026 | Survivorship-biased by construction (chosen because they are large today) | Modules 5, 11 |
| **U3** | Current S&P 500 plus sector ETFs, optionally filtered by point-in-time membership | Delisted members largely missing from Yahoo; reconstruction unreliable before about 2008 | Modules 11–13 |

Results on U2 and U3 must be read with these biases stated. The honest benchmark is always something investable at the time (here, RSP or SPY).

---

## Glossary

| Term | Meaning |
|---|---|
| Universe | The set of assets a strategy may trade |
| Survivorship bias | Testing only on assets that still exist today, dropping those that failed or disappeared |
| Selection bias | Any distortion from a sample chosen in a way correlated with the outcome |
| Delisting | Removal from an exchange (bankruptcy, merger, takeover, failing listing rules) |
| Delisting return | The return from the last trade to what holders finally received; often missing |
| Look-ahead bias | Using information not available at decision time |
| Point-in-time data | Data exactly as known on each historical date |
| Index reconstitution | Periodic changes to an index's membership |
| Backfill bias | History added retroactively when a series enters a database, only for survivors |
| Trading calendar | The days (and hours) an exchange is open, including holidays and half-days |
| Stale price | A price that didn't come from a trade that day |
| Forward fill | Carrying the last known value forward over a gap (causal) |
| Nonsynchronous trading | Prices from different markets or assets recorded at different times |
| Ticker change | The same company trading under a new symbol (FB → META) |
| Recycled ticker | An old symbol reused by a different company |
| Spin-off | A company distributing shares of a subsidiary to its own shareholders |
| GICS | Global Industry Classification Standard: the sector labels used by S&P and MSCI |
| Equal-weight portfolio | Same dollar amount in every asset, rebalanced back to equal weights |
| Truncation test | Look-ahead check: a causal signal's past values don't change when the future is removed |

## Interview questions

Write your own answer under each question first, then compare with the key points.

**1. Define survivorship bias and look-ahead bias, and give a concrete example of each from your project.**

<details><summary>Key points</summary>

Survivorship: conditioning on survival inflates returns. Example: an equal-weight portfolio of today's S&P 500 members earned 18.1% a year over 2016–2025, against 11.6% for the real equal-weight ETF; even with point-in-time membership, about 20% of 2016 members have no free data. Look-ahead: a decision uses information from after the decision time. Example: a full-sample z-score mean-reversion rule earns a Sharpe ratio of about 0.8 on pure random walks. Survivorship is itself a form of look-ahead (a hindsight universe).
</details>

**2. What is point-in-time data, and why is it expensive?**

<details><summary>Key points</summary>

Data as it was known on each date: membership, prices including dead securities and delisting returns, fundamentals as first reported (not restated). It must be recorded at the time and maintained forever, losers included. It can't be reconstructed fully after the fact, as our Wikipedia reconstruction shows (good for about a decade, then it drifts). Vendors: CRSP, Compustat Point-in-Time.
</details>

**3. Your backtest has a Sharpe ratio of 4. What data problems do you check first?**

<details><summary>Key points</summary>

Timing: is the signal from day $t$ traded at $t+1$? Run a truncation test. Full-sample fitting or normalisation. Universe: chosen with hindsight? Delisted names included? Prices: adjusted for splits and dividends, and are the adjustments right (spin-offs)? Outliers: any 50% days, recycled tickers, stale prices creating fake mean reversion? Then costs and capacity (Module 8). Only after all that consider that it might be real.
</details>

**4. How did you handle missing prices and different trading calendars?**

<details><summary>Key points</summary>

Align on the union of dates; drop dates when nothing traded; forward-fill at most 3 days (causal, unlike back-fill or interpolation); leave longer gaps missing so no fake flat prices are invented. Watch for nonsynchronous closes across time zones (the Nikkei vs SPY correlation is 0.17 same day, 0.42 lagged), and for assets with short histories, which need an explicit rule.
</details>

**5. How would you test automatically that a strategy has no look-ahead?**

<details><summary>Key points</summary>

Truncation test: compute signals on the full data and on data cut at many dates $t$; values up to $t$ must match. Equivalently, perturbing future data must not change past signals. It catches full-sample statistics, centred windows, `shift(-1)`, back-filling, and subtle cases such as filling only "inside" gaps. Passing is evidence, not proof (it can't detect revised vendor data or a hindsight universe).
</details>

## References

- Brown, Goetzmann, Ibbotson & Ross, "Survivorship bias in performance studies", *Review of Financial Studies* 5(4) (1992).
- Shumway, "The delisting bias in CRSP data", *Journal of Finance* 52(1) (1997).
- Lo & MacKinlay, "An econometric analysis of nonsynchronous trading", *Journal of Econometrics* 45 (1990).
- López de Prado, *Advances in Financial Machine Learning* (2018), Chapter 2 (financial data structures) and his "seven sins of quantitative investing".
