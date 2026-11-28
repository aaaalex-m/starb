r"""Distance-method pairs trading (Gatev, Goetzmann & Rouwenhorst, 2006; "GGR").

1. **Formation period** (e.g. 12 months): rescale every price to start at 1, measure the
   distance between two stocks as the sum of squared deviations (SSD) of their normalised
   prices, and select the pairs with the smallest distance.
2. **Trading period** (the following e.g. 6 months): re-normalise both legs at the start,
   watch the spread (the gap between the normalised prices), open when it exceeds ``entry``
   formation-period standard deviations, and close when it crosses zero or the period ends.

Prices should be adjusted closes (total-return indices). The P&L here is deliberately naive
(daily-rebalanced legs, no costs); Module 6 replaces it with a proper backtester.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from statarb.signals import threshold_positions


def normalise(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Rescale each price series to start at 1 (its cumulative total-return index)."""
    if isinstance(prices, pd.Series):
        return prices / prices.loc[prices.first_valid_index()]
    return prices.apply(normalise)


def ssd_matrix(normalised: pd.DataFrame) -> pd.DataFrame:
    r"""Sum of squared deviations :math:`\sum_t (p_{i,t} - p_{j,t})^2` for every pair.

    Symmetric, with NaN on the diagonal. The input must have no missing values.
    """
    if normalised.isna().any().any():
        raise ValueError("normalised prices contain NaN; drop incomplete columns first")
    x = normalised.to_numpy()
    gram = x.T @ x
    sq = np.diag(gram)
    ssd = sq[:, None] + sq[None, :] - 2 * gram
    ssd = np.clip(ssd, 0.0, None)
    np.fill_diagonal(ssd, np.nan)
    return pd.DataFrame(ssd, index=normalised.columns, columns=normalised.columns)


def select_pairs(prices: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """The ``n`` pairs with the smallest SSD over the window ``prices``.

    Tickers with any missing price in the window are dropped (GGR require a full formation
    history). A ticker may appear in several pairs. Returns columns ``a``, ``b``, ``ssd``,
    sorted by distance, with ``a`` before ``b`` alphabetically.
    """
    complete = prices.dropna(axis=1, how="any")
    complete = complete[sorted(complete.columns)]
    ssd = ssd_matrix(normalise(complete)).to_numpy()
    i, j = np.triu_indices(len(complete.columns), k=1)
    tickers = complete.columns.to_numpy()
    pairs = pd.DataFrame({"a": tickers[i], "b": tickers[j], "ssd": ssd[i, j]})
    return pairs.sort_values("ssd", kind="stable").head(n).reset_index(drop=True)


def spread(prices: pd.DataFrame, a: str, b: str) -> pd.Series:
    """Normalised price of ``a`` minus normalised price of ``b`` (both start at 1)."""
    norm = normalise(prices[[a, b]])
    return (norm[a] - norm[b]).rename(f"{a}-{b}")


def pair_returns(
    position: pd.Series, returns_a: pd.Series, returns_b: pd.Series, lag: int = 1
) -> pd.Series:
    r"""Naive daily return of a pair position: one dollar in each leg, rebalanced daily.

    ``position`` is in spread units (+1 = long ``a``, short ``b``). The position decided on
    day t earns the return of day t + ``lag``:
    :math:`r_t = s_{t-\text{lag}} (R_{a,t} - R_{b,t})`. ``lag=1`` trades at the close that
    produced the signal; ``lag=2`` waits one day, as GGR do.
    """
    earning = position.shift(lag).fillna(0.0)
    return (earning * (returns_a - returns_b)).fillna(0.0)


@dataclass
class GGRResult:
    """Output of :func:`ggr_backtest`."""

    committed: pd.Series
    """Daily return on committed capital: the average over all selected pairs (flat = 0)."""
    employed: pd.Series
    """Daily return on employed capital: the average over pairs with an open position."""
    open_pairs: pd.Series
    """Number of pairs earning a return each day."""
    pairs: pd.DataFrame
    """One row per selected pair and trading period (distance, formation sigma, trades)."""
    trades: pd.DataFrame
    """One row per round trip: open and close dates, direction, return, convergence."""


def _trades(pair_id: dict, position: pd.Series, daily: pd.Series, lag: int) -> list[dict]:
    """Round trips in one trading window (both series indexed by the window, flat at the end).

    A trade signalled on window day t and closed (position back to 0) on day e earns the
    returns of days t + lag to e + lag - 1.
    """
    pos, ret, index = position.to_numpy(), daily.to_numpy(), position.index
    rows, t, n = [], 0, len(pos)
    while t < n:
        if pos[t] == 0 or (t > 0 and pos[t - 1] == pos[t]):
            t += 1
            continue
        end = t
        while end < n and pos[end] == pos[t]:
            end += 1
        earned = ret[t + lag : min(end + lag, n)]
        rows.append(
            {
                **pair_id,
                "open": index[t],
                "close": index[min(end, n - 1)],
                "direction": int(pos[t]),
                "days": len(earned),
                "return": float(np.prod(1 + earned) - 1),
                "converged": bool(end < n - 1),
            }
        )
        t = end
    return rows


def ggr_backtest(
    prices: pd.DataFrame,
    formation: int = 252,
    trading: int = 126,
    n_pairs: int = 20,
    entry: float = 2.0,
    exit: float = 0.0,
    lag: int = 1,
) -> GGRResult:
    """Roll GGR formation and trading periods through ``prices`` (non-overlapping trading).

    Each trading period starts the day after a ``formation``-day window, selects
    ``n_pairs`` pairs on that window, and trades them for ``trading`` days. Both legs are
    re-normalised to 1 at the last formation close; the spread is divided by its
    formation-period standard deviation, the threshold rule gives positions, and every
    position is closed at the end of the period. Uses only data up to each decision date.
    """
    returns = prices.pct_change(fill_method=None)
    committed, employed, open_count = [], [], []
    pair_rows, trade_rows = [], []

    for start in range(formation, len(prices) - 1, trading):
        form = prices.iloc[start - formation : start]
        window = prices.iloc[start - 1 : start + trading]  # day 0 = last formation close
        days = window.index[1:]
        selected = select_pairs(form, n_pairs)
        pnl = pd.DataFrame(0.0, index=days, columns=range(len(selected)))
        earning = pd.DataFrame(0.0, index=days, columns=range(len(selected)))

        for k, (a, b, ssd) in enumerate(selected.itertuples(index=False)):
            sigma = spread(form, a, b).std()
            z = spread(window, a, b) / sigma
            position = threshold_positions(z, entry, exit)
            position.iloc[-1] = 0.0  # close everything at the end of the period
            daily = pair_returns(position, returns[a], returns[b], lag).loc[window.index]
            pnl[k] = daily.loc[days]
            earning[k] = position.shift(lag).fillna(0.0).loc[days].ne(0.0).astype(float)
            pair_id = {"period": days[0], "a": a, "b": b}
            found = _trades(pair_id, position, daily, lag)
            trade_rows += found
            pair_rows.append({**pair_id, "ssd": ssd, "sigma": sigma, "trades": len(found)})

        n_open = earning.sum(axis=1)
        committed.append(pnl.sum(axis=1) / max(len(selected), 1))
        employed.append((pnl.sum(axis=1) / n_open.where(n_open > 0)).fillna(0.0))
        open_count.append(n_open)

    trade_cols = ["period", "a", "b", "open", "close", "direction", "days", "return", "converged"]
    return GGRResult(
        committed=pd.concat(committed).rename("committed"),
        employed=pd.concat(employed).rename("employed"),
        open_pairs=pd.concat(open_count).rename("open_pairs"),
        pairs=pd.DataFrame(pair_rows),
        trades=pd.DataFrame(trade_rows, columns=trade_cols),
    )
