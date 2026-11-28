"""Automated checks against look-ahead bias.

A trading signal is *causal* if its value on day t depends only on data up to day t. The
truncation test checks this directly: compute the signal on the full history and on the
history cut off at day t; on every date up to t the two must agree. If they don't, the
full-history signal used something from after t.

Typical bugs this catches: full-sample means or standard deviations, centred rolling windows,
``shift(-1)``, back-filling or interpolating missing prices, and parameters fitted on the
whole sample.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

PandasData = pd.Series | pd.DataFrame


def lookahead_violations(
    signal_fn: Callable[[PandasData], PandasData],
    data: PandasData,
    n_checks: int = 10,
    min_history: int = 20,
    atol: float = 1e-10,
) -> list[pd.Timestamp]:
    """Cut-off dates at which ``signal_fn`` changes its past values when given less future.

    ``signal_fn`` maps a time-indexed Series or DataFrame to a signal on the same index.
    ``n_checks`` cut-off dates are spread evenly between ``min_history`` and the end of the
    sample. An empty list means no look-ahead was detected (which is evidence, not proof).
    """
    full = signal_fn(data)
    positions = np.linspace(min_history, len(data) - 2, n_checks).astype(int)
    violations = []
    for cut in data.index[np.unique(positions)]:
        truncated = signal_fn(data.loc[:cut])
        past_full = full.loc[:cut]
        if past_full.shape != truncated.shape or not np.allclose(
            np.asarray(past_full, dtype=float),
            np.asarray(truncated, dtype=float),
            atol=atol,
            equal_nan=True,
        ):
            violations.append(cut)
    return violations


def assert_no_lookahead(
    signal_fn: Callable[[PandasData], PandasData], data: PandasData, **kwargs
) -> None:
    """Raise ``AssertionError`` if :func:`lookahead_violations` finds any violation."""
    violations = lookahead_violations(signal_fn, data, **kwargs)
    if violations:
        dates = ", ".join(str(d.date()) for d in violations[:3])
        raise AssertionError(f"signal changes past values when future data is removed: {dates}")
