"""Trading signals: turning a standardised spread into positions.

Positions are in *spread units*: +1 means long the spread (long the first leg, short the
second), -1 short the spread, 0 flat. A position decided on day t is earned from day t+1
(see ``config``: a signal at t is traded at t+1).
"""

import numpy as np
import pandas as pd

PandasData = pd.Series | pd.DataFrame


def _threshold_path(z: np.ndarray, entry: float, exit_: float) -> np.ndarray:
    position = np.zeros(len(z))
    state = 0.0
    for t, value in enumerate(z):
        if np.isnan(value):
            state = 0.0  # no information: close
        elif state == 0.0:
            if value > entry:
                state = -1.0  # spread too high: short it
            elif value < -entry:
                state = 1.0  # spread too low: buy it
        elif state == -1.0 and value <= exit_:
            state = 0.0
        elif state == 1.0 and value >= -exit_:
            state = 0.0
        position[t] = state
    return position


def threshold_positions(z: PandasData, entry: float = 2.0, exit: float = 0.0) -> PandasData:
    """Two-state threshold rule with hysteresis (a Schmitt trigger).

    * Flat: open short (-1) when ``z > entry``, open long (+1) when ``z < -entry``.
    * Short: close when ``z <= exit``. Long: close when ``z >= -exit``.
    * A missing ``z`` closes the position.

    With ``exit = 0`` a position is closed when the spread crosses its mean. The gap between
    ``entry`` and ``exit`` stops the position flickering on and off around one level. At most
    one transition happens per step, so a position closed on day t can reopen on day t+1 at
    the earliest. Uses only past and present values of ``z``, so it is causal.
    """
    if not 0 <= exit < entry:
        raise ValueError(f"need 0 <= exit < entry, got exit={exit}, entry={entry}")
    if isinstance(z, pd.DataFrame):
        return z.apply(lambda col: threshold_positions(col, entry, exit))
    values = _threshold_path(np.asarray(z, dtype=float), entry, exit)
    return pd.Series(values, index=z.index, name=z.name)
