r"""Synthetic time series with known properties, for testing estimators before using real data.

Every function takes ``seed`` (an int or a ``numpy.random.Generator``) so results are
reproducible, and ``n_paths``: with ``n_paths=None`` a 1-D array of length ``n`` is returned,
otherwise an ``(n, n_paths)`` array with one independent path per column (an ensemble).

Processes (with i.i.d. Gaussian shocks :math:`\varepsilon_t \sim \mathcal N(0, \sigma^2)`):

* white noise: :math:`x_t = \varepsilon_t`
* random walk: :math:`x_t = x_{t-1} + \mu + \varepsilon_t`
* AR(1): :math:`x_t = c + \phi x_{t-1} + \varepsilon_t`
* cointegrated pair: :math:`x` a random walk, :math:`y_t = \alpha + \beta x_t + u_t` with
  :math:`u` an AR(1) spread.
"""

from typing import NamedTuple

import numpy as np
from scipy.signal import lfilter

Seed = int | np.random.Generator | None


def _rng(seed: Seed) -> np.random.Generator:
    return seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)


def _shape(n: int, n_paths: int | None) -> tuple[int, ...]:
    return (n,) if n_paths is None else (n, n_paths)


def white_noise(
    n: int, sigma: float = 1.0, n_paths: int | None = None, seed: Seed = None
) -> np.ndarray:
    """I.i.d. Gaussian noise with standard deviation ``sigma``."""
    return _rng(seed).normal(0.0, sigma, _shape(n, n_paths))


def random_walk(
    n: int,
    sigma: float = 1.0,
    drift: float = 0.0,
    x0: float = 0.0,
    n_paths: int | None = None,
    seed: Seed = None,
) -> np.ndarray:
    r"""Random walk :math:`x_t = x_{t-1} + \mu + \varepsilon_t`, starting from ``x0``.

    The first value is :math:`x_0 + \mu + \varepsilon_1`, so :math:`\operatorname{Var}(x_t)
    = t\sigma^2` for :math:`t = 1, \dots, n`: the variance grows linearly, as in diffusion.
    """
    steps = drift + white_noise(n, sigma, n_paths, seed)
    return x0 + np.cumsum(steps, axis=0)


def ar1(
    n: int,
    phi: float,
    sigma: float = 1.0,
    c: float = 0.0,
    x0: float | None = None,
    n_paths: int | None = None,
    seed: Seed = None,
) -> np.ndarray:
    r"""AR(1) process :math:`x_t = c + \phi x_{t-1} + \varepsilon_t`.

    ``x0`` is the value *before* the first output. By default it is drawn from the stationary
    distribution :math:`\mathcal N\big(c/(1-\phi),\, \sigma^2/(1-\phi^2)\big)` when
    :math:`|\phi| < 1`, so the series is stationary from the start, and set to 0 otherwise
    (for :math:`\phi = 1` the process is a random walk with drift :math:`c`).
    """
    rng = _rng(seed)
    shocks = rng.normal(0.0, sigma, _shape(n, n_paths))
    path_shape = () if n_paths is None else (n_paths,)
    if x0 is not None:
        start = np.full(path_shape, float(x0))
    elif abs(phi) < 1:
        start = rng.normal(c / (1 - phi), sigma / np.sqrt(1 - phi**2), path_shape)
    else:
        start = np.zeros(path_shape)
    # lfilter solves x_t - phi x_{t-1} = c + eps_t along axis 0; zi carries in phi * x_0.
    out, _ = lfilter([1.0], [1.0, -phi], c + shocks, axis=0, zi=(phi * start)[np.newaxis])
    return out


class CointegratedPair(NamedTuple):
    x: np.ndarray
    y: np.ndarray
    spread: np.ndarray


def cointegrated_pair(
    n: int,
    beta: float,
    phi: float,
    sigma: float = 1.0,
    sigma_x: float = 1.0,
    alpha: float = 0.0,
    x0: float = 100.0,
    seed: Seed = None,
) -> CointegratedPair:
    r"""Two series sharing one random-walk trend, with a stationary spread.

    :math:`x_t` is a random walk (step size ``sigma_x``, starting near ``x0``);
    :math:`u_t` is a zero-mean AR(1) with coefficient ``phi`` and shock size ``sigma``;
    :math:`y_t = \alpha + \beta x_t + u_t`. With :math:`|\phi| < 1` the combination
    :math:`y - \beta x` is stationary: :math:`x` and :math:`y` are cointegrated with hedge
    ratio :math:`\beta`. With :math:`\phi = 1` they are not.
    """
    rng = _rng(seed)
    x = random_walk(n, sigma_x, x0=x0, seed=rng)
    spread = ar1(n, phi, sigma, seed=rng)
    return CointegratedPair(x=x, y=alpha + beta * x + spread, spread=spread)
