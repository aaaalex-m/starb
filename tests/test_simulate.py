import numpy as np
import pytest

from statarb import simulate


def test_shapes_and_reproducibility():
    assert simulate.random_walk(50, seed=1).shape == (50,)
    assert simulate.ar1(50, 0.5, n_paths=7, seed=1).shape == (50, 7)
    assert simulate.white_noise(50, n_paths=3, seed=1).shape == (50, 3)
    np.testing.assert_array_equal(simulate.ar1(50, 0.5, seed=4), simulate.ar1(50, 0.5, seed=4))
    assert not np.array_equal(simulate.ar1(50, 0.5, seed=4), simulate.ar1(50, 0.5, seed=5))


def test_accepts_a_generator():
    rng = np.random.default_rng(0)
    a, b = simulate.white_noise(10, seed=rng), simulate.white_noise(10, seed=rng)
    assert not np.array_equal(a, b)  # the generator advances


def test_random_walk_steps_are_drift_plus_noise():
    x = simulate.random_walk(100, sigma=2.0, drift=0.1, x0=5.0, seed=3)
    noise = simulate.white_noise(100, sigma=2.0, seed=3)
    np.testing.assert_allclose(np.diff(x, prepend=5.0), 0.1 + noise)


def test_random_walk_variance_grows_linearly():
    paths = simulate.random_walk(200, sigma=1.0, n_paths=20_000, seed=0)
    var = paths.var(axis=1)
    for t in (10, 50, 200):
        assert var[t - 1] == pytest.approx(t, rel=0.04)


def test_ar1_matches_explicit_recursion():
    phi, c, x0, sigma = 0.7, 0.3, 2.0, 0.5
    x = simulate.ar1(30, phi, sigma=sigma, c=c, x0=x0, seed=9)
    shocks = np.random.default_rng(9).normal(0, sigma, 30)
    expected, prev = [], x0
    for e in shocks:
        prev = c + phi * prev + e
        expected.append(prev)
    np.testing.assert_allclose(x, expected)


def test_ar1_stationary_moments_and_autocorrelation():
    phi, c, sigma = 0.9, 0.5, 1.0
    ens = simulate.ar1(50, phi, sigma=sigma, c=c, n_paths=40_000, seed=2)
    # starts in the stationary distribution, so the moments hold at every date
    for t in (0, 49):
        assert ens[t].mean() == pytest.approx(c / (1 - phi), abs=0.05)
        assert ens[t].var() == pytest.approx(sigma**2 / (1 - phi**2), rel=0.03)
    long = simulate.ar1(200_000, phi, seed=2)
    lag1 = np.corrcoef(long[1:], long[:-1])[0, 1]
    assert lag1 == pytest.approx(phi, abs=0.01)


def test_ar1_with_unit_root_is_a_random_walk():
    np.testing.assert_allclose(
        simulate.ar1(100, 1.0, c=0.2, seed=6), simulate.random_walk(100, drift=0.2, seed=6)
    )


def test_cointegrated_pair_structure():
    pair = simulate.cointegrated_pair(1000, beta=1.5, phi=0.8, alpha=2.0, seed=11)
    np.testing.assert_allclose(pair.y - 1.5 * pair.x, 2.0 + pair.spread)
    # the spread is bounded while x wanders
    assert pair.spread.std() < 3
    assert np.abs(pair.x - 100).max() > 3 * pair.spread.std()


def test_cointegrated_pair_with_unit_root_spread_wanders():
    stationary = simulate.cointegrated_pair(5000, 1.0, 0.5, seed=1).spread
    unit_root = simulate.cointegrated_pair(5000, 1.0, 1.0, seed=1).spread
    assert unit_root.std() > 5 * stationary.std()
