import numpy as np
import pandas as pd
import pytest

from statarb import checks
from statarb.signals import threshold_positions


def test_threshold_state_machine_on_a_hand_worked_path():
    z = pd.Series([0, 1, 2.5, 1.5, 0.5, -0.1, -0.5, -2.5, -1, 0.2, np.nan, 3.0])
    expected = [0, 0, -1, -1, -1, 0, 0, 1, 1, 0, 0, -1]
    np.testing.assert_array_equal(threshold_positions(z, entry=2.0, exit=0.0), expected)


def test_exit_level_controls_when_positions_close():
    z = pd.Series([2.5, 1.0, 0.4, 0.0])
    np.testing.assert_array_equal(threshold_positions(z, 2.0, 0.0), [-1, -1, -1, 0])
    np.testing.assert_array_equal(threshold_positions(z, 2.0, 0.5), [-1, -1, 0, 0])


def test_one_transition_per_step():
    """A jump straight through zero closes the position; it reopens the next day at most."""
    z = pd.Series([2.5, -3.0, -3.0])
    np.testing.assert_array_equal(threshold_positions(z), [-1, 0, 1])


def test_dataframe_columns_are_independent():
    z = pd.DataFrame({"a": [0, 3, 0], "b": [0, -3, -3]})
    out = threshold_positions(z)
    np.testing.assert_array_equal(out["a"], [0, -1, 0])
    np.testing.assert_array_equal(out["b"], [0, 1, 1])


@pytest.mark.parametrize(("entry", "exit_"), [(2.0, 2.0), (2.0, -0.5), (1.0, 1.5)])
def test_invalid_thresholds_raise(entry, exit_):
    with pytest.raises(ValueError, match="exit"):
        threshold_positions(pd.Series([0.0]), entry, exit_)


def test_threshold_rule_is_causal():
    rng = np.random.default_rng(0)
    z = pd.Series(rng.normal(0, 1.5, 300), index=pd.bdate_range("2020-01-01", periods=300))
    assert checks.lookahead_violations(threshold_positions, z, n_checks=25) == []
