import pytest

import statarb
from statarb import config


def test_package_imports():
    assert statarb.__version__


def test_trading_days():
    assert config.TRADING_DAYS == 252


def test_bps_to_decimal():
    assert config.bps_to_decimal(10) == pytest.approx(0.001)
    assert config.bps_to_decimal(100) == pytest.approx(0.01)


def test_data_dir_is_inside_project_by_default(monkeypatch):
    monkeypatch.delenv("STARB_DATA_DIR", raising=False)
    assert config.RAW_DIR.parent == config.DATA_DIR
