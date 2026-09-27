"""Every config file in the repo must load. This catches typos before Kaggle does."""

from pathlib import Path

import pytest

from gptlab.data.prepare import load_data_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("path", sorted((ROOT / "configs" / "data").glob("*.yaml")), ids=lambda p: p.name)
def test_data_configs_load(path):
    cfg = load_data_config(path)
    assert cfg.name
