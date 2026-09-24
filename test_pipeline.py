"""Small checks for the pipeline. Run with `pytest`."""

from pipeline import load_config


def test_config_loads():
    cfg = load_config()
    assert cfg["loop"]["rounds"] == 3
    assert set(cfg["models"]) == {"judge", "rewriter", "answerer"}
