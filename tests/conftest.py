import numpy as np
import pytest

from swt.config import build_context, load_config

# A coarser version of the default setup so the test suite runs in seconds.
SMALL = {
    "geometry": {"grid_mm": 2.0},
    "path": {"station_spacing_mm": 10.0},
    "filter": {"surrogate_n_eta": 256, "n_particles": 4000},
    "experiments": {"robustness_draws": 2},
}


@pytest.fixture(scope="session")
def small_cfg():
    return load_config(overrides=SMALL)


@pytest.fixture(scope="session")
def small_ctx(small_cfg):
    return build_context(small_cfg)


@pytest.fixture
def rng():
    return np.random.default_rng(12345)
