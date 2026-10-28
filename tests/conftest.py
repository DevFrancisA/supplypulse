import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from supplypulse.config import Config  # noqa: E402
from supplypulse.data import generate  # noqa: E402
from supplypulse.features import build_features  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return Config(n_skus=12, n_paths=60)


@pytest.fixture(scope="session")
def data(cfg):
    return generate(cfg)


@pytest.fixture(scope="session")
def feats(data):
    skus, sales, _ = data
    return build_features(sales, skus)
