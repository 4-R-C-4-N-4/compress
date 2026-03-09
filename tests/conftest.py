"""Shared fixtures for the compress test suite."""

import os
import pytest

from python.seed_format import read_seed

SEEDS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "seeds")


# ---------------------------------------------------------------------------
# Sample data blobs
# ---------------------------------------------------------------------------

@pytest.fixture
def short_english():
    return b"The quick brown fox jumps over the lazy dog."

@pytest.fixture
def random_bytes():
    import random
    r = random.Random(42)
    return bytes(r.randint(0, 255) for _ in range(500))

@pytest.fixture
def all_zeros():
    return b"\x00" * 256

@pytest.fixture
def single_byte():
    return b"\x42"

@pytest.fixture
def empty_data():
    return b""


# ---------------------------------------------------------------------------
# Seed fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def null_seed_path():
    return os.path.join(SEEDS_DIR, "null.seedmodel")

@pytest.fixture
def english_seed_path():
    return os.path.join(SEEDS_DIR, "english.seedmodel")

@pytest.fixture
def code_seed_path():
    return os.path.join(SEEDS_DIR, "code.seedmodel")

@pytest.fixture
def log_seed_path():
    return os.path.join(SEEDS_DIR, "log.seedmodel")

@pytest.fixture
def null_seed_counts(null_seed_path):
    _, _, _, counts = read_seed(null_seed_path)
    return counts

@pytest.fixture
def english_seed_counts(english_seed_path):
    _, _, _, counts = read_seed(english_seed_path)
    return counts
