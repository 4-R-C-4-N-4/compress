"""Tests for the .seedmodel binary format (python/seed_format.py)."""

import os
import random
import struct
import tempfile
import pytest

from python.seed_format import write_seed, read_seed, MAGIC, FORMAT_VERSION, NUM_SYMBOLS


class TestWriteReadRoundtrip:
    def test_simple_roundtrip(self, tmp_path):
        """Write a known counts dict, read it back, assert equality."""
        counts = {
            0: {b"": [0] * NUM_SYMBOLS},
            1: {b"a": [0] * NUM_SYMBOLS, b"b": [0] * NUM_SYMBOLS},
        }
        counts[0][b""][65] = 100   # 'A'
        counts[0][b""][66] = 200   # 'B'
        counts[1][b"a"][99] = 50   # 'c' after 'a'
        counts[1][b"b"][100] = 75  # 'd' after 'b'

        path = str(tmp_path / "test.seedmodel")
        write_seed(path, seed_id=7, name="test", max_order=1, counts=counts)

        sid, name, max_order, read_counts = read_seed(path)

        assert sid == 7
        assert name == "test"
        assert max_order == 1

        # Verify counts match
        assert read_counts[0][b""][65] == 100
        assert read_counts[0][b""][66] == 200
        assert read_counts[1][b"a"][99] == 50
        assert read_counts[1][b"b"][100] == 75

    def test_empty_counts(self, tmp_path):
        """Seed with no contexts at any order."""
        counts = {0: {}, 1: {}}
        path = str(tmp_path / "empty.seedmodel")
        write_seed(path, seed_id=0, name="empty", max_order=1, counts=counts)

        sid, name, max_order, read_counts = read_seed(path)
        assert sid == 0
        assert name == "empty"
        assert len(read_counts[0]) == 0
        assert len(read_counts[1]) == 0


class TestBinaryHeader:
    def test_header_fields(self, tmp_path):
        """Verify binary header (magic, version, seed_id, name) by reading raw bytes."""
        counts = {0: {}}
        path = str(tmp_path / "hdr.seedmodel")
        write_seed(path, seed_id=42, name="mymodel", max_order=0, counts=counts)

        with open(path, "rb") as f:
            data = f.read()

        assert data[:4] == MAGIC
        assert data[4] == FORMAT_VERSION
        assert data[5] == 42  # seed_id
        assert data[6] == 0   # max_order
        assert data[7] == 7   # name_len ("mymodel" = 7 chars)
        assert data[8:15] == b"mymodel"


class TestNullSeed:
    def test_null_loads(self, null_seed_path):
        """Verify null.seedmodel loads correctly."""
        sid, name, max_order, counts = read_seed(null_seed_path)
        assert sid == 0
        assert name == "null"
        assert max_order >= 0


class TestRandomRoundtrip:
    @pytest.mark.randomized
    def test_random_counts(self, tmp_path):
        """Random counts dicts with random sparse contexts."""
        seed = random.randrange(2**32)
        print(f"Random seed: {seed}")
        r = random.Random(seed)

        max_order = r.randint(0, 3)
        counts = {}
        for order in range(max_order + 1):
            order_counts = {}
            n_contexts = r.randint(0, 5)
            for _ in range(n_contexts):
                ctx = bytes(r.randint(0, 255) for _ in range(order))
                syms = [0] * NUM_SYMBOLS
                n_nonzero = r.randint(1, 10)
                for _ in range(n_nonzero):
                    s = r.randint(0, 255)
                    syms[s] = r.randint(1, 65535)
                order_counts[ctx] = syms
            counts[order] = order_counts

        path = str(tmp_path / "random.seedmodel")
        write_seed(path, seed_id=r.randint(0, 255), name="rnd", max_order=max_order, counts=counts)

        sid, name, mo, read_counts = read_seed(path)
        assert mo == max_order

        for order in range(max_order + 1):
            for ctx, syms in counts[order].items():
                assert read_counts[order][ctx] == syms


class TestErrorCases:
    def test_bad_magic(self, tmp_path):
        """File with wrong magic should raise ValueError."""
        path = str(tmp_path / "bad.seedmodel")
        with open(path, "wb") as f:
            f.write(b"XXXX" + b"\x00" * 20)

        with pytest.raises(ValueError, match="bad magic"):
            read_seed(path)

    def test_wrong_version(self, tmp_path):
        """File with unsupported version should raise ValueError."""
        path = str(tmp_path / "badver.seedmodel")
        with open(path, "wb") as f:
            f.write(MAGIC + bytes([99]))  # version 99
            f.write(b"\x00" * 20)

        with pytest.raises(ValueError, match="version"):
            read_seed(path)
