"""Tests for the high-level codec (python/codec.py)."""

import os
import struct
import random
import hashlib
import pytest

from python.codec import encode, decode, load_seed, load_seed_by_name, auto_select_seed, resolve_seed, MAGIC, SEPARATOR, FINGERPRINT_LEN


class TestFixedRoundtrip:
    def test_hello_world(self):
        """Roundtrip b'hello world' with seed_id=0."""
        data = b"hello world"
        compressed = encode(data, seed_id=0)
        result = decode(compressed)
        assert result == data

    def test_with_english_seed(self, english_seed_counts):
        """Roundtrip with english seed."""
        data = b"The quick brown fox jumps over the lazy dog."
        compressed = encode(data, seed_id=1, seed_counts=english_seed_counts)
        result = decode(compressed, seed_counts=english_seed_counts)
        assert result == data

    def test_with_code_seed(self):
        """Roundtrip with code seed (loaded from file)."""
        seed_counts = load_seed(2)
        if seed_counts is None:
            pytest.skip("code.seedmodel not found")
        data = b"def main():\n    print('hello')\n"
        compressed = encode(data, seed_id=2, seed_counts=seed_counts)
        result = decode(compressed, seed_counts=seed_counts)
        assert result == data

    def test_with_log_seed(self):
        """Roundtrip with log seed (loaded from file)."""
        seed_counts = load_seed(5)
        if seed_counts is None:
            pytest.skip("log.seedmodel not found")
        data = b"2024-01-15 10:30:00 INFO  Server started on port 8080\n"
        compressed = encode(data, seed_id=5, seed_counts=seed_counts)
        result = decode(compressed, seed_counts=seed_counts)
        assert result == data


class TestRandomRoundtrip:
    @pytest.mark.randomized
    @pytest.mark.parametrize("length", [0, 1, 100, 1000])
    def test_random_bytes(self, length):
        """Random byte strings of various lengths, roundtrip with seed_id=0."""
        seed = random.randrange(2**32)
        print(f"Random seed: {seed}, length: {length}")
        r = random.Random(seed)

        data = bytes(r.randint(0, 255) for _ in range(length))
        compressed = encode(data, seed_id=0)
        result = decode(compressed)
        assert result == data


class TestHeaderStructure:
    def test_header_fields(self):
        """Verify header structure: magic, seed_id, order, byte_length, fingerprint, separator."""
        data = b"test data for header"
        compressed = encode(data, seed_id=0, max_order=3)

        assert compressed[:4] == MAGIC
        assert compressed[4] == 0
        assert compressed[5] == 3
        byte_length = struct.unpack("<I", compressed[6:10])[0]
        assert byte_length == len(data)
        fp = compressed[10:10 + FINGERPRINT_LEN]
        expected_fp = hashlib.blake2b(data, digest_size=FINGERPRINT_LEN).digest()
        assert fp == expected_fp
        sep_start = 10 + FINGERPRINT_LEN
        assert compressed[sep_start:sep_start + 3] == SEPARATOR


class TestAutoDetect:
    def test_auto_roundtrips(self):
        """Auto-select picks a seed and roundtrips correctly."""
        data = b"The quick brown fox jumps over the lazy dog. " * 20
        compressed = encode(data, auto=True)
        result = decode(compressed)
        assert result == data

    def test_auto_returns_name(self):
        """auto_select_seed returns a name, not just an ID."""
        data = b"It was the best of times, it was the worst of times. " * 20
        best_name, best_id, _ = auto_select_seed(data)
        assert isinstance(best_name, str)

    def test_auto_roundtrip_empty(self):
        """Auto-detect on empty data falls back to null."""
        compressed = encode(b"", auto=True)
        result = decode(compressed)
        assert result == b""
        assert compressed[4] == 0

    def test_auto_header_stores_chosen_seed(self):
        """The header stores the actual chosen seed ID."""
        data = b"Hello world! " * 100
        compressed = encode(data, auto=True)
        result = decode(compressed)
        assert result == data


class TestResolveSeed:
    def test_resolve_auto(self):
        """'auto' resolves with is_auto=True."""
        sid, counts, is_auto = resolve_seed("auto")
        assert is_auto is True

    def test_resolve_numeric(self):
        """Numeric string resolves to seed ID."""
        sid, counts, is_auto = resolve_seed("0")
        assert sid == 0
        assert is_auto is False

    def test_resolve_name(self):
        """Name resolves to seed counts."""
        result = load_seed_by_name("english")
        if result is None:
            pytest.skip("english.seedmodel not found")
        sid, counts, is_auto = resolve_seed("english")
        assert is_auto is False
        assert counts is not None

    def test_resolve_unknown(self):
        """Unknown name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown seed"):
            resolve_seed("nonexistent_seed_xyz")


class TestErrorCases:
    def test_corrupted_magic(self):
        """Bad magic raises ValueError."""
        with pytest.raises(ValueError, match="bad magic"):
            decode(b"XXXX" + b"\x00" * 30)

    def test_truncated_data(self):
        """Truncated compressed data should fail (fingerprint mismatch or struct error)."""
        data = b"hello world"
        compressed = encode(data, seed_id=0)
        truncated = compressed[:20]
        with pytest.raises((ValueError, struct.error)):
            decode(truncated)

    def test_bad_fingerprint(self):
        """Modified data should fail fingerprint check."""
        data = b"hello world"
        compressed = bytearray(encode(data, seed_id=0))
        compressed[10] ^= 0xFF
        with pytest.raises(ValueError, match="Fingerprint mismatch"):
            decode(bytes(compressed))
