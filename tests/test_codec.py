"""Tests for the high-level codec (python/codec.py)."""

import os
import struct
import random
import hashlib
import pytest

from python.codec import encode, decode, load_seed, auto_select_seed, MAGIC, SEPARATOR, FINGERPRINT_LEN, AUTO_SEED_ID


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

        # Magic
        assert compressed[:4] == MAGIC

        # seed_id
        assert compressed[4] == 0

        # order
        assert compressed[5] == 3

        # byte_length
        byte_length = struct.unpack("<I", compressed[6:10])[0]
        assert byte_length == len(data)

        # fingerprint
        fp = compressed[10:10 + FINGERPRINT_LEN]
        expected_fp = hashlib.blake2b(data, digest_size=FINGERPRINT_LEN).digest()
        assert fp == expected_fp

        # separator
        sep_start = 10 + FINGERPRINT_LEN
        assert compressed[sep_start:sep_start + 3] == SEPARATOR


class TestAutoDetect:
    def test_auto_picks_seed_and_roundtrips(self):
        """seed_id=255 auto-selects a seed and roundtrips correctly."""
        data = b"The quick brown fox jumps over the lazy dog. " * 20
        compressed = encode(data, seed_id=AUTO_SEED_ID)
        result = decode(compressed)
        assert result == data

    def test_auto_picks_english_for_prose(self):
        """Auto-detect should prefer english seed for English text."""
        data = b"It was the best of times, it was the worst of times. " * 20
        best_id, _ = auto_select_seed(data)
        # Should pick english (1) over null (0)
        assert best_id != 0

    def test_auto_roundtrip_empty(self):
        """Auto-detect on empty data falls back to seed 0."""
        compressed = encode(b"", seed_id=AUTO_SEED_ID)
        result = decode(compressed)
        assert result == b""
        # Header should have seed_id=0
        assert compressed[4] == 0

    def test_auto_header_stores_chosen_seed(self):
        """The header should store the actual chosen seed, not 255."""
        data = b"Hello world! " * 100
        compressed = encode(data, seed_id=AUTO_SEED_ID)
        stored_id = compressed[4]
        assert stored_id != AUTO_SEED_ID
        # Should decode fine without any special handling
        result = decode(compressed)
        assert result == data


class TestErrorCases:
    def test_corrupted_magic(self):
        """Bad magic raises ValueError."""
        with pytest.raises(ValueError, match="bad magic"):
            decode(b"XXXX" + b"\x00" * 30)

    def test_truncated_data(self):
        """Truncated compressed data should fail (fingerprint mismatch or struct error)."""
        data = b"hello world"
        compressed = encode(data, seed_id=0)
        # Truncate the bitstream
        truncated = compressed[:20]
        with pytest.raises((ValueError, struct.error)):
            decode(truncated)

    def test_bad_fingerprint(self):
        """Modified data should fail fingerprint check."""
        data = b"hello world"
        compressed = bytearray(encode(data, seed_id=0))
        # Corrupt the fingerprint
        compressed[10] ^= 0xFF
        with pytest.raises(ValueError, match="Fingerprint mismatch"):
            decode(bytes(compressed))
