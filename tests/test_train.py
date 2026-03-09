"""Tests for the training pipeline (python/train.py)."""

import pytest

from python.train import train_model, extract_counts, prune_counts, quantize_counts


class TestTrainModel:
    def test_basic_training(self):
        """Feed a small known corpus, verify counts are non-zero."""
        corpus = b"abcabc"
        model = train_model(corpus, max_order=2)
        counts = extract_counts(model)

        # Order 0 should have counts for a, b, c
        assert counts[0][b""][ord("a")] == 2
        assert counts[0][b""][ord("b")] == 2
        assert counts[0][b""][ord("c")] == 2

        # Order 1: after 'a', we should see 'b' twice
        assert counts[1][b"a"][ord("b")] == 2

    def test_single_byte_corpus(self):
        """Training on a single byte."""
        model = train_model(b"x", max_order=1)
        counts = extract_counts(model)
        assert counts[0][b""][ord("x")] == 1


class TestPruneCounts:
    def test_prune_removes_low_counts(self):
        """Contexts below threshold are removed."""
        counts = {
            0: {
                b"": [0] * 256,  # will have total 10
            },
            1: {
                b"a": [0] * 256,  # will have total 1 (below threshold)
                b"b": [0] * 256,  # will have total 5 (above threshold)
            },
        }
        counts[0][b""][0] = 10
        counts[1][b"a"][0] = 1
        counts[1][b"b"][0] = 5

        pruned = prune_counts(counts, min_total=3)

        assert b"" in pruned[0]     # total=10, kept
        assert b"a" not in pruned[1]  # total=1, removed
        assert b"b" in pruned[1]     # total=5, kept

    def test_prune_zero_threshold(self):
        """Threshold 0 keeps everything."""
        counts = {0: {b"": [1] + [0] * 255}}
        pruned = prune_counts(counts, min_total=0)
        assert b"" in pruned[0]


class TestQuantizeCounts:
    def test_values_fit_u16(self):
        """After quantization, all values fit in u16."""
        counts = {0: {b"": [0] * 256}}
        counts[0][b""][0] = 100000
        counts[0][b""][1] = 50000

        quantized = quantize_counts(counts)

        for ctx_syms in quantized[0].values():
            for c in ctx_syms:
                assert c <= 65535

    def test_ratios_preserved(self):
        """Quantization preserves approximate ratios."""
        counts = {0: {b"": [0] * 256}}
        counts[0][b""][0] = 200000
        counts[0][b""][1] = 100000

        quantized = quantize_counts(counts)
        syms = quantized[0][b""]

        # Ratio should be approximately 2:1
        assert syms[0] == 65535  # max_val
        assert 32000 <= syms[1] <= 33000  # ~half of max

    def test_no_scaling_needed(self):
        """Counts already in u16 range should pass through."""
        counts = {0: {b"": [0] * 256}}
        counts[0][b""][0] = 100
        counts[0][b""][1] = 200

        quantized = quantize_counts(counts)
        assert quantized[0][b""][0] == 100
        assert quantized[0][b""][1] == 200
