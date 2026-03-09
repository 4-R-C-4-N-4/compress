"""Tests for the PPM model (python/model.py)."""

import random
import pytest

from python.arithmetic import Encoder, Decoder
from python.model import PPMOrder, OrderMinus1, PPMModel, NUM_SYMBOLS


class TestPPMOrder:
    def test_counts_update(self):
        """Feed known bytes through PPMOrder, verify counts update correctly."""
        order = PPMOrder(order=1)
        # Feed "aba"
        order.update(b"x", ord("a"))  # context x, symbol a
        order.update(b"a", ord("b"))  # context a, symbol b
        order.update(b"b", ord("a"))  # context b, symbol a

        counts_x, total_x, distinct_x = order.get_distribution(b"x")
        assert counts_x[ord("a")] == 1
        assert total_x == 1
        assert distinct_x == 1

        counts_a, total_a, distinct_a = order.get_distribution(b"a")
        assert counts_a[ord("b")] == 1
        assert total_a == 1

    def test_order_zero(self):
        """Order 0 uses empty context."""
        order = PPMOrder(order=0)
        order.update(b"anything", 42)
        order.update(b"else", 42)
        order.update(b"", 99)

        counts, total, distinct = order.get_distribution(b"")
        assert counts[42] == 2
        assert counts[99] == 1
        assert total == 3
        assert distinct == 2

    def test_exclusions(self):
        """Verify excluded symbols are masked in get_distribution."""
        order = PPMOrder(order=0)
        for s in [10, 20, 30]:
            order.update(b"", s)

        exclusions = [False] * NUM_SYMBOLS
        exclusions[20] = True

        counts, total, distinct = order.get_distribution(b"", exclusions)
        assert counts[20] == 0
        assert counts[10] == 1
        assert counts[30] == 1
        assert total == 2
        assert distinct == 2

    def test_seed_as_fallback_prior(self):
        """Seed used as fallback when no adaptive data; replaced once adaptive data exists."""
        seed = {b"": [0] * NUM_SYMBOLS}
        seed[b""][65] = 10  # 'A' has seed count 10
        seed[b""][66] = 5   # 'B' has seed count 5

        order = PPMOrder(order=0, seed_counts=seed)

        # Before any adaptive data: seed counts are used
        counts, total, distinct = order.get_distribution(b"")
        assert counts[65] == 10
        assert counts[66] == 5
        assert total == 15
        assert distinct == 2

        # After adaptive data: only adaptive counts used
        order.update(b"", 65)
        counts, total, distinct = order.get_distribution(b"")
        assert counts[65] == 1  # adaptive only
        assert counts[66] == 0  # seed dropped
        assert total == 1
        assert distinct == 1


class TestOrderMinus1:
    def test_uniform(self):
        """Order -1 returns uniform 1/256."""
        om1 = OrderMinus1()
        counts, total, distinct = om1.get_distribution(b"")
        assert total == 256
        assert distinct == 256
        assert all(c == 1 for c in counts)

    def test_exclusions(self):
        """Excluded symbols get 0 in order -1."""
        om1 = OrderMinus1()
        excl = [False] * NUM_SYMBOLS
        excl[0] = True
        excl[255] = True
        counts, total, _ = om1.get_distribution(b"", excl)
        assert counts[0] == 0
        assert counts[255] == 0
        assert total == 254


class TestPPMModelRoundtrip:
    def test_short_deterministic(self):
        """Encode then decode a short deterministic string."""
        data = b"abracadabra"
        model_enc = PPMModel(max_order=3)
        enc = Encoder()

        context = b""
        for byte in data:
            model_enc.encode_symbol(enc, context, byte)
            context = (context + bytes([byte]))[-3:]

        bitstream = enc.finish()

        model_dec = PPMModel(max_order=3)
        dec = Decoder(bitstream)

        result = bytearray()
        context = b""
        for _ in range(len(data)):
            sym = model_dec.decode_symbol(dec, context)
            result.append(sym)
            context = (context + bytes([sym]))[-3:]

        assert bytes(result) == data

    def test_all_same_bytes(self):
        """Sequence of identical bytes."""
        data = b"\x00" * 50
        model_enc = PPMModel(max_order=2)
        enc = Encoder()

        context = b""
        for byte in data:
            model_enc.encode_symbol(enc, context, byte)
            context = (context + bytes([byte]))[-2:]

        bitstream = enc.finish()

        model_dec = PPMModel(max_order=2)
        dec = Decoder(bitstream)

        result = bytearray()
        context = b""
        for _ in range(len(data)):
            sym = model_dec.decode_symbol(dec, context)
            result.append(sym)
            context = (context + bytes([sym]))[-2:]

        assert bytes(result) == data

    @pytest.mark.randomized
    @pytest.mark.parametrize("order", [1, 2, 3, 4, 6])
    def test_random_roundtrip(self, order):
        """Random byte sequences at various orders."""
        seed = random.randrange(2**32)
        print(f"Random seed: {seed}, order: {order}")
        r = random.Random(seed)

        length = r.randint(10, 200)
        data = bytes(r.randint(0, 255) for _ in range(length))

        model_enc = PPMModel(max_order=order)
        enc = Encoder()
        context = b""
        for byte in data:
            model_enc.encode_symbol(enc, context, byte)
            context = (context + bytes([byte]))[-order:]
        bitstream = enc.finish()

        model_dec = PPMModel(max_order=order)
        dec = Decoder(bitstream)
        result = bytearray()
        context = b""
        for _ in range(len(data)):
            sym = model_dec.decode_symbol(dec, context)
            result.append(sym)
            context = (context + bytes([sym]))[-order:]

        assert bytes(result) == data

    def test_with_seed_counts(self, english_seed_counts):
        """Roundtrip with actual seed counts loaded."""
        data = b"Hello, world!"
        model_enc = PPMModel(max_order=4, seed_counts=english_seed_counts)
        enc = Encoder()
        context = b""
        for byte in data:
            model_enc.encode_symbol(enc, context, byte)
            context = (context + bytes([byte]))[-4:]
        bitstream = enc.finish()

        model_dec = PPMModel(max_order=4, seed_counts=english_seed_counts)
        dec = Decoder(bitstream)
        result = bytearray()
        context = b""
        for _ in range(len(data)):
            sym = model_dec.decode_symbol(dec, context)
            result.append(sym)
            context = (context + bytes([sym]))[-4:]

        assert bytes(result) == data
