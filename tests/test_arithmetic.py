"""Tests for the arithmetic coder (python/arithmetic.py)."""

import random
import pytest

from python.arithmetic import Encoder, Decoder


class TestFixedRoundtrip:
    """Encode/decode known symbol sequences and verify exact roundtrip."""

    def test_simple_binary_alphabet(self):
        """Two-symbol alphabet with known frequencies."""
        # Alphabet: 0 (freq 3), 1 (freq 1). Total = 4.
        # Cumulative: sym 0 -> [0, 3), sym 1 -> [3, 4)
        symbols = [0, 1, 0, 0, 1, 0]
        freq = {0: (0, 3, 4), 1: (3, 4, 4)}

        enc = Encoder()
        for s in symbols:
            enc.encode_symbol(*freq[s])
        data = enc.finish()

        dec = Decoder(data)
        decoded = []
        for _ in range(len(symbols)):
            val = dec.get_value(4)
            if val < 3:
                sym = 0
            else:
                sym = 1
            dec.decode_symbol(*freq[sym])
            decoded.append(sym)

        assert decoded == symbols

    def test_four_symbol_alphabet(self):
        """Four symbols with varying frequencies."""
        # Frequencies: A=5, B=3, C=1, D=1. Total = 10
        # Cumulative: A->[0,5), B->[5,8), C->[8,9), D->[9,10)
        freqs = {0: (0, 5, 10), 1: (5, 8, 10), 2: (8, 9, 10), 3: (9, 10, 10)}
        symbols = [0, 0, 1, 2, 3, 0, 1, 0]

        enc = Encoder()
        for s in symbols:
            enc.encode_symbol(*freqs[s])
        data = enc.finish()

        dec = Decoder(data)
        decoded = []
        for _ in range(len(symbols)):
            val = dec.get_value(10)
            if val < 5:
                sym = 0
            elif val < 8:
                sym = 1
            elif val < 9:
                sym = 2
            else:
                sym = 3
            dec.decode_symbol(*freqs[sym])
            decoded.append(sym)

        assert decoded == symbols


class TestRandomRoundtrip:
    """Random symbol sequences with random frequency distributions."""

    @pytest.mark.randomized
    def test_random_uniform(self):
        seed = random.randrange(2**32)
        print(f"Random seed: {seed}")
        r = random.Random(seed)

        n_symbols = r.randint(2, 20)
        freqs_list = [r.randint(1, 100) for _ in range(n_symbols)]
        total = sum(freqs_list)

        # Build cumulative ranges
        cum = []
        lo = 0
        for f in freqs_list:
            cum.append((lo, lo + f, total))
            lo += f

        # Generate random sequence
        seq = [r.randint(0, n_symbols - 1) for _ in range(r.randint(10, 200))]

        enc = Encoder()
        for s in seq:
            enc.encode_symbol(*cum[s])
        data = enc.finish()

        dec = Decoder(data)
        decoded = []
        for _ in range(len(seq)):
            val = dec.get_value(total)
            # Find which symbol this value falls into
            for sym_idx, (cl, ch, t) in enumerate(cum):
                if cl <= val < ch:
                    dec.decode_symbol(cl, ch, t)
                    decoded.append(sym_idx)
                    break

        assert decoded == seq

    @pytest.mark.randomized
    def test_random_multiple_runs(self):
        """Run multiple random roundtrips to increase coverage."""
        seed = random.randrange(2**32)
        print(f"Random seed: {seed}")
        r = random.Random(seed)

        for _ in range(10):
            n_symbols = r.randint(2, 8)
            freqs_list = [r.randint(1, 50) for _ in range(n_symbols)]
            total = sum(freqs_list)

            cum = []
            lo = 0
            for f in freqs_list:
                cum.append((lo, lo + f, total))
                lo += f

            seq = [r.randint(0, n_symbols - 1) for _ in range(r.randint(5, 50))]

            enc = Encoder()
            for s in seq:
                enc.encode_symbol(*cum[s])
            data = enc.finish()

            dec = Decoder(data)
            decoded = []
            for _ in range(len(seq)):
                val = dec.get_value(total)
                for sym_idx, (cl, ch, t) in enumerate(cum):
                    if cl <= val < ch:
                        dec.decode_symbol(cl, ch, t)
                        decoded.append(sym_idx)
                        break

            assert decoded == seq


class TestEdgeCases:
    def test_single_symbol_alphabet(self):
        """Alphabet with only one symbol (total=1, range [0,1))."""
        enc = Encoder()
        for _ in range(100):
            enc.encode_symbol(0, 1, 1)
        data = enc.finish()

        dec = Decoder(data)
        for _ in range(100):
            val = dec.get_value(1)
            assert val == 0
            dec.decode_symbol(0, 1, 1)

    def test_very_skewed_distribution(self):
        """One dominant symbol with a rare symbol."""
        # sym 0: freq 9999, sym 1: freq 1. Total = 10000
        freqs = {0: (0, 9999, 10000), 1: (9999, 10000, 10000)}
        symbols = [0] * 50 + [1] + [0] * 49

        enc = Encoder()
        for s in symbols:
            enc.encode_symbol(*freqs[s])
        data = enc.finish()

        dec = Decoder(data)
        decoded = []
        for _ in range(len(symbols)):
            val = dec.get_value(10000)
            sym = 0 if val < 9999 else 1
            dec.decode_symbol(*freqs[sym])
            decoded.append(sym)

        assert decoded == symbols
