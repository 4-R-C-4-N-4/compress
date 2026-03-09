"""PPM model with PPMD-style escape and logistic context mixing."""

import math
from collections import defaultdict

NUM_SYMBOLS = 256
ESCAPE = NUM_SYMBOLS  # virtual escape symbol index


class PPMOrder:
    """Single-order adaptive context model with escape."""

    def __init__(self, order, seed_counts=None):
        self.order = order
        self.seed = seed_counts or {}
        self.counts = defaultdict(lambda: [0] * NUM_SYMBOLS)
        self.totals = defaultdict(int)

    def get_distribution(self, context, exclusions=None):
        """Return (symbol_counts, total, num_distinct) for this context.

        Applies exclusion filtering: symbols already excluded by higher
        orders are masked out (PPMD exclusion).

        Seed acts as a fallback prior: if the adaptive model has data for
        this context, use adaptive counts only. If not, use seed counts.
        This keeps the escape probability tight once real data arrives.
        """
        ctx = context[-self.order:] if self.order > 0 else b""
        raw = self.counts[ctx]
        seed_raw = self.seed.get(ctx)
        has_adaptive = self.totals[ctx] > 0

        counts = [0] * NUM_SYMBOLS
        distinct = 0
        total = 0

        for s in range(NUM_SYMBOLS):
            if exclusions and exclusions[s]:
                continue
            if has_adaptive:
                c = raw[s]
            else:
                c = (seed_raw[s] if seed_raw else 0)
            if c > 0:
                counts[s] = c
                total += c
                distinct += 1

        return counts, total, distinct

    def update(self, context, symbol):
        ctx = context[-self.order:] if self.order > 0 else b""
        self.counts[ctx][symbol] += 1
        self.totals[ctx] += 1


class OrderMinus1:
    """Order -1: uniform distribution over all bytes."""

    def get_distribution(self, context, exclusions=None):
        counts = [0] * NUM_SYMBOLS
        total = 0
        for s in range(NUM_SYMBOLS):
            if exclusions and exclusions[s]:
                continue
            counts[s] = 1
            total += 1
        return counts, total, total

    def update(self, context, symbol):
        pass

    order = -1


SEED_WEIGHT = 256  # max total counts per seed context


def _scale_seed_order(order_counts, target_total):
    """Scale seed counts so each context's total is at most target_total.

    Preserves ratios. Non-zero counts stay >= 1.
    """
    scaled = {}
    for ctx, syms in order_counts.items():
        total = sum(syms)
        if total <= target_total:
            scaled[ctx] = syms
            continue
        factor = target_total / total
        new_syms = [max(1, int(c * factor)) if c > 0 else 0 for c in syms]
        scaled[ctx] = new_syms
    return scaled


class PPMModel:
    """Full PPM model with PPMD-style escape mechanism.

    Seed counts are scaled down so adaptive counts can take over quickly.
    """

    def __init__(self, max_order=4, seed_counts=None, seed_weight=SEED_WEIGHT):
        self.max_order = max_order
        self.orders = []
        for o in range(max_order + 1):
            sc = seed_counts.get(o) if seed_counts else None
            if sc is not None:
                sc = _scale_seed_order(sc, seed_weight)
            self.orders.append(PPMOrder(o, sc))
        self.order_m1 = OrderMinus1()

    def encode_symbol(self, encoder, context, symbol):
        """Encode a single symbol using PPM escape, updating the model."""
        exclusions = [False] * NUM_SYMBOLS

        for o in range(self.max_order, -1, -1):
            order_model = self.orders[o]
            counts, total, distinct = order_model.get_distribution(context, exclusions)

            if counts[symbol] > 0:
                # Symbol found at this order — encode it
                escape_count = distinct  # PPMD Method D
                grand_total = total + escape_count

                # Build cumulative for the target symbol
                cum_low = 0
                for s in range(symbol):
                    cum_low += counts[s]
                cum_high = cum_low + counts[symbol]

                encoder.encode_symbol(cum_low, cum_high, grand_total)
                self._update_all(context, symbol)
                return

            if total > 0 or distinct > 0:
                # Context exists but symbol not seen — encode escape
                escape_count = max(distinct, 1)
                grand_total = total + escape_count

                # Escape symbol is at the end
                encoder.encode_symbol(total, grand_total, grand_total)

                # Exclusion: mask out symbols seen at this order
                for s in range(NUM_SYMBOLS):
                    if counts[s] > 0:
                        exclusions[s] = True
            # else: empty context, fall through silently

        # Order -1: uniform fallback
        counts, total, _ = self.order_m1.get_distribution(context, exclusions)
        cum_low = 0
        for s in range(symbol):
            cum_low += counts[s]
        cum_high = cum_low + counts[symbol]
        encoder.encode_symbol(cum_low, cum_high, total)
        self._update_all(context, symbol)

    def decode_symbol(self, decoder, context):
        """Decode a single symbol using PPM escape, updating the model."""
        exclusions = [False] * NUM_SYMBOLS

        for o in range(self.max_order, -1, -1):
            order_model = self.orders[o]
            counts, total, distinct = order_model.get_distribution(context, exclusions)

            if total == 0 and distinct == 0:
                continue

            escape_count = max(distinct, 1)
            grand_total = total + escape_count

            value = decoder.get_value(grand_total)

            if value < total:
                # Decode actual symbol
                cum = 0
                for s in range(NUM_SYMBOLS):
                    if counts[s] == 0:
                        continue
                    if cum + counts[s] > value:
                        decoder.decode_symbol(cum, cum + counts[s], grand_total)
                        self._update_all(context, s)
                        return s
                    cum += counts[s]

            # It's an escape
            decoder.decode_symbol(total, grand_total, grand_total)
            for s in range(NUM_SYMBOLS):
                if counts[s] > 0:
                    exclusions[s] = True

        # Order -1 fallback
        counts, total, _ = self.order_m1.get_distribution(context, exclusions)
        value = decoder.get_value(total)
        cum = 0
        for s in range(NUM_SYMBOLS):
            if counts[s] == 0:
                continue
            if cum + counts[s] > value:
                decoder.decode_symbol(cum, cum + counts[s], total)
                self._update_all(context, s)
                return s
            cum += counts[s]

        raise RuntimeError("Decode failed: no symbol found in order -1")

    def _update_all(self, context, symbol):
        """Update all order models with the observed symbol."""
        for order_model in self.orders:
            order_model.update(context, symbol)
