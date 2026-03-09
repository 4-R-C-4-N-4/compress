"""Range-based arithmetic coder with 32-bit precision."""

PRECISION = 32
WHOLE = 1 << PRECISION
HALF = WHOLE >> 1
QUARTER = WHOLE >> 2


class Encoder:
    def __init__(self):
        self.low = 0
        self.high = WHOLE - 1
        self.pending = 0
        self.bits = bytearray()
        self._bit_count = 0
        self._current_byte = 0

    def encode_symbol(self, cumulative_low, cumulative_high, total):
        """Encode a symbol given its cumulative frequency range [cum_low, cum_high) out of total."""
        range_ = self.high - self.low + 1
        self.high = self.low + (range_ * cumulative_high) // total - 1
        self.low = self.low + (range_ * cumulative_low) // total
        self._normalize()

    def _emit_bit(self, bit):
        self._current_byte = (self._current_byte << 1) | bit
        self._bit_count += 1
        if self._bit_count == 8:
            self.bits.append(self._current_byte)
            self._current_byte = 0
            self._bit_count = 0

    def _normalize(self):
        while True:
            if self.high < HALF:
                self._emit_bit(0)
                for _ in range(self.pending):
                    self._emit_bit(1)
                self.pending = 0
            elif self.low >= HALF:
                self._emit_bit(1)
                for _ in range(self.pending):
                    self._emit_bit(0)
                self.pending = 0
                self.low -= HALF
                self.high -= HALF
            elif self.low >= QUARTER and self.high < 3 * QUARTER:
                self.pending += 1
                self.low -= QUARTER
                self.high -= QUARTER
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1

    def finish(self):
        """Flush remaining state and return the compressed bytes."""
        self.pending += 1
        if self.low < QUARTER:
            self._emit_bit(0)
            for _ in range(self.pending):
                self._emit_bit(1)
        else:
            self._emit_bit(1)
            for _ in range(self.pending):
                self._emit_bit(0)
        # Pad the last byte
        if self._bit_count > 0:
            self._current_byte <<= (8 - self._bit_count)
            self.bits.append(self._current_byte)
        return bytes(self.bits)


class Decoder:
    def __init__(self, data: bytes):
        self.low = 0
        self.high = WHOLE - 1
        self.data = data
        self._bit_pos = 0
        self._total_bits = len(data) * 8
        # Initialize code value from first PRECISION bits
        self.code = 0
        for _ in range(PRECISION):
            self.code = (self.code << 1) | self._read_bit()

    def _read_bit(self):
        if self._bit_pos < self._total_bits:
            byte_idx = self._bit_pos >> 3
            bit_idx = 7 - (self._bit_pos & 7)
            self._bit_pos += 1
            return (self.data[byte_idx] >> bit_idx) & 1
        return 0

    def get_value(self, total):
        """Return the current cumulative count value for the given total."""
        range_ = self.high - self.low + 1
        return ((self.code - self.low + 1) * total - 1) // range_

    def decode_symbol(self, cumulative_low, cumulative_high, total):
        """Update state after decoding a symbol with the given cumulative range."""
        range_ = self.high - self.low + 1
        self.high = self.low + (range_ * cumulative_high) // total - 1
        self.low = self.low + (range_ * cumulative_low) // total
        self._normalize()

    def _normalize(self):
        while True:
            if self.high < HALF:
                pass
            elif self.low >= HALF:
                self.code -= HALF
                self.low -= HALF
                self.high -= HALF
            elif self.low >= QUARTER and self.high < 3 * QUARTER:
                self.code -= QUARTER
                self.low -= QUARTER
                self.high -= QUARTER
            else:
                break
            self.low <<= 1
            self.high = (self.high << 1) | 1
            self.code = (self.code << 1) | self._read_bit()
