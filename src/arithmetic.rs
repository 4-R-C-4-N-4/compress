/// Range-based arithmetic coder with 32-bit precision.
///
/// Bit-identical to the Python reference implementation in python/arithmetic.py.

const PRECISION: u32 = 32;
const WHOLE: u64 = 1 << PRECISION;
const HALF: u64 = WHOLE >> 1;
const QUARTER: u64 = WHOLE >> 2;

pub struct Encoder {
    low: u64,
    high: u64,
    pending: u64,
    bits: Vec<u8>,
    bit_count: u8,
    current_byte: u8,
}

impl Encoder {
    pub fn new() -> Self {
        Encoder {
            low: 0,
            high: WHOLE - 1,
            pending: 0,
            bits: Vec::new(),
            bit_count: 0,
            current_byte: 0,
        }
    }

    pub fn encode_symbol(&mut self, cum_low: u64, cum_high: u64, total: u64) {
        let range = self.high - self.low + 1;
        self.high = self.low + (range as u128 * cum_high as u128 / total as u128) as u64 - 1;
        self.low = self.low + (range as u128 * cum_low as u128 / total as u128) as u64;
        self.normalize();
    }

    fn emit_bit(&mut self, bit: u8) {
        self.current_byte = (self.current_byte << 1) | bit;
        self.bit_count += 1;
        if self.bit_count == 8 {
            self.bits.push(self.current_byte);
            self.current_byte = 0;
            self.bit_count = 0;
        }
    }

    fn normalize(&mut self) {
        loop {
            if self.high < HALF {
                self.emit_bit(0);
                for _ in 0..self.pending {
                    self.emit_bit(1);
                }
                self.pending = 0;
            } else if self.low >= HALF {
                self.emit_bit(1);
                for _ in 0..self.pending {
                    self.emit_bit(0);
                }
                self.pending = 0;
                self.low -= HALF;
                self.high -= HALF;
            } else if self.low >= QUARTER && self.high < 3 * QUARTER {
                self.pending += 1;
                self.low -= QUARTER;
                self.high -= QUARTER;
            } else {
                break;
            }
            self.low <<= 1;
            self.high = (self.high << 1) | 1;
        }
    }

    pub fn finish(mut self) -> Vec<u8> {
        self.pending += 1;
        if self.low < QUARTER {
            self.emit_bit(0);
            for _ in 0..self.pending {
                self.emit_bit(1);
            }
        } else {
            self.emit_bit(1);
            for _ in 0..self.pending {
                self.emit_bit(0);
            }
        }
        if self.bit_count > 0 {
            self.current_byte <<= 8 - self.bit_count;
            self.bits.push(self.current_byte);
        }
        self.bits
    }
}

pub struct Decoder<'a> {
    low: u64,
    high: u64,
    code: u64,
    data: &'a [u8],
    bit_pos: usize,
    total_bits: usize,
}

impl<'a> Decoder<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        let total_bits = data.len() * 8;
        let mut dec = Decoder {
            low: 0,
            high: WHOLE - 1,
            code: 0,
            data,
            bit_pos: 0,
            total_bits,
        };
        for _ in 0..PRECISION {
            dec.code = (dec.code << 1) | dec.read_bit();
        }
        dec
    }

    fn read_bit(&mut self) -> u64 {
        if self.bit_pos < self.total_bits {
            let byte_idx = self.bit_pos >> 3;
            let bit_idx = 7 - (self.bit_pos & 7);
            self.bit_pos += 1;
            ((self.data[byte_idx] >> bit_idx) & 1) as u64
        } else {
            0
        }
    }

    pub fn get_value(&self, total: u64) -> u64 {
        let range = self.high - self.low + 1;
        ((((self.code - self.low + 1) as u128) * (total as u128) - 1) / (range as u128)) as u64
    }

    pub fn decode_symbol(&mut self, cum_low: u64, cum_high: u64, total: u64) {
        let range = self.high - self.low + 1;
        self.high = self.low + (range as u128 * cum_high as u128 / total as u128) as u64 - 1;
        self.low = self.low + (range as u128 * cum_low as u128 / total as u128) as u64;
        self.normalize();
    }

    fn normalize(&mut self) {
        loop {
            if self.high < HALF {
                // pass
            } else if self.low >= HALF {
                self.code -= HALF;
                self.low -= HALF;
                self.high -= HALF;
            } else if self.low >= QUARTER && self.high < 3 * QUARTER {
                self.code -= QUARTER;
                self.low -= QUARTER;
                self.high -= QUARTER;
            } else {
                break;
            }
            self.low <<= 1;
            self.high = (self.high << 1) | 1;
            self.code = (self.code << 1) | self.read_bit();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_binary() {
        // Encode sequence [0, 1, 0, 1, 1] with P(0)=1/2, P(1)=1/2
        let symbols = [0u64, 1, 0, 1, 1];
        let mut enc = Encoder::new();
        for &s in &symbols {
            let (cl, ch) = if s == 0 { (0, 1) } else { (1, 2) };
            enc.encode_symbol(cl, ch, 2);
        }
        let data = enc.finish();

        let mut dec = Decoder::new(&data);
        for &expected in &symbols {
            let v = dec.get_value(2);
            let (cl, ch) = if v < 1 { (0u64, 1u64) } else { (1, 2) };
            let sym = if v < 1 { 0 } else { 1 };
            assert_eq!(sym, expected);
            dec.decode_symbol(cl, ch, 2);
        }
    }

    #[test]
    fn test_roundtrip_skewed() {
        // Skewed: P(A)=99/100, P(B)=1/100
        let symbols: Vec<u64> = vec![0; 50];
        let mut enc = Encoder::new();
        for &s in &symbols {
            let (cl, ch) = if s == 0 { (0, 99) } else { (99, 100) };
            enc.encode_symbol(cl, ch, 100);
        }
        let data = enc.finish();

        let mut dec = Decoder::new(&data);
        for &expected in &symbols {
            let v = dec.get_value(100);
            let sym = if v < 99 { 0 } else { 1 };
            assert_eq!(sym, expected);
            let (cl, ch) = if sym == 0 { (0u64, 99u64) } else { (99, 100) };
            dec.decode_symbol(cl, ch, 100);
        }
    }
}
