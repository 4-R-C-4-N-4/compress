/// PPM model with PPMD-style escape and seed fallback prior.
///
/// Bit-identical to the Python reference implementation in python/model.py.

use std::collections::HashMap;

use crate::arithmetic::{Decoder, Encoder};
use crate::seed_format::SeedCounts;

pub const NUM_SYMBOLS: usize = 256;
const SEED_WEIGHT: u32 = 256;

/// Scale seed counts so each context's total is at most target_total.
fn scale_seed_order(
    order_counts: &HashMap<Vec<u8>, Vec<u32>>,
    target_total: u32,
) -> HashMap<Vec<u8>, Vec<u32>> {
    let mut scaled = HashMap::new();
    for (ctx, syms) in order_counts {
        let total: u32 = syms.iter().sum();
        if total <= target_total {
            scaled.insert(ctx.clone(), syms.clone());
            continue;
        }
        let factor = target_total as f64 / total as f64;
        let new_syms: Vec<u32> = syms
            .iter()
            .map(|&c| {
                if c > 0 {
                    1u32.max((c as f64 * factor) as u32)
                } else {
                    0
                }
            })
            .collect();
        scaled.insert(ctx.clone(), new_syms);
    }
    scaled
}

struct PPMOrder {
    order: usize,
    seed: HashMap<Vec<u8>, Vec<u32>>,
    counts: HashMap<Vec<u8>, Vec<u32>>,
    totals: HashMap<Vec<u8>, u32>,
}

impl PPMOrder {
    fn new(order: usize, seed_counts: Option<HashMap<Vec<u8>, Vec<u32>>>) -> Self {
        PPMOrder {
            order,
            seed: seed_counts.unwrap_or_default(),
            counts: HashMap::new(),
            totals: HashMap::new(),
        }
    }

    fn get_distribution(
        &self,
        context: &[u8],
        exclusions: &[bool; NUM_SYMBOLS],
    ) -> ([u32; NUM_SYMBOLS], u64, u64) {
        let ctx = if self.order > 0 && context.len() >= self.order {
            &context[context.len() - self.order..]
        } else if self.order == 0 {
            &[]
        } else {
            context
        };
        let ctx_vec = ctx.to_vec();

        let raw = self.counts.get(&ctx_vec);
        let seed_raw = self.seed.get(&ctx_vec);
        let has_adaptive = self.totals.get(&ctx_vec).copied().unwrap_or(0) > 0;

        let mut dist = [0u32; NUM_SYMBOLS];
        let mut total: u64 = 0;
        let mut distinct: u64 = 0;

        for s in 0..NUM_SYMBOLS {
            if exclusions[s] {
                continue;
            }
            let c = if has_adaptive {
                raw.map_or(0, |r| r[s])
            } else {
                seed_raw.map_or(0, |sr| sr[s])
            };
            if c > 0 {
                dist[s] = c;
                total += c as u64;
                distinct += 1;
            }
        }

        (dist, total, distinct)
    }

    fn update(&mut self, context: &[u8], symbol: u8) {
        let ctx = if self.order > 0 && context.len() >= self.order {
            context[context.len() - self.order..].to_vec()
        } else if self.order == 0 {
            vec![]
        } else {
            context.to_vec()
        };
        let entry = self.counts.entry(ctx.clone()).or_insert_with(|| vec![0u32; NUM_SYMBOLS]);
        entry[symbol as usize] += 1;
        *self.totals.entry(ctx).or_insert(0) += 1;
    }
}

fn order_minus1_distribution(exclusions: &[bool; NUM_SYMBOLS]) -> ([u32; NUM_SYMBOLS], u64) {
    let mut dist = [0u32; NUM_SYMBOLS];
    let mut total: u64 = 0;
    for s in 0..NUM_SYMBOLS {
        if !exclusions[s] {
            dist[s] = 1;
            total += 1;
        }
    }
    (dist, total)
}

pub struct PPMModel {
    orders: Vec<PPMOrder>,
    max_order: usize,
}

impl PPMModel {
    pub fn new(max_order: u8, seed_counts: Option<&SeedCounts>) -> Self {
        let max_order_usize = max_order as usize;
        let mut orders = Vec::with_capacity(max_order_usize + 1);
        for o in 0..=max_order_usize {
            let sc = seed_counts.and_then(|sc| sc.get(&(o as u8)).map(|oc| {
                scale_seed_order(oc, SEED_WEIGHT)
            }));
            orders.push(PPMOrder::new(o, sc));
        }
        PPMModel {
            orders,
            max_order: max_order_usize,
        }
    }

    pub fn encode_symbol(&mut self, encoder: &mut Encoder, context: &[u8], symbol: u8) {
        let mut exclusions = [false; NUM_SYMBOLS];
        let sym_idx = symbol as usize;

        for o in (0..=self.max_order).rev() {
            let (counts, total, distinct) = self.orders[o].get_distribution(context, &exclusions);

            if counts[sym_idx] > 0 {
                // Symbol found at this order
                let escape_count = distinct; // PPMD Method D
                let grand_total = total + escape_count;

                let mut cum_low: u64 = 0;
                for s in 0..sym_idx {
                    cum_low += counts[s] as u64;
                }
                let cum_high = cum_low + counts[sym_idx] as u64;

                encoder.encode_symbol(cum_low, cum_high, grand_total);
                self.update_all(context, symbol);
                return;
            }

            if total > 0 || distinct > 0 {
                // Escape
                let escape_count = distinct.max(1);
                let grand_total = total + escape_count;

                encoder.encode_symbol(total, grand_total, grand_total);

                for s in 0..NUM_SYMBOLS {
                    if counts[s] > 0 {
                        exclusions[s] = true;
                    }
                }
            }
            // else: empty context, fall through silently
        }

        // Order -1 fallback
        let (counts, total) = order_minus1_distribution(&exclusions);
        let mut cum_low: u64 = 0;
        for s in 0..sym_idx {
            cum_low += counts[s] as u64;
        }
        let cum_high = cum_low + counts[sym_idx] as u64;
        encoder.encode_symbol(cum_low, cum_high, total);
        self.update_all(context, symbol);
    }

    pub fn decode_symbol(&mut self, decoder: &mut Decoder, context: &[u8]) -> u8 {
        let mut exclusions = [false; NUM_SYMBOLS];

        for o in (0..=self.max_order).rev() {
            let (counts, total, distinct) = self.orders[o].get_distribution(context, &exclusions);

            if total == 0 && distinct == 0 {
                continue;
            }

            let escape_count = distinct.max(1);
            let grand_total = total + escape_count;

            let value = decoder.get_value(grand_total);

            if value < total {
                // Decode actual symbol
                let mut cum: u64 = 0;
                for s in 0..NUM_SYMBOLS {
                    if counts[s] == 0 {
                        continue;
                    }
                    if cum + counts[s] as u64 > value {
                        decoder.decode_symbol(cum, cum + counts[s] as u64, grand_total);
                        self.update_all(context, s as u8);
                        return s as u8;
                    }
                    cum += counts[s] as u64;
                }
            }

            // Escape
            decoder.decode_symbol(total, grand_total, grand_total);
            for s in 0..NUM_SYMBOLS {
                if counts[s] > 0 {
                    exclusions[s] = true;
                }
            }
        }

        // Order -1 fallback
        let (counts, total) = order_minus1_distribution(&exclusions);
        let value = decoder.get_value(total);
        let mut cum: u64 = 0;
        for s in 0..NUM_SYMBOLS {
            if counts[s] == 0 {
                continue;
            }
            if cum + counts[s] as u64 > value {
                decoder.decode_symbol(cum, cum + counts[s] as u64, total);
                self.update_all(context, s as u8);
                return s as u8;
            }
            cum += counts[s] as u64;
        }

        panic!("Decode failed: no symbol found in order -1");
    }

    fn update_all(&mut self, context: &[u8], symbol: u8) {
        for order_model in &mut self.orders {
            order_model.update(context, symbol);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_simple() {
        let data = b"hello world";
        let mut model = PPMModel::new(4, None);
        let mut enc = Encoder::new();

        let mut context: Vec<u8> = Vec::new();
        for &byte in data.iter() {
            model.encode_symbol(&mut enc, &context, byte);
            context.push(byte);
            if context.len() > 4 {
                context = context[context.len() - 4..].to_vec();
            }
        }
        let compressed = enc.finish();

        let mut model2 = PPMModel::new(4, None);
        let mut dec = Decoder::new(&compressed);
        let mut context2: Vec<u8> = Vec::new();
        let mut output = Vec::new();
        for _ in 0..data.len() {
            let sym = model2.decode_symbol(&mut dec, &context2);
            output.push(sym);
            context2.push(sym);
            if context2.len() > 4 {
                context2 = context2[context2.len() - 4..].to_vec();
            }
        }
        assert_eq!(output, data);
    }

    #[test]
    fn test_roundtrip_empty() {
        let model = PPMModel::new(4, None);
        let enc = Encoder::new();
        let compressed = enc.finish();
        assert!(!compressed.is_empty()); // at least the finish bits
        drop(model);
    }
}
