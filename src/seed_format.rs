/// Binary .seedmodel format reader.
///
/// Bit-identical to the Python reference implementation in python/seed_format.py.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

const MAGIC: &[u8; 4] = b"SDML";
const FORMAT_VERSION: u8 = 1;
pub const NUM_SYMBOLS: usize = 256;

/// Seed counts: order -> (context_bytes -> [256 counts])
pub type SeedCounts = HashMap<u8, HashMap<Vec<u8>, Vec<u32>>>;

#[derive(Debug)]
pub struct SeedModel {
    pub seed_id: u8,
    pub name: String,
    pub max_order: u8,
    pub counts: SeedCounts,
}

pub fn read_seed(path: &Path) -> Result<SeedModel, String> {
    let data = fs::read(path).map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;
    parse_seed(&data)
}

pub fn parse_seed(data: &[u8]) -> Result<SeedModel, String> {
    if data.len() < 8 || &data[..4] != MAGIC {
        return Err("Not a .seedmodel file (bad magic)".into());
    }

    let mut off = 4;
    let version = data[off];
    off += 1;
    if version != FORMAT_VERSION {
        return Err(format!("Unsupported seedmodel version {}", version));
    }

    let seed_id = data[off];
    off += 1;
    let max_order = data[off];
    off += 1;
    let name_len = data[off] as usize;
    off += 1;

    if off + name_len > data.len() {
        return Err("Truncated name".into());
    }
    let name = String::from_utf8(data[off..off + name_len].to_vec())
        .map_err(|_| "Invalid UTF-8 in name")?;
    off += name_len;

    let mut counts: SeedCounts = HashMap::new();

    for order in 0..=max_order {
        if off + 4 > data.len() {
            return Err("Truncated data".into());
        }
        let num_contexts = u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]) as usize;
        off += 4;

        let mut order_counts: HashMap<Vec<u8>, Vec<u32>> = HashMap::new();
        let order_usize = order as usize;

        for _ in 0..num_contexts {
            if off + order_usize > data.len() {
                return Err("Truncated context".into());
            }
            let ctx = data[off..off + order_usize].to_vec();
            off += order_usize;

            if off + 2 > data.len() {
                return Err("Truncated entries header".into());
            }
            let num_entries = u16::from_le_bytes([data[off], data[off + 1]]) as usize;
            off += 2;

            let mut syms = vec![0u32; NUM_SYMBOLS];
            for _ in 0..num_entries {
                if off + 3 > data.len() {
                    return Err("Truncated entry".into());
                }
                let symbol = data[off] as usize;
                off += 1;
                let count = u16::from_le_bytes([data[off], data[off + 1]]) as u32;
                off += 2;
                syms[symbol] = count;
            }
            order_counts.insert(ctx, syms);
        }
        counts.insert(order, order_counts);
    }

    Ok(SeedModel {
        seed_id,
        name,
        max_order,
        counts,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn seeds_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("seeds")
    }

    #[test]
    fn test_read_null() {
        let path = seeds_dir().join("null.seedmodel");
        let seed = read_seed(&path).expect("Failed to read null.seedmodel");
        assert_eq!(seed.seed_id, 0);
        assert_eq!(seed.name, "null");
        assert_eq!(seed.max_order, 4);
    }

    #[test]
    fn test_read_english() {
        let path = seeds_dir().join("english.seedmodel");
        let seed = read_seed(&path).expect("Failed to read english.seedmodel");
        assert_eq!(seed.seed_id, 1);
        assert_eq!(seed.name, "english");
        assert_eq!(seed.max_order, 4);
        // Should have contexts at order 0
        let order0 = seed.counts.get(&0).unwrap();
        assert!(!order0.is_empty());
    }

    #[test]
    fn test_bad_magic() {
        let data = b"BAAD\x01\x00\x04\x04null";
        let result = parse_seed(data);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("bad magic"));
    }
}
