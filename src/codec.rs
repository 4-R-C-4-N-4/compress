/// High-level encode/decode with .seed file format.
///
/// Bit-identical to the Python reference implementation in python/codec.py.

use std::fs;
use std::path::{Path, PathBuf};

use blake2::digest::{Update, VariableOutput};
use blake2::Blake2bVar;

use crate::arithmetic::{Decoder, Encoder};
use crate::model::PPMModel;
use crate::seed_format::{self, SeedCounts};

const MAGIC: &[u8; 4] = b"SEED";
const SEPARATOR: &[u8; 3] = b"---";
const FINGERPRINT_LEN: usize = 8;
const AUTO_PROBE_SIZE: usize = 1024;

fn fingerprint(data: &[u8]) -> [u8; FINGERPRINT_LEN] {
    let mut hasher = Blake2bVar::new(FINGERPRINT_LEN).unwrap();
    hasher.update(data);
    let mut buf = [0u8; FINGERPRINT_LEN];
    hasher.finalize_variable(&mut buf).unwrap();
    buf
}

fn seeds_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("seeds")
}

pub struct SeedInfo {
    pub name: String,
    pub seed_id: u8,
    pub max_order: u8,
    pub counts: SeedCounts,
}

pub fn list_seeds() -> Vec<SeedInfo> {
    let dir = seeds_dir();
    let mut seeds = Vec::new();
    if !dir.is_dir() {
        return seeds;
    }
    let mut entries: Vec<_> = fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map_or(false, |ext| ext == "seedmodel")
        })
        .collect();
    entries.sort_by_key(|e| e.file_name());

    for entry in entries {
        if let Ok(model) = seed_format::read_seed(&entry.path()) {
            seeds.push(SeedInfo {
                name: model.name,
                seed_id: model.seed_id,
                max_order: model.max_order,
                counts: model.counts,
            });
        }
    }
    seeds
}

pub fn load_seed(seed_id: u8) -> Option<SeedCounts> {
    if seed_id == 0 {
        return None;
    }
    for seed in list_seeds() {
        if seed.seed_id == seed_id {
            return Some(seed.counts);
        }
    }
    None
}

pub fn load_seed_by_name(name: &str) -> Option<(u8, u8, SeedCounts)> {
    for seed in list_seeds() {
        if seed.name == name {
            return Some((seed.seed_id, seed.max_order, seed.counts));
        }
    }
    None
}

fn probe_size(data: &[u8], max_order: u8, seed_counts: Option<&SeedCounts>) -> usize {
    let mut model = PPMModel::new(max_order, seed_counts);
    let mut enc = Encoder::new();
    let mut context: Vec<u8> = Vec::new();
    let max_order_usize = max_order as usize;
    for &byte in data {
        model.encode_symbol(&mut enc, &context, byte);
        context.push(byte);
        if context.len() > max_order_usize {
            context = context[context.len() - max_order_usize..].to_vec();
        }
    }
    enc.finish().len()
}

pub fn auto_select_seed(data: &[u8], max_order: u8) -> (String, u8, Option<SeedCounts>) {
    let probe = if data.len() > AUTO_PROBE_SIZE {
        &data[..AUTO_PROBE_SIZE]
    } else {
        data
    };
    if probe.is_empty() {
        return ("null".to_string(), 0, None);
    }

    let mut best_name = "null".to_string();
    let mut best_id = 0u8;
    let mut best_counts: Option<SeedCounts> = None;
    let mut best_size = probe_size(probe, max_order, None);

    for seed in list_seeds() {
        if seed.name == "null" {
            continue;
        }
        let size = probe_size(probe, max_order, Some(&seed.counts));
        if size < best_size {
            best_size = size;
            best_name = seed.name;
            best_id = seed.seed_id;
            best_counts = Some(seed.counts);
        }
    }

    (best_name, best_id, best_counts)
}

pub fn resolve_seed(seed_arg: &str) -> Result<(u8, Option<SeedCounts>, bool), String> {
    if seed_arg == "auto" {
        return Ok((0, None, true));
    }
    // Try as numeric ID
    if let Ok(sid) = seed_arg.parse::<u8>() {
        return Ok((sid, load_seed(sid), false));
    }
    // Try as name
    if let Some((sid, _order, counts)) = load_seed_by_name(seed_arg) {
        return Ok((sid, Some(counts), false));
    }
    Err(format!("Unknown seed: {:?} (not a known ID or name)", seed_arg))
}

pub fn encode(
    data: &[u8],
    seed_id: u8,
    max_order: u8,
    seed_counts: Option<&SeedCounts>,
    auto: bool,
) -> Vec<u8> {
    let (actual_seed_id, actual_counts);

    if auto {
        let (_name, sid, sc) = auto_select_seed(data, max_order);
        actual_seed_id = sid;
        actual_counts = sc;
    } else if seed_counts.is_some() {
        actual_seed_id = seed_id;
        actual_counts = None; // we'll use the passed-in counts
    } else {
        actual_seed_id = seed_id;
        actual_counts = load_seed(seed_id);
    }

    let counts_ref = if seed_counts.is_some() && !auto {
        seed_counts
    } else {
        actual_counts.as_ref()
    };

    let mut model = PPMModel::new(max_order, counts_ref);
    let mut enc = Encoder::new();

    let max_order_usize = max_order as usize;
    let mut context: Vec<u8> = Vec::new();
    for &byte in data {
        model.encode_symbol(&mut enc, &context, byte);
        context.push(byte);
        if context.len() > max_order_usize {
            context = context[context.len() - max_order_usize..].to_vec();
        }
    }

    let bitstream = enc.finish();
    let fp = fingerprint(data);

    let sid = if auto { actual_seed_id } else { seed_id };

    let mut header = Vec::new();
    header.extend_from_slice(MAGIC);
    header.push(sid);
    header.push(max_order);
    header.extend_from_slice(&(data.len() as u32).to_le_bytes());
    header.extend_from_slice(&fp);
    header.extend_from_slice(SEPARATOR);

    header.extend_from_slice(&bitstream);
    header
}

pub fn decode(compressed: &[u8], seed_counts: Option<&SeedCounts>) -> Result<Vec<u8>, String> {
    if compressed.len() < 4 || &compressed[..4] != MAGIC {
        return Err("Not a .seed file (bad magic)".into());
    }

    let mut offset = 4;
    let seed_id = compressed[offset];
    offset += 1;
    let max_order = compressed[offset];
    offset += 1;

    if offset + 4 > compressed.len() {
        return Err("Truncated header".into());
    }
    let byte_length = u32::from_le_bytes([
        compressed[offset],
        compressed[offset + 1],
        compressed[offset + 2],
        compressed[offset + 3],
    ]) as usize;
    offset += 4;

    if offset + FINGERPRINT_LEN > compressed.len() {
        return Err("Truncated fingerprint".into());
    }
    let mut fp = [0u8; FINGERPRINT_LEN];
    fp.copy_from_slice(&compressed[offset..offset + FINGERPRINT_LEN]);
    offset += FINGERPRINT_LEN;

    if offset + 3 > compressed.len() || &compressed[offset..offset + 3] != SEPARATOR {
        return Err("Missing separator in .seed header".into());
    }
    offset += 3;

    let bitstream = &compressed[offset..];

    let loaded_counts;
    let counts_ref = if seed_counts.is_some() {
        seed_counts
    } else {
        loaded_counts = load_seed(seed_id);
        loaded_counts.as_ref()
    };

    let mut model = PPMModel::new(max_order, counts_ref);
    let mut dec = Decoder::new(bitstream);

    let mut output = Vec::with_capacity(byte_length);
    let max_order_usize = max_order as usize;
    let mut context: Vec<u8> = Vec::new();
    for _ in 0..byte_length {
        let symbol = model.decode_symbol(&mut dec, &context);
        output.push(symbol);
        context.push(symbol);
        if context.len() > max_order_usize {
            context = context[context.len() - max_order_usize..].to_vec();
        }
    }

    if fingerprint(&output) != fp {
        return Err("Fingerprint mismatch — data corrupted".into());
    }

    Ok(output)
}

pub fn read_recipe(path: &Path) -> Result<(u8, SeedCounts), String> {
    let model = seed_format::read_seed(path)?;
    Ok((model.max_order, model.counts))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_hello_world() {
        let data = b"hello world";
        let compressed = encode(data, 0, 4, None, false);
        let result = decode(&compressed, None).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_roundtrip_empty() {
        let data = b"";
        let compressed = encode(data, 0, 4, None, false);
        let result = decode(&compressed, None).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_roundtrip_single_byte() {
        let data = b"A";
        let compressed = encode(data, 0, 4, None, false);
        let result = decode(&compressed, None).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_roundtrip_zeros() {
        let data = vec![0u8; 256];
        let compressed = encode(&data, 0, 4, None, false);
        let result = decode(&compressed, None).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_roundtrip_with_english_seed() {
        let counts = load_seed(1);
        if counts.is_none() {
            return; // skip if seed not available
        }
        let data = b"The quick brown fox jumps over the lazy dog.";
        let compressed = encode(data, 1, 4, counts.as_ref(), false);
        let result = decode(&compressed, counts.as_ref()).unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_bad_magic() {
        let result = decode(b"XXXX\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00", None);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("bad magic"));
    }

    #[test]
    fn test_bad_fingerprint() {
        let data = b"hello world";
        let mut compressed = encode(data, 0, 4, None, false);
        compressed[10] ^= 0xFF; // corrupt fingerprint
        let result = decode(&compressed, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Fingerprint mismatch"));
    }

    #[test]
    fn test_header_structure() {
        let data = b"test data for header";
        let compressed = encode(data, 0, 3, None, false);

        assert_eq!(&compressed[..4], b"SEED");
        assert_eq!(compressed[4], 0); // seed_id
        assert_eq!(compressed[5], 3); // order
        let byte_length = u32::from_le_bytes([compressed[6], compressed[7], compressed[8], compressed[9]]);
        assert_eq!(byte_length as usize, data.len());
        let fp = &compressed[10..10 + FINGERPRINT_LEN];
        assert_eq!(fp, &fingerprint(data));
        assert_eq!(&compressed[10 + FINGERPRINT_LEN..10 + FINGERPRINT_LEN + 3], b"---");
    }

    #[test]
    fn test_auto_roundtrip() {
        let data = b"The quick brown fox jumps over the lazy dog. ".repeat(20);
        let compressed = encode(&data, 0, 4, None, true);
        let result = decode(&compressed, None).unwrap();
        assert_eq!(result, data);
    }
}
