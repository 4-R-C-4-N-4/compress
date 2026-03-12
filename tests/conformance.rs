/// Cross-language conformance tests.
///
/// Verifies that the Rust implementation produces bit-identical output
/// to the Python reference implementation for the same inputs and seeds.

use std::fs;
use std::path::PathBuf;

use serde::Deserialize;

use seedac::codec;

#[derive(Deserialize)]
struct TestVector {
    name: String,
    input_hex: String,
    compressed_hex: String,
    seed_id: u8,
    order: u8,
}

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
}

fn load_vectors() -> Vec<TestVector> {
    let path = fixtures_dir().join("vectors.json");
    let data = fs::read_to_string(&path).expect("Failed to read vectors.json");
    serde_json::from_str(&data).expect("Failed to parse vectors.json")
}

fn hex_to_bytes(hex: &str) -> Vec<u8> {
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
        .collect()
}

#[test]
fn test_bit_identical_encode() {
    let vectors = load_vectors();
    for v in &vectors {
        let input = hex_to_bytes(&v.input_hex);
        let expected = hex_to_bytes(&v.compressed_hex);

        let seed_counts = codec::load_seed(v.seed_id);
        let compressed = codec::encode(&input, v.seed_id, v.order, seed_counts.as_ref(), false);

        assert_eq!(
            compressed, expected,
            "Bit-identity failed for vector '{}': Rust output differs from Python.\n\
             Rust   ({} bytes): {}\n\
             Python ({} bytes): {}",
            v.name,
            compressed.len(),
            hex::encode(&compressed),
            expected.len(),
            hex::encode(&expected),
        );
    }
}

#[test]
fn test_decode_python_output() {
    let vectors = load_vectors();
    for v in &vectors {
        let input = hex_to_bytes(&v.input_hex);
        let compressed = hex_to_bytes(&v.compressed_hex);

        let seed_counts = codec::load_seed(v.seed_id);
        let result = codec::decode(&compressed, seed_counts.as_ref())
            .unwrap_or_else(|e| panic!("Decode failed for vector '{}': {}", v.name, e));

        assert_eq!(
            result, input,
            "Decode mismatch for vector '{}'",
            v.name
        );
    }
}

/// Helper module for hex encoding in error messages
mod hex {
    pub fn encode(data: &[u8]) -> String {
        data.iter().map(|b| format!("{:02x}", b)).collect()
    }
}
