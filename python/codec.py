"""High-level encode/decode with .seed file format."""

import os
import struct
import hashlib

from .arithmetic import Encoder, Decoder
from .model import PPMModel
from .seed_format import read_seed

MAGIC = b"SEED"
SEPARATOR = b"---"
FINGERPRINT_LEN = 8  # truncated hash

SEEDS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "seeds")


def _fingerprint(data: bytes) -> bytes:
    return hashlib.blake2b(data, digest_size=FINGERPRINT_LEN).digest()


def load_seed(seed_id):
    """Load a .seedmodel file for the given seed_id from the seeds/ directory.

    Returns seed_counts dict suitable for PPMModel, or None if seed_id is 0
    or no matching file is found.
    """
    if seed_id == 0:
        return None

    # Search seeds/ directory for a file with matching seed_id
    if not os.path.isdir(SEEDS_DIR):
        return None

    for fname in os.listdir(SEEDS_DIR):
        if not fname.endswith(".seedmodel"):
            continue
        path = os.path.join(SEEDS_DIR, fname)
        try:
            file_seed_id, _, _, counts = read_seed(path)
            if file_seed_id == seed_id:
                return counts
        except (ValueError, OSError):
            continue

    return None


AUTO_SEED_ID = 255
AUTO_PROBE_SIZE = 1024


def _list_seeds():
    """Return list of (seed_id, counts) for all available seeds."""
    seeds = [(0, None)]
    if not os.path.isdir(SEEDS_DIR):
        return seeds
    for fname in os.listdir(SEEDS_DIR):
        if not fname.endswith(".seedmodel"):
            continue
        path = os.path.join(SEEDS_DIR, fname)
        try:
            sid, _, _, counts = read_seed(path)
            if sid != 0:
                seeds.append((sid, counts))
        except (ValueError, OSError):
            continue
    return seeds


def _probe_size(data, max_order, seed_id, seed_counts):
    """Encode data and return compressed bitstream length in bytes."""
    model = PPMModel(max_order=max_order, seed_counts=seed_counts)
    enc = Encoder()
    context = b""
    for byte in data:
        model.encode_symbol(enc, context, byte)
        context = (context + bytes([byte]))[-max_order:]
    return len(enc.finish())


def auto_select_seed(data, max_order=4):
    """Try all available seeds on a probe of data, return (best_seed_id, best_seed_counts)."""
    probe = data[:AUTO_PROBE_SIZE]
    if not probe:
        return 0, None

    best_id = 0
    best_counts = None
    best_size = _probe_size(probe, max_order, 0, None)

    for sid, counts in _list_seeds():
        if sid == 0:
            continue
        size = _probe_size(probe, max_order, sid, counts)
        if size < best_size:
            best_size = size
            best_id = sid
            best_counts = counts

    return best_id, best_counts


def encode(data: bytes, seed_id: int = 0, max_order: int = 4, seed_counts=None) -> bytes:
    """Compress data into .seed format.

    If seed_id=255 (auto), probes all available seeds and picks the best.
    """
    if seed_id == AUTO_SEED_ID:
        seed_id, seed_counts = auto_select_seed(data, max_order)
    if seed_counts is None:
        seed_counts = load_seed(seed_id)
    model = PPMModel(max_order=max_order, seed_counts=seed_counts)
    enc = Encoder()

    context = b""
    for byte in data:
        model.encode_symbol(enc, context, byte)
        context = (context + bytes([byte]))[-max_order:]

    bitstream = enc.finish()
    fp = _fingerprint(data)

    # Build header
    header = bytearray()
    header.extend(MAGIC)
    header.append(seed_id)
    header.append(max_order)
    header.extend(struct.pack("<I", len(data)))
    header.extend(fp)
    header.extend(SEPARATOR)

    return bytes(header) + bitstream


def decode(compressed: bytes, seed_counts=None) -> bytes:
    """Decompress .seed format back to original data."""
    if compressed[:4] != MAGIC:
        raise ValueError("Not a .seed file (bad magic)")

    offset = 4
    seed_id = compressed[offset]; offset += 1
    max_order = compressed[offset]; offset += 1
    byte_length = struct.unpack("<I", compressed[offset:offset + 4])[0]; offset += 4
    fp = compressed[offset:offset + FINGERPRINT_LEN]; offset += FINGERPRINT_LEN

    if compressed[offset:offset + 3] != SEPARATOR:
        raise ValueError("Missing separator in .seed header")
    offset += 3

    bitstream = compressed[offset:]

    if seed_counts is None:
        seed_counts = load_seed(seed_id)
    model = PPMModel(max_order=max_order, seed_counts=seed_counts)
    dec = Decoder(bitstream)

    output = bytearray()
    context = b""
    for _ in range(byte_length):
        symbol = model.decode_symbol(dec, context)
        output.append(symbol)
        context = (context + bytes([symbol]))[-max_order:]

    result = bytes(output)

    if _fingerprint(result) != fp:
        raise ValueError("Fingerprint mismatch — data corrupted")

    return result
