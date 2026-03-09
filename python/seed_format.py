"""Binary .seedmodel format — cross-language seed model serialization.

Format:
    SDML                          <- 4-byte magic
    version: u8                   <- format version (1)
    seed_id: u8                   <- seed type ID (0-255)
    max_order: u8                 <- max PPM order stored
    name_len: u8                  <- length of name string
    name: [u8; name_len]          <- UTF-8 name (e.g. "english")

    For each order 0..max_order:
      num_contexts: u32 (LE)
      For each context:
        context_bytes: [u8; order]
        num_entries: u16 (LE)     <- number of non-zero symbols (sparse)
        For each entry:
          symbol: u8
          count: u16 (LE)
"""

import struct

MAGIC = b"SDML"
FORMAT_VERSION = 1
NUM_SYMBOLS = 256


def write_seed(path, seed_id, name, max_order, counts):
    """Serialize seed counts to a .seedmodel file.

    Args:
        path: Output file path.
        seed_id: Seed type ID (0-255).
        name: Human-readable name (e.g. "english").
        max_order: Max PPM order stored.
        counts: {order: {context_bytes: [256 counts]}}
    """
    name_bytes = name.encode("utf-8")
    buf = bytearray()

    # Header
    buf.extend(MAGIC)
    buf.append(FORMAT_VERSION)
    buf.append(seed_id)
    buf.append(max_order)
    buf.append(len(name_bytes))
    buf.extend(name_bytes)

    # Per-order data
    for order in range(max_order + 1):
        order_counts = counts.get(order, {})
        buf.extend(struct.pack("<I", len(order_counts)))
        for ctx, syms in order_counts.items():
            assert len(ctx) == order, f"context length {len(ctx)} != order {order}"
            buf.extend(ctx)
            # Sparse: only non-zero entries
            entries = [(s, c) for s, c in enumerate(syms) if c > 0]
            buf.extend(struct.pack("<H", len(entries)))
            for symbol, count in entries:
                buf.append(symbol)
                buf.extend(struct.pack("<H", count))

    with open(path, "wb") as f:
        f.write(buf)


def read_seed(path):
    """Deserialize a .seedmodel file.

    Returns:
        (seed_id, name, max_order, counts)
        counts: {order: {context_bytes: [256 counts]}}
    """
    with open(path, "rb") as f:
        data = f.read()

    if data[:4] != MAGIC:
        raise ValueError("Not a .seedmodel file (bad magic)")

    off = 4
    version = data[off]; off += 1
    if version != FORMAT_VERSION:
        raise ValueError(f"Unsupported seedmodel version {version}")

    seed_id = data[off]; off += 1
    max_order = data[off]; off += 1
    name_len = data[off]; off += 1
    name = data[off:off + name_len].decode("utf-8"); off += name_len

    counts = {}
    for order in range(max_order + 1):
        num_contexts = struct.unpack_from("<I", data, off)[0]; off += 4
        order_counts = {}
        for _ in range(num_contexts):
            ctx = bytes(data[off:off + order]); off += order
            num_entries = struct.unpack_from("<H", data, off)[0]; off += 2
            syms = [0] * NUM_SYMBOLS
            for _ in range(num_entries):
                symbol = data[off]; off += 1
                count = struct.unpack_from("<H", data, off)[0]; off += 2
                syms[symbol] = count
            order_counts[ctx] = syms
        counts[order] = order_counts

    return seed_id, name, max_order, counts
