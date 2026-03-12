"""Benchmark seedac against zlib, bzip2, and zstd on various data types."""

import os
import sys
import time
import zlib
import bz2

try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False

from python.codec import encode, load_seed, auto_select_seed
from python.train import train_model, extract_counts, prune_counts, quantize_counts

SEEDS_DIR = os.path.join(os.path.dirname(__file__), "seeds")

# Map of seed names to IDs for display
SEED_NAMES = {0: "null", 1: "english", 2: "code", 3: "json", 4: "binary", 5: "log"}


def bench_seedac(data, seed_id=0):
    t0 = time.perf_counter()
    compressed = encode(data, seed_id=seed_id)
    elapsed = time.perf_counter() - t0
    return len(compressed), elapsed


def bench_seedac_auto(data):
    t0 = time.perf_counter()
    best_name, best_id, best_counts = auto_select_seed(data)
    compressed = encode(data, seed_id=best_id, seed_counts=best_counts)
    elapsed = time.perf_counter() - t0
    return len(compressed), elapsed, best_name


def bench_seedac_recipe(data, recipe_counts):
    t0 = time.perf_counter()
    compressed = encode(data, seed_id=0, seed_counts=recipe_counts)
    elapsed = time.perf_counter() - t0
    return len(compressed), elapsed


def make_recipe(data, max_order=4):
    """Create seed_counts from data."""
    model = train_model(data, max_order)
    counts = extract_counts(model)
    counts = prune_counts(counts, 1)
    counts = quantize_counts(counts)
    return counts


def bench_zlib(data, level=6):
    t0 = time.perf_counter()
    compressed = zlib.compress(data, level)
    elapsed = time.perf_counter() - t0
    return len(compressed), elapsed


def bench_bzip2(data, level=9):
    t0 = time.perf_counter()
    compressed = bz2.compress(data, level)
    elapsed = time.perf_counter() - t0
    return len(compressed), elapsed


def bench_zstd(data, level=3):
    if not HAS_ZSTD:
        return None, None
    t0 = time.perf_counter()
    cctx = zstd.ZstdCompressor(level=level)
    compressed = cctx.compress(data)
    elapsed = time.perf_counter() - t0
    return len(compressed), elapsed


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

def make_test_data():
    """Return list of (label, data) tuples."""
    samples = []

    # English prose
    english_1k = (
        b"It was the best of times, it was the worst of times, "
        b"it was the age of wisdom, it was the age of foolishness, "
        b"it was the epoch of belief, it was the epoch of incredulity, "
        b"it was the season of Light, it was the season of Darkness, "
        b"it was the spring of hope, it was the winter of despair, "
        b"we had everything before us, we had nothing before us, "
        b"we were all going direct to Heaven, we were all going direct "
        b"the other way. In short, the period was so far like the present "
        b"period, that some of its noisiest authorities insisted on its "
        b"being received, for good or for evil, in the superlative degree "
        b"of comparison only. There were a king with a large jaw and a "
        b"queen with a plain face, on the throne of England; there were "
        b"a king with a large jaw and a queen with a fair face, on the "
        b"throne of France. In both countries it was clearer than crystal "
        b"to the lords of the State preserves of loaves and fishes, that "
        b"things in general were settled for ever."
    )
    samples.append(("english 1K", english_1k))
    samples.append(("english 5K", english_1k * 5))

    # Source code
    code_sample = (
        b"def fibonacci(n):\n"
        b"    if n <= 1:\n"
        b"        return n\n"
        b"    a, b = 0, 1\n"
        b"    for _ in range(2, n + 1):\n"
        b"        a, b = b, a + b\n"
        b"    return b\n\n"
        b"class TreeNode:\n"
        b"    def __init__(self, val=0, left=None, right=None):\n"
        b"        self.val = val\n"
        b"        self.left = left\n"
        b"        self.right = right\n\n"
        b"    def insert(self, val):\n"
        b"        if val < self.val:\n"
        b"            if self.left is None:\n"
        b"                self.left = TreeNode(val)\n"
        b"            else:\n"
        b"                self.left.insert(val)\n"
        b"        else:\n"
        b"            if self.right is None:\n"
        b"                self.right = TreeNode(val)\n"
        b"            else:\n"
        b"                self.right.insert(val)\n\n"
        b"import json\nimport os\nimport sys\n\n"
        b"def main():\n"
        b"    parser = argparse.ArgumentParser()\n"
        b"    parser.add_argument('--input', required=True)\n"
        b"    args = parser.parse_args()\n"
        b"    with open(args.input) as f:\n"
        b"        data = json.load(f)\n"
        b"    print(json.dumps(data, indent=2))\n"
    )
    samples.append(("code 1K", code_sample[:1024]))
    samples.append(("code 5K", (code_sample * 8)[:5120]))

    # JSON
    json_sample = (
        b'{"users": [\n'
        b'  {"id": 1, "name": "Alice", "email": "alice@example.com", "age": 30, "active": true},\n'
        b'  {"id": 2, "name": "Bob", "email": "bob@example.com", "age": 25, "active": false},\n'
        b'  {"id": 3, "name": "Charlie", "email": "charlie@example.com", "age": 35, "active": true},\n'
        b'  {"id": 4, "name": "Diana", "email": "diana@example.com", "age": 28, "active": true},\n'
        b'  {"id": 5, "name": "Eve", "email": "eve@example.com", "age": 32, "active": false}\n'
        b'],\n'
        b'"metadata": {"total": 5, "page": 1, "per_page": 10, "timestamp": "2024-01-15T10:30:00Z"}}\n'
    )
    samples.append(("json 500B", json_sample))
    samples.append(("json 5K", (json_sample * 12)[:5120]))

    # Log lines
    log_sample = (
        b"2024-01-15 10:30:00.123 INFO  [main] Server started on port 8080\n"
        b"2024-01-15 10:30:01.456 DEBUG [pool-1] Connection established from 192.168.1.100\n"
        b"2024-01-15 10:30:02.789 INFO  [handler-3] GET /api/users 200 12ms\n"
        b"2024-01-15 10:30:03.012 WARN  [handler-5] Slow query: SELECT * FROM users WHERE id=42 (234ms)\n"
        b"2024-01-15 10:30:04.345 ERROR [handler-2] NullPointerException at UserService.java:42\n"
        b"2024-01-15 10:30:05.678 INFO  [handler-1] POST /api/login 200 45ms\n"
        b"2024-01-15 10:30:06.901 DEBUG [pool-1] Connection closed from 192.168.1.100\n"
        b"2024-01-15 10:30:07.234 INFO  [handler-4] GET /api/health 200 2ms\n"
    )
    samples.append(("log 500B", log_sample[:512]))
    samples.append(("log 5K", (log_sample * 10)[:5120]))

    # Random (incompressible baseline)
    import random
    r = random.Random(42)
    random_data = bytes(r.randint(0, 255) for _ in range(1024))
    samples.append(("random 1K", random_data))

    # All zeros (best case)
    samples.append(("zeros 1K", b"\x00" * 1024))

    return samples


def make_recipe_pairs():
    """Return list of (label, reference, target) for recipe benchmarks.

    Each pair has a reference file (used to build the recipe) and a target
    file (compressed with that recipe). The target is similar but not identical.
    """
    pairs = []

    # Config files — same schema, different values
    config_ref = b'{"server": "prod-east-1", "port": 8080, "debug": false, "workers": 4, "db": "postgres://prod:5432/app"}\n' * 20
    config_tgt = b'{"server": "prod-west-2", "port": 9090, "debug": true, "workers": 8, "db": "postgres://prod:5432/app"}\n' * 20
    pairs.append(("config", config_ref, config_tgt))

    # Log batches — same service, different timestamps/details
    log_ref = (
        b"2024-01-15 10:30:00 INFO  [handler-1] GET /api/users 200 12ms\n"
        b"2024-01-15 10:30:01 INFO  [handler-2] POST /api/login 200 45ms\n"
        b"2024-01-15 10:30:02 WARN  [handler-3] GET /api/search 200 234ms\n"
        b"2024-01-15 10:30:03 INFO  [handler-1] GET /api/health 200 2ms\n"
    ) * 10
    log_tgt = (
        b"2024-01-16 14:22:10 INFO  [handler-4] GET /api/users 200 8ms\n"
        b"2024-01-16 14:22:11 INFO  [handler-1] POST /api/login 401 30ms\n"
        b"2024-01-16 14:22:12 ERROR [handler-2] GET /api/search 500 15ms\n"
        b"2024-01-16 14:22:13 INFO  [handler-3] GET /api/health 200 1ms\n"
    ) * 10
    pairs.append(("log batch", log_ref, log_tgt))

    # API responses — same schema, different data
    api_ref = b''.join(
        b'{"id": %d, "name": "User%d", "email": "user%d@example.com", "score": %d}\n' % (i, i, i, i * 10)
        for i in range(50)
    )
    api_tgt = b''.join(
        b'{"id": %d, "name": "User%d", "email": "user%d@example.com", "score": %d}\n' % (i + 100, i + 100, i + 100, i * 7)
        for i in range(50)
    )
    pairs.append(("API resp", api_ref, api_tgt))

    # Versioned doc — minor edits
    doc_ref = b"The quick brown fox jumps over the lazy dog. " * 40
    doc_tgt = b"The quick brown cat leaps over the lazy dog. " * 40
    pairs.append(("doc edit", doc_ref, doc_tgt))

    # Dissimilar — recipe from english, target is code (worst case)
    code_tgt = b"fn main() { println!(\"hello\"); }\n" * 30
    pairs.append(("mismatch", doc_ref, code_tgt))

    return pairs


def fmt_ratio(compressed_size, original_size):
    if original_size == 0:
        return "N/A"
    return f"{compressed_size / original_size * 100:.1f}%"


def main():
    samples = make_test_data()

    # --- Standard benchmark ---
    zstd_hdr = "    zstd" if HAS_ZSTD else ""
    print(f"{'data':<16} {'size':>6}  {'seedac':>8} {'(seed)':>8} {'(auto)':>8}  {'zlib':>8}  {'bzip2':>8}{zstd_hdr}")
    print("-" * (88 if HAS_ZSTD else 80))

    for label, data in samples:
        size = len(data)

        # Find best manual seed
        best_manual_size = float('inf')
        best_manual_id = 0
        for sid in [0, 1, 2, 3, 4, 5]:
            try:
                sc = load_seed(sid)
                cs, _ = bench_seedac(data, seed_id=sid)
                if cs < best_manual_size:
                    best_manual_size = cs
                    best_manual_id = sid
            except Exception:
                continue

        # Auto
        auto_size, auto_time, auto_name = bench_seedac_auto(data)

        # Others
        zlib_size, zlib_time = bench_zlib(data)
        bz2_size, bz2_time = bench_bzip2(data)
        zstd_size, zstd_time = bench_zstd(data)

        seed_name = SEED_NAMES.get(best_manual_id, str(best_manual_id))

        best_r = fmt_ratio(best_manual_size, size)
        auto_r = fmt_ratio(auto_size, size)
        zlib_r = fmt_ratio(zlib_size, size)
        bz2_r = fmt_ratio(bz2_size, size)

        zstd_col = f"  {fmt_ratio(zstd_size, size):>8}" if HAS_ZSTD else ""

        print(
            f"{label:<16} {size:>5}B  "
            f"{best_r:>7} {seed_name:>8}  {auto_r:>7} {auto_name:>8}  "
            f"{zlib_r:>7}  {bz2_r:>8}{zstd_col}"
        )

    # --- Recipe benchmark ---
    pairs = make_recipe_pairs()

    print()
    zstd_rhdr = f"  {'zstd':>8}" if HAS_ZSTD else ""
    print(f"{'recipe pair':<16} {'ref':>6} {'tgt':>6}  {'recipe':>8}  {'null':>8}  {'zlib':>8}{zstd_rhdr}")
    print("-" * (76 if HAS_ZSTD else 66))

    for label, ref, tgt in pairs:
        recipe_counts = make_recipe(ref)

        recipe_size, _ = bench_seedac_recipe(tgt, recipe_counts)
        null_size, _ = bench_seedac(tgt, seed_id=0)
        zlib_size, _ = bench_zlib(tgt)
        zstd_size, _ = bench_zstd(tgt)

        zstd_col = f"  {fmt_ratio(zstd_size, len(tgt)):>8}" if HAS_ZSTD else ""

        print(
            f"{label:<16} {len(ref):>5}B {len(tgt):>5}B  "
            f"{fmt_ratio(recipe_size, len(tgt)):>7}  "
            f"{fmt_ratio(null_size, len(tgt)):>7}  "
            f"{fmt_ratio(zlib_size, len(tgt)):>7}"
            f"{zstd_col}"
        )

    if not HAS_ZSTD:
        print("\n(install 'zstandard' for zstd comparison: pip install zstandard)")


if __name__ == "__main__":
    main()
