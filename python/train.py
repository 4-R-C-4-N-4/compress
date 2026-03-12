"""CLI tool to train .seedmodel files from corpus data.

Usage:
    python3 -m python.train --corpus <dir_or_file> --seed-id 1 --name "english" --order 4 --output english.seedmodel
"""

import argparse
import os
import sys

from .model import PPMModel, NUM_SYMBOLS
from .seed_format import write_seed


def collect_corpus_files(path):
    """Collect all files from a path (file or directory)."""
    if os.path.isfile(path):
        return [path]
    files = []
    for root, _, names in os.walk(path):
        for name in names:
            files.append(os.path.join(root, name))
    files.sort()
    return files


def train_model(corpus_data, max_order):
    """Feed corpus through a PPMModel and return its raw counts."""
    model = PPMModel(max_order=max_order)
    context = b""
    for byte in corpus_data:
        # Just update the model — we don't need to encode
        model._update_all(context, byte)
        context = (context + bytes([byte]))[-max_order:]
    return model


def extract_counts(model):
    """Extract count tables from a trained PPMModel."""
    counts = {}
    for order_model in model.orders:
        order_counts = {}
        for ctx, syms in order_model.counts.items():
            # Skip contexts with wrong length (e.g. b"" in order > 0
            # from the start of stream before context is long enough)
            if len(ctx) != order_model.order:
                continue
            if any(c > 0 for c in syms):
                order_counts[ctx] = list(syms)
        counts[order_model.order] = order_counts
    return counts


def prune_counts(counts, min_total):
    """Remove contexts with fewer than min_total total occurrences."""
    pruned = {}
    for order, order_counts in counts.items():
        pruned_order = {}
        for ctx, syms in order_counts.items():
            if sum(syms) >= min_total:
                pruned_order[ctx] = syms
        pruned[order] = pruned_order
    return pruned


def quantize_counts(counts, max_val=65535):
    """Scale counts to fit in u16 range, preserving ratios."""
    quantized = {}
    for order, order_counts in counts.items():
        q_order = {}
        for ctx, syms in order_counts.items():
            peak = max(syms)
            if peak == 0:
                continue
            if peak <= max_val:
                q_order[ctx] = syms
            else:
                scale = max_val / peak
                q_order[ctx] = [max(1, int(c * scale)) if c > 0 else 0 for c in syms]
        quantized[order] = q_order
    return quantized


def main():
    parser = argparse.ArgumentParser(
        description="Train a .seedmodel from corpus data",
    )
    parser.add_argument("--corpus", required=True, help="Corpus file or directory")
    parser.add_argument("--seed-id", type=int, required=True, help="Seed type ID (0-255)")
    parser.add_argument("--name", required=True, help="Seed name (e.g. 'english')")
    parser.add_argument("--order", type=int, default=4, help="Max PPM order (default: 4)")
    parser.add_argument("--output", required=True, help="Output .seedmodel path")
    parser.add_argument("--prune", type=int, default=3, help="Min total count per context (default: 3)")
    args = parser.parse_args()

    # Collect corpus
    files = collect_corpus_files(args.corpus)
    if not files:
        print(f"No files found at {args.corpus}", file=sys.stderr)
        sys.exit(1)

    corpus = bytearray()
    for f in files:
        with open(f, "rb") as fh:
            corpus.extend(fh.read())

    print(f"Corpus: {len(files)} file(s), {len(corpus)} bytes")

    # Train
    print(f"Training order-{args.order} model...")
    model = train_model(bytes(corpus), args.order)
    counts = extract_counts(model)

    # Stats before pruning
    total_contexts = sum(len(oc) for oc in counts.values())
    print(f"Raw contexts: {total_contexts}")

    # Prune
    counts = prune_counts(counts, args.prune)
    total_contexts = sum(len(oc) for oc in counts.values())
    print(f"After pruning (min_total={args.prune}): {total_contexts} contexts")

    # Quantize
    counts = quantize_counts(counts)

    # Write
    write_seed(args.output, args.seed_id, args.name, args.order, counts)
    file_size = os.path.getsize(args.output)
    print(f"Wrote {args.output} ({file_size} bytes)")


if __name__ == "__main__":
    main()
