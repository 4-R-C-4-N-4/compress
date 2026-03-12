"""CLI entry point for the seeded arithmetic coder."""

import argparse
import sys
import os

from .codec import encode, decode
from .seed_format import read_seed, write_seed
from .train import train_model, extract_counts, prune_counts, quantize_counts


def main():
    parser = argparse.ArgumentParser(
        prog="seedac",
        description="Seeded Arithmetic Coder — PPM-based compression",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Compress
    c = sub.add_parser("c", help="Compress a file")
    c.add_argument("input", help="Input file path")
    c.add_argument("-o", "--output", help="Output file path (default: input.seed)")
    c.add_argument("--order", type=int, default=None, help="Max PPM order (default: 4, or recipe's order)")
    c.add_argument("--seed", type=int, default=0, help="Seed ID (default: 0 = null/uniform, 255 = auto-detect)")
    c.add_argument("--recipe", help="Path to a .seedmodel recipe file")

    # Decompress
    d = sub.add_parser("d", help="Decompress a .seed file")
    d.add_argument("input", help="Input .seed file path")
    d.add_argument("-o", "--output", help="Output file path (default: strip .seed extension)")
    d.add_argument("--recipe", help="Path to a .seedmodel recipe file")

    # Recipe
    r = sub.add_parser("recipe", help="Create a recipe from a file")
    r.add_argument("input", help="Input file to create recipe from")
    r.add_argument("-o", "--output", help="Output .seedmodel path (default: input.seedmodel)")
    r.add_argument("--order", type=int, default=4, help="Max PPM order (default: 4)")
    r.add_argument("--prune", type=int, default=1, help="Min total count per context (default: 1)")

    args = parser.parse_args()

    if args.command == "c":
        if args.recipe and args.seed != 0:
            print("Error: cannot use --seed and --recipe together", file=sys.stderr)
            sys.exit(1)

        seed_counts = None
        max_order = args.order or 4

        if args.recipe:
            _, _, recipe_order, seed_counts = read_seed(args.recipe)
            if args.order is None:
                max_order = recipe_order

        with open(args.input, "rb") as f:
            data = f.read()

        compressed = encode(data, seed_id=args.seed, max_order=max_order, seed_counts=seed_counts)

        out_path = args.output or args.input + ".seed"
        with open(out_path, "wb") as f:
            f.write(compressed)

        ratio = len(compressed) / len(data) * 100 if data else 0
        print(f"{len(data)} -> {len(compressed)} bytes ({ratio:.1f}%)")

    elif args.command == "d":
        with open(args.input, "rb") as f:
            compressed = f.read()

        seed_counts = None
        if args.recipe:
            _, _, _, seed_counts = read_seed(args.recipe)

        data = decode(compressed, seed_counts=seed_counts)

        if args.output:
            out_path = args.output
        elif args.input.endswith(".seed"):
            out_path = args.input[:-5]
        else:
            out_path = args.input + ".out"

        with open(out_path, "wb") as f:
            f.write(data)

        print(f"{len(compressed)} -> {len(data)} bytes")

    elif args.command == "recipe":
        with open(args.input, "rb") as f:
            data = f.read()

        print(f"Training order-{args.order} model from {len(data)} bytes...")
        model = train_model(data, args.order)
        counts = extract_counts(model)

        total_contexts = sum(len(oc) for oc in counts.values())
        print(f"Raw contexts: {total_contexts}")

        counts = prune_counts(counts, args.prune)
        total_contexts = sum(len(oc) for oc in counts.values())
        print(f"After pruning (min_total={args.prune}): {total_contexts} contexts")

        counts = quantize_counts(counts)

        name = os.path.basename(args.input)
        out_path = args.output or args.input + ".seedmodel"
        write_seed(out_path, seed_id=0, name=name, max_order=args.order, counts=counts)

        file_size = os.path.getsize(out_path)
        print(f"Wrote {out_path} ({file_size} bytes)")


if __name__ == "__main__":
    main()
