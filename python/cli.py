"""CLI entry point for the seeded arithmetic coder."""

import argparse
import sys
import os

from .codec import encode, decode


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
    c.add_argument("--order", type=int, default=4, help="Max PPM order (default: 4)")
    c.add_argument("--seed", type=int, default=0, help="Seed ID (default: 0 = null/uniform, 255 = auto-detect)")

    # Decompress
    d = sub.add_parser("d", help="Decompress a .seed file")
    d.add_argument("input", help="Input .seed file path")
    d.add_argument("-o", "--output", help="Output file path (default: strip .seed extension)")

    args = parser.parse_args()

    if args.command == "c":
        with open(args.input, "rb") as f:
            data = f.read()

        compressed = encode(data, seed_id=args.seed, max_order=args.order)

        out_path = args.output or args.input + ".seed"
        with open(out_path, "wb") as f:
            f.write(compressed)

        ratio = len(compressed) / len(data) * 100 if data else 0
        print(f"{len(data)} -> {len(compressed)} bytes ({ratio:.1f}%)")

    elif args.command == "d":
        with open(args.input, "rb") as f:
            compressed = f.read()

        data = decode(compressed)

        if args.output:
            out_path = args.output
        elif args.input.endswith(".seed"):
            out_path = args.input[:-5]
        else:
            out_path = args.input + ".out"

        with open(out_path, "wb") as f:
            f.write(data)

        print(f"{len(compressed)} -> {len(data)} bytes")


if __name__ == "__main__":
    main()
