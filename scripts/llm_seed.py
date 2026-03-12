#!/usr/bin/env python3
"""Generate a .seedmodel from LLM-generated synthetic training data.

Uses any OpenAI-compatible API (OpenRouter, OpenAI, local ollama, etc.)
to generate representative samples of a data type, then trains a seed
from the synthetic corpus using the existing pipeline.

Usage:
    # With OpenRouter
    export LLM_BASE_URL=https://openrouter.ai/api/v1
    export LLM_API_KEY=sk-or-...
    export LLM_MODEL=anthropic/claude-haiku-3

    # With local ollama
    export LLM_BASE_URL=http://localhost:11434/v1
    export LLM_MODEL=llama3

    python3 scripts/llm_seed.py \
        --type "JSON API responses" \
        --seed-id 3 \
        --name json \
        --samples 20 \
        --order 4 \
        --output seeds/json_llm.seedmodel

Environment variables:
    LLM_BASE_URL   API base URL (required)
    LLM_API_KEY    API key (optional for local models)
    LLM_MODEL      Model name (required)
"""

import argparse
import os
import sys
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from python.train import train_model, extract_counts, prune_counts, quantize_counts
from python.seed_format import write_seed


def get_client():
    """Create an OpenAI-compatible client from environment variables."""
    try:
        from openai import OpenAI
    except ImportError:
        print("Error: install openai package: pip install openai", file=sys.stderr)
        sys.exit(1)

    base_url = os.environ.get("LLM_BASE_URL")
    api_key = os.environ.get("LLM_API_KEY", "no-key")
    model = os.environ.get("LLM_MODEL")

    if not base_url:
        print("Error: set LLM_BASE_URL environment variable", file=sys.stderr)
        sys.exit(1)
    if not model:
        print("Error: set LLM_MODEL environment variable", file=sys.stderr)
        sys.exit(1)

    return OpenAI(base_url=base_url, api_key=api_key), model


SYSTEM_PROMPT = """You are a data generator. You produce realistic, diverse samples of the requested data type. Output ONLY the raw data — no markdown fences, no explanations, no labels. Each sample should be different from the others in content but consistent in format. Vary the content significantly between samples."""


def generate_samples(client, model, data_type, num_samples, max_tokens=2000):
    """Generate synthetic samples of a data type via LLM."""
    corpus = bytearray()

    for i in range(num_samples):
        prompt = (
            f"Generate a realistic example of: {data_type}\n\n"
            f"This is sample {i + 1} of {num_samples}. "
            f"Make it different from other samples — vary names, values, "
            f"structure, and content. Output ONLY the raw data."
        )

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=1.0,
            )
            text = response.choices[0].message.content
            if text:
                corpus.extend(text.encode("utf-8"))
                print(f"  sample {i + 1}/{num_samples}: {len(text)} chars")
        except Exception as e:
            print(f"  sample {i + 1}/{num_samples}: error: {e}", file=sys.stderr)

    return bytes(corpus)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a .seedmodel from LLM-generated training data",
    )
    parser.add_argument("--type", required=True,
                        help="Data type description (e.g. 'JSON API responses', 'Python source code', 'nginx access logs')")
    parser.add_argument("--seed-id", type=int, required=True, help="Seed type ID (0-255)")
    parser.add_argument("--name", required=True, help="Seed name")
    parser.add_argument("--samples", type=int, default=20, help="Number of samples to generate (default: 20)")
    parser.add_argument("--max-tokens", type=int, default=2000, help="Max tokens per sample (default: 2000)")
    parser.add_argument("--order", type=int, default=4, help="Max PPM order (default: 4)")
    parser.add_argument("--prune", type=int, default=3, help="Min total count per context (default: 3)")
    parser.add_argument("--output", required=True, help="Output .seedmodel path")
    parser.add_argument("--save-corpus", help="Also save the raw corpus to this path")
    args = parser.parse_args()

    client, model = get_client()

    print(f"Generating {args.samples} samples of '{args.type}' using {model}...")
    corpus = generate_samples(client, model, args.type, args.samples, args.max_tokens)

    if not corpus:
        print("Error: no data generated", file=sys.stderr)
        sys.exit(1)

    print(f"\nCorpus: {len(corpus)} bytes")

    if args.save_corpus:
        with open(args.save_corpus, "wb") as f:
            f.write(corpus)
        print(f"Saved corpus to {args.save_corpus}")

    print(f"Training order-{args.order} model...")
    trained = train_model(corpus, args.order)
    counts = extract_counts(trained)

    total_contexts = sum(len(oc) for oc in counts.values())
    print(f"Raw contexts: {total_contexts}")

    counts = prune_counts(counts, args.prune)
    total_contexts = sum(len(oc) for oc in counts.values())
    print(f"After pruning (min_total={args.prune}): {total_contexts} contexts")

    counts = quantize_counts(counts)

    write_seed(args.output, args.seed_id, args.name, args.order, counts)
    file_size = os.path.getsize(args.output)
    print(f"Wrote {args.output} ({file_size} bytes)")


if __name__ == "__main__":
    main()
