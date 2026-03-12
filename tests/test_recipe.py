"""Tests for recipe mode — single-file seed creation and use."""

import os
import subprocess
import sys
import random
import pytest

from python.codec import encode, decode
from python.train import train_model, extract_counts, prune_counts, quantize_counts
from python.seed_format import write_seed, read_seed

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-c", "from python.cli import main; main()", *args],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )


def make_recipe(data, max_order=4, prune=1):
    """Create seed_counts from data, like the recipe command does."""
    model = train_model(data, max_order)
    counts = extract_counts(model)
    counts = prune_counts(counts, prune)
    counts = quantize_counts(counts)
    return counts


class TestRecipeRoundtrip:
    def test_self_recipe(self):
        """Compress data with its own recipe — should roundtrip perfectly."""
        data = b"The quick brown fox jumps over the lazy dog. " * 10
        counts = make_recipe(data)

        compressed = encode(data, seed_id=0, seed_counts=counts)
        result = decode(compressed, seed_counts=counts)
        assert result == data

    def test_self_recipe_small(self):
        """Self-recipe on very small data still roundtrips."""
        data = b"hello"
        counts = make_recipe(data)

        compressed = encode(data, seed_id=0, seed_counts=counts)
        result = decode(compressed, seed_counts=counts)
        assert result == data

    def test_similar_file_benefits(self):
        """Compressing a similar file with a recipe should beat null seed."""
        original = b'{"name": "Alice", "age": 30, "city": "Portland"}\n' * 20
        similar = b'{"name": "Bob", "age": 25, "city": "Seattle"}\n' * 20

        counts = make_recipe(original)

        # With recipe
        compressed_recipe = encode(similar, seed_id=0, seed_counts=counts)
        # Without recipe
        compressed_null = encode(similar, seed_id=0)

        result = decode(compressed_recipe, seed_counts=counts)
        assert result == similar
        assert len(compressed_recipe) < len(compressed_null)

    def test_dissimilar_graceful_degradation(self):
        """Recipe from english text on random data — should still roundtrip."""
        english = b"This is a perfectly normal English sentence. " * 10
        counts = make_recipe(english)

        r = random.Random(42)
        random_data = bytes(r.randint(0, 255) for _ in range(200))

        compressed = encode(random_data, seed_id=0, seed_counts=counts)
        result = decode(compressed, seed_counts=counts)
        assert result == random_data

    def test_empty_data(self):
        """Recipe on empty data."""
        data = b""
        # Can't train a recipe from empty data, but compressing empty with a recipe works
        english = b"some training data here"
        counts = make_recipe(english)

        compressed = encode(data, seed_id=0, seed_counts=counts)
        result = decode(compressed, seed_counts=counts)
        assert result == data


class TestRecipeCLI:
    def test_create_recipe(self, tmp_path):
        """Create a recipe via CLI."""
        input_file = tmp_path / "input.txt"
        recipe_file = tmp_path / "input.txt.seedmodel"
        input_file.write_bytes(b"Hello world! " * 50)

        result = run_cli("recipe", str(input_file), "-o", str(recipe_file))
        assert result.returncode == 0, f"Recipe failed: {result.stderr}"
        assert recipe_file.exists()
        assert "Wrote" in result.stdout

    def test_full_workflow(self, tmp_path):
        """recipe → compress with recipe → decompress with recipe."""
        # Create source and recipe
        source = tmp_path / "source.txt"
        source.write_bytes(b'{"key": "value", "count": 42}\n' * 30)

        recipe_file = tmp_path / "source.seedmodel"
        result = run_cli("recipe", str(source), "-o", str(recipe_file))
        assert result.returncode == 0

        # Compress a similar file with the recipe
        target = tmp_path / "target.txt"
        target.write_bytes(b'{"key": "other", "count": 99}\n' * 30)

        compressed = tmp_path / "target.seed"
        result = run_cli("c", str(target), "--recipe", str(recipe_file), "-o", str(compressed))
        assert result.returncode == 0, f"Compress failed: {result.stderr}"

        # Decompress with the recipe
        output = tmp_path / "target.out"
        result = run_cli("d", str(compressed), "--recipe", str(recipe_file), "-o", str(output))
        assert result.returncode == 0, f"Decompress failed: {result.stderr}"

        assert output.read_bytes() == target.read_bytes()

    def test_seed_recipe_mutual_exclusion(self, tmp_path):
        """--seed and --recipe together should error."""
        input_file = tmp_path / "input.txt"
        input_file.write_bytes(b"test")

        result = run_cli("c", str(input_file), "--seed", "1", "--recipe", "foo.seedmodel")
        assert result.returncode != 0
        assert "cannot use" in result.stderr.lower() or "cannot use" in result.stdout.lower()

    def test_default_recipe_output(self, tmp_path):
        """Default recipe output is input.seedmodel."""
        input_file = tmp_path / "data.json"
        input_file.write_bytes(b'{"hello": "world"}')

        result = run_cli("recipe", str(input_file))
        assert result.returncode == 0
        assert (tmp_path / "data.json.seedmodel").exists()

    def test_recipe_uses_recipe_order(self, tmp_path):
        """Compress with --recipe should use recipe's order by default."""
        source = tmp_path / "source.txt"
        source.write_bytes(b"abcdef" * 100)

        recipe_file = tmp_path / "source.seedmodel"
        # Create recipe with order 2
        result = run_cli("recipe", str(source), "-o", str(recipe_file), "--order", "2")
        assert result.returncode == 0

        # Read recipe to confirm order
        _, _, max_order, _ = read_seed(str(recipe_file))
        assert max_order == 2

        # Compress — should use order 2 from recipe
        compressed = tmp_path / "source.seed"
        result = run_cli("c", str(source), "--recipe", str(recipe_file), "-o", str(compressed))
        assert result.returncode == 0

        # Decompress
        output = tmp_path / "source.out"
        result = run_cli("d", str(compressed), "--recipe", str(recipe_file), "-o", str(output))
        assert result.returncode == 0
        assert output.read_bytes() == source.read_bytes()
