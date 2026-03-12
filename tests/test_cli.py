"""End-to-end CLI tests (python/cli.py)."""

import os
import subprocess
import sys
import tempfile
import pytest


PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))


def run_cli(*args):
    """Run the CLI as a subprocess and return the result."""
    return subprocess.run(
        [sys.executable, "-c", "from python.cli import main; main()", *args],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )


class TestCompressDecompress:
    def test_roundtrip(self, tmp_path):
        """Compress a temp file via CLI, decompress, compare."""
        input_file = tmp_path / "input.txt"
        compressed_file = tmp_path / "input.txt.seed"
        output_file = tmp_path / "output.txt"

        original = b"Hello, this is a test of the CLI interface!"
        input_file.write_bytes(original)

        # Compress
        result = run_cli("c", str(input_file), "-o", str(compressed_file))
        assert result.returncode == 0, f"Compress failed: {result.stderr}"
        assert compressed_file.exists()

        # Decompress
        result = run_cli("d", str(compressed_file), "-o", str(output_file))
        assert result.returncode == 0, f"Decompress failed: {result.stderr}"
        assert output_file.exists()

        assert output_file.read_bytes() == original

    def test_roundtrip_with_seed(self, tmp_path):
        """Compress with a seed, decompress, compare."""
        input_file = tmp_path / "input.txt"
        compressed_file = tmp_path / "input.txt.seed"
        output_file = tmp_path / "output.txt"

        original = b"The quick brown fox jumps over the lazy dog."
        input_file.write_bytes(original)

        # Compress with english seed
        result = run_cli("c", str(input_file), "-o", str(compressed_file), "--seed", "1")
        assert result.returncode == 0, f"Compress failed: {result.stderr}"

        # Decompress
        result = run_cli("d", str(compressed_file), "-o", str(output_file))
        assert result.returncode == 0, f"Decompress failed: {result.stderr}"

        assert output_file.read_bytes() == original

    def test_compress_output_message(self, tmp_path):
        """Verify compress prints ratio info."""
        input_file = tmp_path / "input.txt"
        input_file.write_bytes(b"test data")

        result = run_cli("c", str(input_file), "-o", str(tmp_path / "out.seed"))
        assert result.returncode == 0
        assert "->" in result.stdout
        assert "bytes" in result.stdout

    def test_decompress_output_message(self, tmp_path):
        """Verify decompress prints size info."""
        input_file = tmp_path / "input.txt"
        compressed_file = tmp_path / "input.seed"
        output_file = tmp_path / "output.txt"

        input_file.write_bytes(b"test data")
        run_cli("c", str(input_file), "-o", str(compressed_file))

        result = run_cli("d", str(compressed_file), "-o", str(output_file))
        assert result.returncode == 0
        assert "->" in result.stdout
        assert "bytes" in result.stdout

    def test_roundtrip_auto_seed(self, tmp_path):
        """Compress with --seed auto, decompress, compare."""
        input_file = tmp_path / "input.txt"
        compressed_file = tmp_path / "input.txt.seed"
        output_file = tmp_path / "output.txt"

        original = b"The quick brown fox jumps over the lazy dog. " * 10
        input_file.write_bytes(original)

        result = run_cli("c", str(input_file), "-o", str(compressed_file), "--seed", "auto")
        assert result.returncode == 0, f"Compress failed: {result.stderr}"

        result = run_cli("d", str(compressed_file), "-o", str(output_file))
        assert result.returncode == 0, f"Decompress failed: {result.stderr}"

        assert output_file.read_bytes() == original

    def test_missing_input(self):
        """Missing input file should fail."""
        result = run_cli("c", "/nonexistent/file.txt")
        assert result.returncode != 0

    def test_default_output_name(self, tmp_path):
        """Default output is input.seed for compress."""
        input_file = tmp_path / "data.txt"
        input_file.write_bytes(b"hello")

        result = run_cli("c", str(input_file))
        assert result.returncode == 0
        assert (tmp_path / "data.txt.seed").exists()
