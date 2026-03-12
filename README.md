# compress

Adventures in shrinkage

A seeded arithmetic coder that uses PPM (Prediction by Partial Matching) with pretrained probability tables to compress data. The key idea: if the encoder and decoder share a seed model that already knows the statistical patterns of a data type, compression starts strong from byte zero instead of slowly learning.

## Usage

```bash
# Compress (auto-detect best seed)
python3 -m python.cli c input.txt --seed 255 -o input.txt.seed

# Compress with a specific seed
python3 -m python.cli c input.txt --seed 1 -o input.txt.seed

# Decompress
python3 -m python.cli d input.txt.seed -o input.txt
```

Seed ID 0 (null/uniform) is the default — pure adaptive, no prior knowledge. Use `--seed 255` to auto-detect the best seed, or `--seed N` to select a specific one.

## Recipe Mode

Use any file as a compression model for similar files. If two files share statistical structure, the second compresses to almost nothing.

```bash
# Create a recipe from a reference file
python3 -m python.cli recipe reference.json -o reference.seedmodel

# Compress a similar file using the recipe
python3 -m python.cli c similar.json --recipe reference.seedmodel -o similar.seed

# Decompress (same recipe required)
python3 -m python.cli d similar.seed --recipe reference.seedmodel -o similar.json
```

The recipe captures the byte-level statistical patterns of the reference file — character frequencies, common substrings, structural regularities. When a similar file is compressed with this recipe, only the *differences* from the expected patterns need to be encoded.

**When to use recipes:**
- Config files that differ by a few values (deploy configs, env files)
- Log batches from the same service
- API responses with the same schema but different data
- Versioned documents with incremental changes

**How it works:** A recipe is a standard `.seedmodel` file created from a single file instead of a large corpus. It stores the same PPM probability tables as a trained seed. The `--recipe` flag overrides the built-in seed lookup, passing the recipe's counts directly to the encoder/decoder.

Options:
- `--order N` — max PPM context depth (default: 4). Higher orders capture longer patterns but produce larger recipe files.
- `--prune N` — remove contexts seen fewer than N times (default: 1). Increase to shrink recipe size.

## Benchmark

Compression ratio (lower = better). Seedac dominates on small data where traditional compressors haven't built up enough context.

```
data               size    seedac   (seed)      zlib     bzip2     zstd
----------------------------------------------------------------------
english 1K         944B    27.6%  english    44.7%     51.4%     47.0%
english 5K        4720B    12.4%  english     9.7%     16.0%      9.7%
code 1K            880B    31.4%     code    41.8%     46.1%     45.3%
code 5K           5120B    12.6%     code     8.1%     12.2%      8.0%
json 500B          537B    36.7%     json    42.3%     50.7%     44.1%
json 5K           5120B     9.5%     json     5.2%      7.9%      4.7%
log 500B           512B    45.3%     code    57.8%     69.5%     59.6%
log 5K            5120B     9.0%     code     7.0%     11.2%      6.4%
random 1K         1024B   114.1%     null   101.1%    129.8%    101.0%
zeros 1K          1024B     2.3%     null     1.7%      4.1%      1.9%
```

Run `python3 benchmark.py` to reproduce (requires `zstandard` for the zstd column).

## Seed Models

Seed models are binary `.seedmodel` files in `seeds/`. They contain frozen PPM probability tables that serve as fallback priors — used when the adaptive model hasn't seen a context yet, then replaced once real data arrives.

| ID | Name | Order | Seed Size | Description |
|----|------|-------|-----------|-------------|
| 0 | `null` | 4 | 32 B | Empty/uniform — pure adaptive baseline |
| 1 | `english` | 4 | 922 KB | English prose |
| 2 | `code` | 3 | 1.5 MB | Mixed source code |
| 3 | `json` | 4 | 1.8 MB | JSON/structured data |
| 4 | `binary` | 3 | 3.1 MB | Binary formats (ELF, Wasm, PNG, protobuf) |
| 5 | `log` | 4 | 808 KB | Server/system logs |
| 255 | `auto` | — | — | Try all seeds, pick best (2-pass) |

### How seeds work

Seeds act as a **fallback prior**: when the adaptive model encounters a context it hasn't seen before, it uses the seed's probability distribution for that context. Once the adaptive model has observed any data for a context, the seed is replaced entirely. This keeps escape probabilities tight while still benefiting from pretrained knowledge on the first pass through unfamiliar contexts.

### Training a new seed

```bash
python3 -m python.train \
    --corpus <dir_or_file> \
    --seed-id 3 \
    --name json \
    --order 4 \
    --prune 5 \
    --output seeds/json.seedmodel
```

`--prune N` removes contexts with fewer than N total occurrences. `--order` controls max context depth (higher = better compression but larger seed files).

## Training Data Sources

### Seed 1 — English

Trained on [Project Gutenberg](https://www.gutenberg.org/) plain-text novels:

- *Pride and Prejudice* by Jane Austen ([gutenberg.org/ebooks/1342](https://www.gutenberg.org/ebooks/1342))
- *Moby Dick* by Herman Melville ([gutenberg.org/ebooks/2701](https://www.gutenberg.org/ebooks/2701))

~2 MB total corpus. Captures English letter frequencies, common words, punctuation patterns, and prose structure.

### Seed 2 — Code

Trained on samples from [The Stack (deduplicated)](https://huggingface.co/datasets/bigcode/the-stack-dedup), a curated dataset of permissively-licensed source code hosted on Hugging Face:

- Python, JavaScript, C, Rust, Go — 200 files each
- ~11 MB total corpus, trained at order 3 with prune threshold 10

The Stack is described in: Kocetkov et al., "The Stack: 3 TB of permissively licensed source code" (2022). [arxiv.org/abs/2211.15533](https://arxiv.org/abs/2211.15533)

### Seed 3 — JSON

Trained on a mix of public JSON sources:

- [Swagger/OpenAPI](https://swagger.io/specification/) specification files
- [Open Data](https://catalog.data.gov/) catalog metadata
- Miscellaneous API response samples (JSONPlaceholder-style)

~6 MB total corpus. Captures JSON structural patterns: braces, brackets, quoted keys, colons, commas, and common value types.

### Seed 4 — Binary

Trained on a mix of binary format samples:

- ELF executables (Linux ARM64, x86)
- Mach-O binaries (macOS, iOS)
- [WebAssembly](https://webassembly.org/) `.wasm` modules from [mdn/webassembly-examples](https://github.com/mdn/webassembly-examples)
- [Protocol Buffers](https://protobuf.dev/) serialized messages
- [MNIST](http://yann.lecun.com/exdb/mnist/) PNG images (small 28x28 samples)

~3 MB total corpus, trained at order 3. Captures binary headers, LEB128 varints, PNG chunk structure, and ELF/Mach-O section patterns.

### Seed 5 — Log

Trained on [Loghub](https://github.com/logpai/loghub), a collection of system log datasets for log analytics research:

- Apache, Linux, OpenSSH, HDFS, Hadoop, Spark, Zookeeper logs
- ~4.3 MB total corpus

Loghub is described in: He et al., "Loghub: A Large Collection of System Log Datasets towards Automated Log Analytics" (2020). [arxiv.org/abs/2008.06319](https://arxiv.org/abs/2008.06319)

## File Formats

### .seed (compressed file)

```
SEED                          <- 4-byte magic
seed_id: u8                   <- which seed model (0-255)
order: u8                     <- max PPM order
byte_length: u32 (LE)        <- original file size
fingerprint: [u8; 8]          <- BLAKE2b-64 integrity check
---                           <- 3-byte separator
<arithmetic coded bitstream>
```

### .seedmodel (seed model)

```
SDML                          <- 4-byte magic
version: u8                   <- format version (1)
seed_id: u8                   <- seed type ID (0-255)
max_order: u8                 <- max PPM order stored
name_len: u8                  <- length of name string
name: [u8; name_len]          <- UTF-8 name

For each order 0..max_order:
  num_contexts: u32 (LE)
  For each context:
    context_bytes: [u8; order]
    num_entries: u16 (LE)      <- non-zero symbol count (sparse)
    For each entry:
      symbol: u8
      count: u16 (LE)          <- quantized frequency
```

All integers are little-endian. Sparse encoding stores only non-zero counts.

## Testing

```bash
pip install pytest
pytest tests/ -v
```

67 tests covering arithmetic coder, PPM model, seed format, codec roundtrips, training pipeline, recipe mode, and CLI end-to-end.

## Architecture

See [spec.md](spec.md) for the full design specification covering PPM escape mechanics, seed integration, and the file formats.

## License

MIT
