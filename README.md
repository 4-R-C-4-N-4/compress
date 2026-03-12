# compress

Adventures in shrinkage

A seeded arithmetic coder that uses PPM (Prediction by Partial Matching) with pretrained probability tables to compress data. The key idea: if the encoder and decoder share a seed model that already knows the statistical patterns of a data type, compression starts strong from byte zero instead of slowly learning.

## Usage

```bash
# Compress (auto-detect best seed — the default)
python3 -m python.cli c input.txt -o input.txt.seed

# Compress with a specific seed by name
python3 -m python.cli c input.txt --seed english -o input.txt.seed

# Decompress
python3 -m python.cli d input.txt.seed -o input.txt

# List all available seeds
python3 -m python.cli seeds
```

Auto-detect is the default — it probes every `.seedmodel` in `seeds/` on the first 1 KB of input and picks whichever compresses smallest. Use `--seed <name>` to force a specific seed, or `--seed 0` for pure adaptive (no prior).

## Seed Models

Seed models are `.seedmodel` files in `seeds/`. They contain frozen PPM probability tables that serve as fallback priors — used when the adaptive model hasn't seen a context yet, then replaced entirely once real data arrives.

Any `.seedmodel` dropped into `seeds/` is automatically discovered by auto-detect, the `seeds` command, and `--seed <name>`. This means corpus-trained seeds, recipe files, and LLM-generated seeds are all first-class.

### Shipped seeds

| Name | Order | Size | Description |
|------|-------|------|-------------|
| `null` | 4 | 32 B | Empty/uniform — pure adaptive baseline |
| `english` | 4 | 901 KB | English prose |
| `code` | 3 | 1.5 MB | Mixed source code |
| `json` | 4 | 1.8 MB | JSON/structured data |
| `json_llm` | 4 | 55 KB | JSON (LLM-generated synthetic corpus) |
| `binary` | 3 | 3.0 MB | Binary formats (ELF, Wasm, PNG, protobuf) |
| `log` | 4 | 789 KB | Server/system logs |

### How seeds work

Seeds act as a **fallback prior**: when the adaptive model encounters a context it hasn't seen before, it uses the seed's probability distribution for that context. Once the adaptive model has observed any data for a context, the seed is replaced entirely. This keeps escape probabilities tight while still benefiting from pretrained knowledge on the first pass through unfamiliar contexts.

## Creating Seed Models

There are three ways to create a `.seedmodel`:

### 1. From a corpus

Train directly on representative files:

```bash
python3 -m python.cli recipe corpus.txt -o seeds/my_seed.seedmodel --order 4 --prune 3
```

### 2. From a reference file (recipe)

Use a single file as a compression model for similar files:

```bash
# Create a recipe from a reference file
python3 -m python.cli recipe reference.json -o reference.seedmodel

# Compress a similar file using the recipe
python3 -m python.cli c similar.json --recipe reference.seedmodel -o similar.seed

# Decompress (same recipe required)
python3 -m python.cli d similar.seed --recipe reference.seedmodel -o similar.json
```

The recipe captures byte-level statistical patterns of the reference file. When a similar file is compressed with this recipe, only the *differences* from the expected patterns need to be encoded.

**Good use cases for recipes:**
- Config files that differ by a few values (deploy configs, env files)
- Log batches from the same service
- API responses with the same schema but different data
- Versioned documents with incremental changes

Options:
- `--order N` — max PPM context depth (default: 4). Higher orders capture longer patterns but produce larger recipe files.
- `--prune N` — remove contexts seen fewer than N times (default: 1). Increase to shrink recipe size.

### 3. From an LLM (synthetic corpus)

Use any OpenAI-compatible API to generate synthetic training data for a seed. The LLM produces representative samples of a data type, which are then trained into a `.seedmodel` using the standard pipeline.

This is the fastest way to create a seed for a new data type — no real corpus needed.

```bash
# Set up API access (OpenRouter, OpenAI, local ollama, etc.)
export LLM_BASE_URL=https://openrouter.ai/api/v1
export LLM_API_KEY=<your openrouter key>
export LLM_MODEL=nvidia/nemotron-3-super-120b-a12b:free

# Generate a seed
python3 scripts/llm_seed.py \
    --type "JSON API responses with nested objects, arrays, and mixed types" \
    --seed-id 3 --name json_llm \
    --samples 20 --order 4 \
    --output seeds/json_llm.seedmodel
```

The `--type` flag is a natural-language description passed to the LLM. Be specific about the format you want to compress — the more precise the description, the better the seed.

**Examples for different data types:**

```bash
# Kubernetes manifests
python3 scripts/llm_seed.py \
    --type "Kubernetes YAML manifests (Deployments, Services, ConfigMaps, Ingress)" \
    --seed-id 0 --name k8s \
    --samples 30 --order 4 \
    --output seeds/k8s.seedmodel

# nginx access logs
python3 scripts/llm_seed.py \
    --type "nginx combined access log lines with varied HTTP methods, paths, status codes, and user agents" \
    --seed-id 0 --name nginx \
    --samples 25 --order 4 \
    --output seeds/nginx.seedmodel

# CSV tabular data
python3 scripts/llm_seed.py \
    --type "CSV files with headers and numeric/string columns (sales data, sensor readings, user events)" \
    --seed-id 0 --name csv \
    --samples 20 --order 4 \
    --output seeds/csv.seedmodel

# Markdown documentation
python3 scripts/llm_seed.py \
    --type "Markdown documentation with headings, code blocks, bullet lists, and links" \
    --seed-id 0 --name markdown \
    --samples 20 --order 4 \
    --output seeds/markdown.seedmodel

# TOML/INI configuration files
python3 scripts/llm_seed.py \
    --type "TOML configuration files with sections, key-value pairs, arrays, and inline tables" \
    --seed-id 0 --name toml \
    --samples 20 --order 4 \
    --output seeds/toml.seedmodel

# HTML pages
python3 scripts/llm_seed.py \
    --type "HTML pages with semantic tags, forms, tables, and CSS class attributes" \
    --seed-id 0 --name html \
    --samples 15 --order 4 \
    --output seeds/html.seedmodel

# SQL queries
python3 scripts/llm_seed.py \
    --type "SQL queries: SELECT with JOINs, WHERE, GROUP BY, INSERT, CREATE TABLE, ALTER TABLE" \
    --seed-id 0 --name sql \
    --samples 25 --order 4 \
    --output seeds/sql.seedmodel

# Protocol Buffers definitions
python3 scripts/llm_seed.py \
    --type "Protocol Buffers .proto definition files with messages, enums, services, and imports" \
    --seed-id 0 --name protobuf \
    --samples 20 --order 4 \
    --output seeds/protobuf.seedmodel

# Systemd journal / syslog
python3 scripts/llm_seed.py \
    --type "systemd journal entries with timestamps, severity levels, unit names, and structured fields" \
    --seed-id 0 --name syslog \
    --samples 25 --order 4 \
    --output seeds/syslog.seedmodel
```

**Using a local model with ollama:**

```bash
export LLM_BASE_URL=http://localhost:11434/v1
export LLM_MODEL=llama3
# No API key needed for local models
python3 scripts/llm_seed.py --type "..." --seed-id 0 --name my_seed \
    --samples 20 --output seeds/my_seed.seedmodel
```

LLM-generated seeds are typically much smaller (50-200 KB vs 1-3 MB for corpus-trained) because they're built from 20 synthetic samples rather than megabytes of real data. Despite this, they can outperform corpus-trained seeds — `json_llm` (55 KB) beats the 1.8 MB `json` seed on small JSON inputs because the LLM generates more diverse structural patterns.

Options:
- `--samples N` — number of samples to generate (default: 20). More samples = better coverage, slower generation.
- `--max-tokens N` — max tokens per sample (default: 2000).
- `--order N` — max PPM order (default: 4).
- `--prune N` — min total count per context (default: 3).
- `--save-corpus <path>` — save the raw generated corpus for inspection.

## Benchmark

Compression ratio (lower = better). Seedac with auto-detect dominates on small data where traditional compressors haven't built up enough context.

```
data               size    seedac   (seed)    (auto)              zlib     bzip2     zstd
-----------------------------------------------------------------------------------------
english 1K         944B    27.6%  english    27.6%  english    44.7%     51.4%     47.0%
english 5K        4720B    12.4%  english    12.4%  english     9.7%     16.0%      9.7%
code 1K            880B    31.4%     code    31.4%     code    41.8%     46.1%     45.3%
code 5K           5120B    12.6%     code    12.6%     code     8.1%     12.2%      8.0%
json 500B          537B    36.7%     json    29.8% json_llm    42.3%     50.7%     44.1%
json 5K           5120B     9.5%     json     8.8% json_llm     5.2%      7.9%      4.7%
log 500B           512B    45.3%     code    45.3%     code    57.8%     69.5%     59.6%
log 5K            5120B     9.0%     code     9.0%     code     7.0%     11.2%      6.4%
random 1K         1024B   114.1%     null   114.1%     null   101.1%    129.8%    101.0%
zeros 1K          1024B     2.3%     null     2.3%     null     1.7%      4.1%      1.9%
```

Recipe benchmark (compress a target file using a model trained on a similar reference file):

```
recipe pair         ref    tgt    recipe      null      zlib      zstd
----------------------------------------------------------------------------
config            2080B  2060B     6.1%     8.3%     5.4%      5.0%
log batch         2510B  2490B     7.3%     9.6%     6.3%      5.8%
API resp          3659B  3833B    13.4%    14.3%    15.0%      8.9%
doc edit          1800B  1800B     3.5%     5.2%     3.8%      3.6%
mismatch          1800B   990B     8.2%     7.5%     5.2%      5.3%
```

Run `python3 benchmark.py` to reproduce (requires `zstandard` for the zstd column).

## Training Data Sources

### english

Trained on [Project Gutenberg](https://www.gutenberg.org/) plain-text novels:

- *Pride and Prejudice* by Jane Austen ([gutenberg.org/ebooks/1342](https://www.gutenberg.org/ebooks/1342))
- *Moby Dick* by Herman Melville ([gutenberg.org/ebooks/2701](https://www.gutenberg.org/ebooks/2701))

~2 MB total corpus. Captures English letter frequencies, common words, punctuation patterns, and prose structure.

### code

Trained on samples from [The Stack (deduplicated)](https://huggingface.co/datasets/bigcode/the-stack-dedup), a curated dataset of permissively-licensed source code hosted on Hugging Face:

- Python, JavaScript, C, Rust, Go — 200 files each
- ~11 MB total corpus, trained at order 3 with prune threshold 10

The Stack is described in: Kocetkov et al., "The Stack: 3 TB of permissively licensed source code" (2022). [arxiv.org/abs/2211.15533](https://arxiv.org/abs/2211.15533)

### json

Trained on a mix of public JSON sources:

- [Swagger/OpenAPI](https://swagger.io/specification/) specification files
- [Open Data](https://catalog.data.gov/) catalog metadata
- Miscellaneous API response samples (JSONPlaceholder-style)

~6 MB total corpus. Captures JSON structural patterns: braces, brackets, quoted keys, colons, commas, and common value types.

### json_llm

LLM-generated synthetic corpus using `nvidia/nemotron-3-super-120b-a12b:free` via OpenRouter. 20 samples of "JSON API responses with nested objects, arrays, and mixed types." Despite being 55 KB (vs 1.8 MB for corpus-trained `json`), it beats the corpus seed on small JSON inputs thanks to more diverse structural patterns.

### binary

Trained on a mix of binary format samples:

- ELF executables (Linux ARM64, x86)
- Mach-O binaries (macOS, iOS)
- [WebAssembly](https://webassembly.org/) `.wasm` modules from [mdn/webassembly-examples](https://github.com/mdn/webassembly-examples)
- [Protocol Buffers](https://protobuf.dev/) serialized messages
- [MNIST](http://yann.lecun.com/exdb/mnist/) PNG images (small 28x28 samples)

~3 MB total corpus, trained at order 3. Captures binary headers, LEB128 varints, PNG chunk structure, and ELF/Mach-O section patterns.

### log

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

71 tests covering arithmetic coder, PPM model, seed format, codec roundtrips, training pipeline, recipe mode, and CLI end-to-end.

## Architecture

See [spec.md](spec.md) for the full design specification covering PPM escape mechanics, seed integration, and the file formats.

## License

MIT
