# compress

Adventures in shrinkage

A seeded arithmetic coder that uses PPM (Prediction by Partial Matching) with pretrained probability tables to compress data. The key idea: if the encoder and decoder share a seed model that already knows the statistical patterns of a data type, compression starts strong from byte zero instead of slowly learning.

## Usage

```bash
# Compress
python3 -m python c input.txt --seed 1 -o input.txt.seed

# Decompress
python3 -m python d input.txt.seed -o input.txt
```

Seed ID 0 (null/uniform) is the default — pure adaptive, no prior knowledge. Use `--seed N` to select a pretrained seed for better compression on matching data.

## Seed Models

Seed models are binary `.seedmodel` files stored in `seeds/`. They contain frozen PPM probability tables serialized in a cross-language format (designed for both Python and Rust).

| ID | Name | Order | Seed Size | Description |
|----|------|-------|-----------|-------------|
| 0 | `null` | 4 | 32 B | Empty/uniform — pure adaptive baseline |
| 1 | `english` | 4 | 922 KB | English prose |
| 2 | `code` | 3 | 1.5 MB | Mixed source code |
| 5 | `log` | 4 | 808 KB | Server/system logs |

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

## .seedmodel Binary Format

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

### Seed 5 — Log

Trained on [Loghub](https://github.com/logpai/loghub), a collection of system log datasets for log analytics research:

- Apache, Linux, OpenSSH, HDFS, Hadoop, Spark, Zookeeper logs
- ~4.3 MB total corpus

Loghub is described in: He et al., "Loghub: A Large Collection of System Log Datasets towards Automated Log Analytics" (2020). [arxiv.org/abs/2008.06319](https://arxiv.org/abs/2008.06319)

## Architecture

See [spec.md](spec.md) for the full design specification covering PPM escape mechanics, context mixing, seed integration, and the `.seed` compressed file format.

## License

MIT
