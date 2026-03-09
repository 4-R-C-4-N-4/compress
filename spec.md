# Seeded Arithmetic Coder — Design Specification

## Core Idea

The encoder and decoder share a **seed model** — a pretrained probability
distribution over byte sequences for a specific data type. The seed is
identified by a short ID in the file header. Both sides deterministically
reconstruct the identical model from the seed ID alone.

The arithmetic coder then encodes the **surprise** — only the bits where
the actual data deviates from what the seed model predicted. If the seed
is a good match, this surprise is tiny. If it's a poor match, the coder
gracefully degrades to adaptive-only (no worse than the current system).

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  ENCODER                         │
│                                                  │
│  input bytes ──► PPM model (escape + mixing) ──► arithmetic coder ──► .seed file
│                     ▲                                                    │
│                     │                                                    │
│              seed prior (pretrained)                              header: seed_id,
│              + adaptive updates                                  byte_length, fp
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│                  DECODER                         │
│                                                  │
│  .seed file ──► arithmetic decoder ──► PPM model ──► output bytes
│       │                                   ▲
│       │                                   │
│  header: seed_id ──► same seed prior + adaptive updates
└─────────────────────────────────────────────────┘
```

## Seed Models

### What makes a good seed?

A seed is a **frozen probability table**: for each context (byte history),
a distribution over the next byte. The better this distribution matches
the actual data, the fewer bits the arithmetic coder needs.

### Proposed seed types:

| ID  | Name       | Description                                | Fallback? |
|-----|------------|--------------------------------------------|-----------|
| 0   | `null`     | Uniform distribution. Pure adaptive.       | YES       |
| 1   | `english`  | English prose/text. Trained on literature. | no        |
| 2   | `code`     | Source code (mixed languages).             | no        |
| 3   | `json`     | JSON/structured data.                      | no        |
| 4   | `binary`   | Generic binary (executables, images).      | no        |
| 5   | `log`      | Server logs, timestamps, IPs.              | no        |
| 255 | `auto`     | Try all seeds, pick best (2-pass).         | YES       |

### Seed 0 (`null`) as universal fallback

Seed 0 is the uniform prior — every byte equally likely in every context.
This is identical to the current adaptive-only coder. It's the guaranteed
safe fallback: never worse than starting from scratch.

If a specialized seed is a BAD match for the data, it actively hurts
(the model wastes bits on wrong predictions). The encoder should detect
this and fall back to seed 0.

### Auto-detection (seed 255)

Encode the first ~1KB with each seed model. Pick whichever produces the
fewest bits. Store that seed ID. Cost: a few ms of extra encoding time.
Benefit: always picks the best seed without user intervention.

## PPM Model with Escape + Mixing

### Escape mechanism (PPMD-style)

For each context of length K:
1. Look up counts for all symbols seen in this context.
2. If the target symbol HAS been seen: encode it using its frequency.
3. If NOT seen: encode an ESCAPE symbol, then try context of length K-1.
4. Repeat down to order 0. If still not seen, use order -1 (uniform 1/256).

The escape probability is calculated as:
  P(escape) = d / (total + d)
where d = number of distinct symbols seen in this context.
(This is the PPMD "Method D" exclusion technique.)

### Seed integration

The seed prior provides the INITIAL counts for each context.
As data is processed, adaptive counts are added ON TOP of the seed counts.
This means:
- At byte 0: predictions are purely from the seed.
- After N bytes: predictions blend seed + observed data.
- For long files: observed data dominates (seed becomes irrelevant).

Implementation:
```python
class SeededModel:
    def __init__(self, seed_counts, order):
        self.order = order
        self.seed = seed_counts          # frozen prior
        self.adaptive = defaultdict(...)  # learned from data

    def get_counts(self, context):
        ctx = context[-self.order:]
        seed_c = self.seed.get(ctx, uniform)
        adapt_c = self.adaptive.get(ctx, zeros)
        return seed_c + adapt_c  # blended
```

## Seed Training

Seeds are built offline by processing a large corpus:

1. Feed corpus through the adaptive model.
2. After processing, freeze the count tables.
3. Prune: remove contexts seen fewer than T times.
4. Quantize: reduce counts to 4-8 bit precision (saves memory).
5. Serialize as a compact binary blob, embedded in the codec.

The blob size per seed: ~50KB-500KB depending on order and corpus size.
This is compiled INTO the encoder/decoder — not transmitted per file.

## File Format (.seed)

```
SEED                          ← 4-byte magic
seed_id: u8                   ← which seed model (0-255)
order: u8                     ← max PPM order
byte_length: u32              ← original file size
fingerprint: [u8; N]          ← base-188 integrity check
---                           ← separator
<arithmetic coded bitstream>  ← the actual compressed data
```
### Model transfer ("recipe mode")
- After encoding file A, export trained model as a custom seed
- Encode file B using A's model: if similar, B is tiny
- This is the "send a seed, regenerate the data" vision
