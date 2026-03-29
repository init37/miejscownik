# Polish Locality Name Generator

> A character-level GRU language model trained on the official Polish registry of place names (SIMC), capable of generating new, plausible-sounding Polish locality names.

Sample output:

```
morek
kolonia lasowsk
łabędzin
wielka strona
ciechanka nowa
święte górki
```

---

## The Idea

Poland's official place names registry (SIMC) contains nearly **58,000 unique locality names** - villages, settlements, colonies - with a very distinctive phonetic and morphological style. The goal of this project was to train a model to learn that style well enough to invent new names that sound genuinely Polish.

---

## Dataset

- **Source:** `SIMC_Urzedowy_2026-01-18.csv` - the official Polish state registry of localities
- **Size:** 57,692 unique names after deduplication and lowercasing
- **Vocabulary:** 39 characters — the Polish alphabet plus special start (`^`) and end (`$`) tokens
- **Split:** 80% train / 10% validation / 10% test (minimum 1,000 examples per split)

---

## Architecture

The model is a **character-level n-gram language model** - at each step it takes the last `n` characters as context and predicts the next one.

```
[context: last 12 chars] → Embedding → GRU (2 layers) → Linear → next char
```

| Component | Details |
|---|---|
| `NGramDataset` | Sliding window over each word with SOS padding; produces `(context, next_char)` pairs |
| `Embedding` | Maps each character index to a 128-dim vector |
| `GRU` | 2 layers, hidden size 256, dropout 0.3 between layers |
| `Linear` | Projects final hidden state to logits over the 39-char vocabulary |

Generation is autoregressive: the model samples the next character from the output distribution (with temperature), appends it to the context window, and repeats until the EOS token is produced.

---

## Training

| Setting | Value |
|---|---|
| Context size | 12 characters |
| Batch size | 128 |
| Optimizer | AdamW (`lr=1e-3`, `weight_decay=1e-2`) |
| Scheduler | ReduceLROnPlateau (`patience=3`, `factor=0.5`) |
| Epochs | 25 |
| Hardware | NVIDIA T4 (Google Colab) |

Random baseline (uniform over 39 chars): loss ≈ 3.66

---

## Results

| Split | Loss | Top-1 Acc | Top-3 Acc |
|---|---|---|---|
| Validation | 1.5798 | 50.08% | 73.83% |
| Test | 1.5849 | 49.65% | 73.66% |

The model predicts the correct next character **50% of the time** and has the correct answer in its top 3 predictions **74% of the time** — compared to a 2.6% / 7.7% random baseline.

---

## Stack

- **Python 3**, **PyTorch**, **pandas**
- Trained on **Google Colab** (T4 GPU)
