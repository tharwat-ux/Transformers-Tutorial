# Transformers — Theory & Practical

Notes and hands-on exercises on Transformer architecture, split into two parts:
theory (written summaries) and practical (runnable notebooks using pretrained
Hugging Face models).

```
theory/
  01-tokenization-and-attention-fundamentals.md
  02-encoder-only-architecture.md
  03-decoder-only-architecture.md
  04-encoder-decoder-architecture.md
practical/
  transformer_playground.ipynb
  huggingface_tutorial.ipynb
```

---

## theory/

### 01 — Tokenization and Attention Fundamentals
The building blocks shared by every architecture that follows:
- **Tokenization**: why subwords beat word- or character-level vocabularies,
  and the three main algorithms — BPE (GPT-family), WordPiece (BERT-family),
  Unigram LM (T5/ALBERT-family).
- **Positional encoding**: absolute schemes (learned embeddings, sinusoidal)
  vs. relative schemes (Transformer-XL/T5/ALiBi score-bias, and RoPE).
- **Self-attention**: the Q/K/V mechanism, the `√d_k` scaling trick, the
  quadratic cost problem and how FlashAttention avoids it, multi-head
  attention, and padding masks.
- **Residual stream & normalization**: why residual connections matter,
  BatchNorm vs. LayerNorm vs. RMSNorm, and Post-LN vs. Pre-LN.
- **Feed-forward networks**: the attention:FFN parameter ratio, and SwiGLU.
- **Mixture of Experts (MoE)**: sparse FFNs and load balancing.

### 02 — Encoder-Only Architecture
The BERT-style family: one stack, full bidirectional (unmasked) attention,
built for *understanding* rather than generation. Covers Masked Language
Modeling (MLM) as the pretraining objective, and the `[CLS]` token convention.

### 03 — Decoder-Only Architecture
The GPT-style family: causal (masked) self-attention, why this architecture
ended up dominating general-purpose LLMs, and the machinery around serving
it efficiently — KV caching, Multi-Query/Grouped-Query Attention, context
window limits, decoding strategies (greedy, beam search, temperature,
top-k, top-p), and scaling laws (Kaplan vs. Chinchilla).

### 04 — Encoder-Decoder Architecture
The original Transformer (Vaswani et al., 2017): why translation-style tasks
need both a bidirectional encoder and a causal decoder, how cross-attention
bridges them, the three masks working together, and where this architecture
is still used today (T5, BART, Whisper). Closes with a side-by-side
comparison of BERT vs. GPT vs. T5.

---

## practical/

### transformer_playground.ipynb
Six hands-on projects applying the theory above with pretrained Hugging Face
models — no from-scratch implementation, just normal use of the library:

1. **Tokenizer comparison** — same sentence through GPT-2 (BPE), BERT
   (WordPiece), and T5 (Unigram) tokenizers.
2. **Attention visualization** — heatmaps comparing BERT's bidirectional
   attention to GPT-2's causal attention.
3. **MLM playground** — feeding masked sentences to BERT and inspecting
   top-k predictions.
4. **Decoding strategies** — greedy / temperature / top-k / top-p implemented
   manually from GPT-2's raw logits.
5. **KV cache benchmark** — timing generation with `use_cache=True` vs.
   `False` across different lengths.
   for each architecture (MLM, left-to-right continuation, span corruption).

### huggingface_tutorial.ipynb
A general introduction to the Hugging Face ecosystem itself (Hub,
`transformers`, `huggingface_hub`):
- `pipeline()` as the fastest way to try any task.
- What's happening underneath — tokenizer → model → logits → softmax.
- Batching sentences: padding, truncation, and the `attention_mask`.
- Saving/loading a model locally.
- A pointer to fine-tuning as the natural next step beyond this tutorial.
