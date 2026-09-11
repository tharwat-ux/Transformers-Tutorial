# The Encoder-Decoder Transformer (Vaswani et al., 2017)

## 1. Why encoder-decoder in the first place?

The original Transformer was designed for **machine translation** (WMT English↔German /
English↔French). The task structure itself demands two different jobs:

1. **Understand the entire source sentence** (all of it is available up front, order
   doesn't need to be causal — you can look at the whole German sentence while
   translating word 3).
2. **Generate the target sentence one token at a time**, where each new token can only
   depend on tokens already generated (you can't peek at the English translation you
   haven't produced yet).

These are fundamentally different constraints — full bidirectional access vs strict
left-to-right generation — so the architecture splits into two stacks that specialize in
each job, connected by a bridge (cross-attention) that lets the generation process
consult the fully-understood source at every step.

## 2. High-level architecture

![Encoder-Decoder](images\Encoder-Decoder.png)

If you want the classic side-by-side picture from the paper itself (two stacked towers
with the arrow connecting them in the middle), a rendered version of Figure 1 from
"Attention is All You Need" is a good visual reference:

![Encoder-Decoder Architecture — Attention is All You Need, Figure 1](encoder-decoder-architecture.png)

## 3. The Encoder block, in detail

This is exactly the encoder stack from `02-encoder-only-architecture.md`, unchanged: `N`
identical blocks, each doing two things to every token's representation:

1. **Bidirectional self-attention** — the same no-mask attention you already saw in the
   encoder-only file: every token attends to every other token in the input, including
   tokens to its right, since the entire source sentence is known in advance.
2. **Position-wise Feed-Forward** (see fundamentals file §6), applied identically and
   independently to each token's vector.

Each sub-layer is wrapped with a residual connection and normalization (Post-LN in the
original paper — see fundamentals file §5.5 for why later models moved this). Nothing
about this stack changes just because a decoder is now attached on the other side — it
produces the exact same kind of fully-contextualized, same-length output sequence
described in the encoder-only file.

The **final encoder output** is a sequence of vectors, one per input token, each one now
"contextualized" — infused with information from the whole sentence — but the *sequence
length stays the same* as the input (encoding doesn't compress the sequence down to a
single vector, contrary to a common misconception from older RNN-encoder architectures).

## 4. The Decoder block, in detail

Each decoder block does **three** things (one more sub-layer than the encoder):

### 4.1 Masked (causal) self-attention — same mechanism as the decoder-only file

This is exactly the causal masking you already met in `03-decoder-only-architecture.md`
§2: position `i` is only allowed to attend to positions `≤ i`, enforced by setting
disallowed entries of `QKᵀ` to `-∞` before the softmax. Nothing changes here — the
decoder half of this full architecture uses the identical masked self-attention
sub-layer, for the identical reason (the target sentence is fed in all at once during
training, so the mask is what stops the model from "cheating" by looking at the answer
it's supposed to predict).

### 4.2 Cross-attention — the actual encoder-decoder bridge

This is the sub-layer that makes it an "encoder-**decoder**" model rather than two
independent stacks.

```
Cross-Attention(Q, K, V), inside decoder layer l:
   Q       comes from decoder layer l's own (masked self-attention) output,
           projected by that layer's own W_Q^l
   K, V    are produced by projecting the ENCODER's final hidden states through
           layer l's OWN K/V projection matrices, W_K^l and W_V^l

   Attention(Q_decoder, K_encoder, V_encoder) = softmax(Q Kᵀ / √d_k) V
```

**What's actually shared across decoder layers, and what isn't**: every decoder layer's
cross-attention reads from the *same* underlying data — the encoder's final hidden
states, computed once. But each layer has its **own** `W_K^l` and `W_V^l` projection
matrices, so each layer projects those shared hidden states into a **different** set of
Key and Value vectors. It's the encoder's output that's shared and reused across layers
— not the K and V tensors themselves, which differ from layer to layer because the
projection weights differ. This distinction matters because it's exactly analogous to
how a single set of token embeddings gets turned into different Q, K, V vectors by
different projection matrices within one layer (§3.1 of the fundamentals file) — the
same "shared input, layer-specific projection" pattern, just shared *across* layers here
instead of within one.

```
                 ENCODER
                    │
                    ▼
             [ Encoder Output ]
                    │
          ┌─────────┼─────────┐
          │         │         │
          ▼         ▼         ▼
       Layer 1   Layer 2   Layer 3
          │         │         │
       W¹_QKV    W²_QKV    W³_QKV
          │         │         │
          ▼         ▼         ▼
       Q¹ K¹ V¹  Q² K² V²  Q³ K³
          │         │         │
          └─────────┼─────────┘
                    │
              NOT SHARED
```

### 4.3 Feed-forward

Identical role to the encoder's FFN — applied per-token, independently.

## 5. The masks working together

There are actually **three** places masking matters in an encoder-decoder model, and
it's easy to under-count them because introductory explanations usually focus only on
the causal mask. The padding mask itself is not new — it's the same general-purpose
mechanism introduced once, for all architectures, in the fundamentals file §3.6 — what's
specific to the encoder-decoder case is simply that it has to be applied in *two*
separate places instead of one:

| Mask | Where | Purpose |
|---|---|---|
| **Padding mask** (fundamentals §3.6) | Encoder self-attention, decoder cross-attention | Prevents attention to `<PAD>` tokens added to make **source** sequences in a batch the same length — padding carries no real information and must contribute a weight of ~0 |
| **Causal / look-ahead mask** | Decoder self-attention | Prevents attending to future target tokens (§4.1) |
| **Padding mask (target side)** | Decoder self-attention | In batched training, the **target** sequences are padded too, just like the source — so decoder self-attention actually needs a *combined* causal-and-padding mask: an entry is masked out if it's either a future position **or** a padding token, whichever applies. This target-side padding mask is easy to leave out of a first-pass explanation because the causal-mask diagram (§4.1) is usually drawn for one clean, un-padded example sequence, but any real batched-training setup needs it. |

## 6. Training vs inference: parallel vs sequential

- **Training**: the entire target sequence is available (it's the ground-truth label),
  so the decoder processes **all target positions in parallel** in a single forward
  pass, relying entirely on the causal mask (combined with the target-side padding mask,
  §5) to prevent cheating. This is why Transformers train much faster than the RNNs they
  replaced (RNNs are forced to be sequential even during training).
- **Inference**: the target doesn't exist yet, so generation genuinely must happen
  **one token at a time** — predict token 1, feed it back in, predict token 2, etc.
  This is the origin of the parallel-training-but-sequential-inference asymmetry that
  still defines LLM serving costs today (see the decoding strategies section in
  `03-decoder-only-architecture.md`).

## 7. Where this architecture is still used today

The full encoder-decoder design didn't disappear — it's the right tool specifically when
a task has this "fully-known input → different-but-related output sequence" shape:

- **T5** ("Text-to-Text Transfer Transformer") — reframes *every* NLP task (translation,
  summarization, classification) as text-in/text-out, using this exact architecture.
- **BART** — encoder-decoder pretrained as a denoising autoencoder (corrupt the input,
  train the decoder to reconstruct the original), strong for summarization.
- **Original machine translation systems**, and modern ones like NLLB, mBART.
- **Whisper** (speech-to-text) — encoder processes raw audio spectrogram, decoder
  generates the transcription autoregressively, with cross-attention connecting the two.

The main drawback is the cost of maintaining both an encoder and decoder, making it heavier
than a decoder-only model of comparable stack size.

## 8. BERT vs GPT vs T5 — the three families side by side


| | **BERT** (encoder-only) | **GPT** (decoder-only) | **T5** (encoder-decoder) |
|---|---|---|---|
| Stacks | One encoder stack | One decoder stack | Encoder stack **+** decoder stack, bridged by cross-attention |
| Attention masking | None — fully bidirectional (`02-encoder-only` §1, §3) | Causal / look-ahead only (`03-decoder-only` §2) | Encoder: none. Decoder: causal, **plus** cross-attention into the encoder (§4–§5 above) |
| Positional encoding | Learned absolute (fundamentals §2.1.1) | Learned absolute in the original GPT-2/3; RoPE or ALiBi in most post-2020 decoder-only models like LLaMA/Mistral (fundamentals §2.2, `03-decoder-only` §4) | Learned relative-position bucket bias (fundamentals §2.2.1, this file §8) |
| Pretraining objective | Masked Language Modeling — predict ~15% of randomly masked tokens from bidirectional context (`02-encoder-only` §4.1) | Next-token prediction over the *entire* sequence, left to right (`03-decoder-only` §1) | Span corruption: mask out contiguous spans of the input, decoder generates the missing spans as text (a generative cousin of BERT's MLM, adapted to the encoder-decoder shape) |
| Natural output shape | A label, a span within the input, or a fixed-size vector — never a freely generated new sequence (`02-encoder-only` §2) | An arbitrary-length freely generated continuation of the input | A new, potentially different-length output sequence, conditioned on a fully-known input |
| Best-fit task shape | "Understand this text" — classification, NER, semantic search/embeddings, extractive QA | "Continue/generate this text" — chat, open-ended generation, few-shot in-context tasks, code completion | "Transform this fixed input into a related but different output" — translation, summarization, structured text-to-text |
| Relative cost for a given parameter "size" | Cheapest — one stack, no generation apparatus | Cheap — one stack, but pays for autoregressive generation (KV cache, decoding, `03-decoder-only` §6–§9) | Most expensive — pays for two full stacks plus the cross-attention bridge (§7 above) |
| Can it generate free-form text? | No — no causal mask, no decoder, bidirectionality makes autoregressive generation structurally impossible (`02-encoder-only` §3) | Yes — this is its entire purpose | Yes — via its decoder half, but only conditioned on a separately-encoded input, not as a standalone continuation the way GPT works |
