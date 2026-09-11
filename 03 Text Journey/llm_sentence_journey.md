# The Journey of a Sentence Through an LLM

This document traces exactly what happens to a sentence from the moment you type it to the moment the model produces its next word — step by step, with the math made concrete.

**Example input:** `"The cat is"`
**Goal:** predict the next word (e.g., `"sleeping"`)

---

## Full Pipeline — Overview

Before diving into each step in detail, here is the entire journey in one picture. Every box below is explained in its own section further down.

```
┌────────────────────────────────────────────────────────────────────┐
│  "The cat is"                                                        │
└───────────────────────────────┬────────────────────────────────────┘
                                 ▼
┌────────────────────────────────────────────────────────────────────┐
│  STEP 1 — Tokenization                                                │
│  "The cat is"  →  ["The", " cat", " is"]  →  [464, 3797, 318]        │
└───────────────────────────────┬────────────────────────────────────┘
                                 ▼
┌────────────────────────────────────────────────────────────────────┐
│  STEP 2 — Embedding (+ positional encoding)                          │
│  Each token ID  →  one vector of d_model numbers (e.g. 4096)         │
│  Shape: [3 tokens, 4096]                                              │
└───────────────────────────────┬────────────────────────────────────┘
                                 ▼
┌────────────────────────────────────────────────────────────────────┐
│  STEP 3 — Transformer Layers  (repeated N times, e.g. N = 32)        │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  one Transformer layer:                                        │  │
│  │   3a) Self-Attention   → tokens exchange context                │  │
│  │   3b) Feed-Forward Net → each token refined individually        │  │
│  │   3c) Residual + Norm  → keeps values stable                    │  │
│  └──────────────────────────────────────────────────────────────┘  │
│  Shape IN = Shape OUT = [3 tokens, 4096]  at every single layer      │
└───────────────────────────────┬────────────────────────────────────┘
                                 ▼
┌────────────────────────────────────────────────────────────────────┐
│  STEP 4 — Take the Last Token's Final Hidden State                   │
│  [3 tokens, 4096]  →  keep only the last one  →  [1, 4096]           │
└───────────────────────────────┬────────────────────────────────────┘
                                 ▼
┌────────────────────────────────────────────────────────────────────┐
│  STEP 5 — LM Head (a separate weight matrix, applied ONCE)           │
│  [1, 4096]  ×  LM_Head [4096, 50000]  =  logits [1, 50000]           │
│  (one raw score per word in the entire vocabulary)                   │
└───────────────────────────────┬────────────────────────────────────┘
                                 ▼
┌────────────────────────────────────────────────────────────────────┐
│  STEP 6 — Softmax (optionally scaled by Temperature)                 │
│  logits [1, 50000]  →  probabilities [1, 50000]  (sums to 100%)      │
└───────────────────────────────┬────────────────────────────────────┘
                                 ▼
┌────────────────────────────────────────────────────────────────────┐
│  STEP 7 — Decoding: pick ONE token (Greedy / Sampling)               │
│  probabilities  →  "sleeping"                                        │
└───────────────────────────────┬────────────────────────────────────┘
                                 ▼
┌────────────────────────────────────────────────────────────────────┐
│  STEP 8 — Append token, feed back in as new input, repeat            │
│  "The cat is" + "sleeping"  →  "The cat is sleeping"  → Step 2 again │
└────────────────────────────────────────────────────────────────────┘
```

---

## Step 1 — Tokenization

The raw text is broken into **tokens**. Tokens aren't always whole words — they can be sub-words, punctuation, or even single characters, depending on the tokenizer (e.g., BPE — Byte Pair Encoding).

```
Input text:  "The cat is"
Tokens:      ["The", " cat", " is"]
Token IDs:   [464, 3797, 318]
```

Each token is mapped to a unique integer ID from the model's **vocabulary** (a fixed list of all possible tokens, often 30,000–150,000+ entries).

```
 INPUT                              OUTPUT
┌─────────────────┐   tokenizer   ┌───────┬────────┬───────┐
│ "The cat is"     │ ───────────► │ "The" │ " cat" │ " is" │
└─────────────────┘               └───────┴────────┴───────┘
                                        │        │       │
                                        ▼        ▼       ▼
                                      464      3797     318
                                   (Token IDs, looked up in vocabulary)
```

---

## Step 2 — Embedding

Each token ID is converted into a **vector** — a list of numbers representing that token's meaning in a high-dimensional space (e.g., 4096 dimensions for a large model).

```
Token 464  ("The") → [0.12, -0.87, 0.33, ..., 0.05]   (4096 numbers)
Token 3797 (" cat") → [0.91, 0.14, -0.22, ..., 0.67]
Token 318  (" is")  → [-0.05, 0.44, 0.19, ..., -0.31]
```

This lookup happens via an **embedding matrix** of shape `[vocab_size × hidden_size]`. Similar words end up with similar vectors (this is learned during training, not hand-coded).

### Positional Information

Since the model has no inherent sense of word order, a **positional encoding** is added to each embedding so the model knows token 1 came before token 2, etc.

```
Final input vector = Token Embedding + Positional Encoding
```

```
 INPUT                    OUTPUT
┌──────┐   embedding    ┌──────────────────────────────┐
│ 464  │ ─────matrix──► │ [0.12, -0.87, 0.33, ..., 0.05]│ 4096 numbers
└──────┘   lookup       └──────────────────────────────┘
┌──────┐                ┌──────────────────────────────┐
│ 3797 │ ─────────────► │ [0.91,  0.14,-0.22, ..., 0.67]│ 4096 numbers
└──────┘                └──────────────────────────────┘
┌──────┐                ┌──────────────────────────────┐
│ 318  │ ─────────────► │[-0.05,  0.44, 0.19, ...,-0.31]│ 4096 numbers
└──────┘                └──────────────────────────────┘
                                     +
                        ┌──────────────────────────────┐
                        │   positional encoding vectors │  (position 1,2,3)
                        └──────────────────────────────┘
                                     ▼
                        3 vectors × 4096 dims, position-aware
```

---

## Step 3 — The Transformer Layers

The sequence of vectors now passes through a stack of **Transformer blocks** (modern LLMs have anywhere from a few dozen to over 100 of these). Each block does two main things:

### 3a) Self-Attention

For every token, the model asks: *"Which other tokens in this sentence matter most for understanding me?"*

Each token generates three vectors:
- **Query (Q)** — what am I looking for?
- **Key (K)** — what do I contain?
- **Value (V)** — what information do I offer?

The attention score between two tokens is computed as:

```
Attention(Q, K, V) = softmax( (Q · Kᵀ) / √d_k ) · V
```

For `"The cat is"`, when processing the word `" is"`, attention might assign:

```
"The" → 10% relevance
"cat" → 85% relevance   ← strongly attends to "cat" (the subject)
"is"  → 5%  relevance (attending to itself)
```

This lets the model understand that `" is"` relates grammatically to `"cat"`, not `"The"`.

```
 INPUT: 3 token vectors, each turned into a Query, a Key, and a Value

     "The"        " cat"        " is"
   ┌────────┐   ┌────────┐   ┌────────┐
   │Q  K  V │   │Q  K  V │   │Q  K  V │
   └────────┘   └────────┘   └────────┘

 Focus on token " is" — take ITS Query, compare to EVERY token's Key:

   Q(" is") · K("The")  = score 1.2  ─┐
   Q(" is") · K(" cat") = score 4.8   ├─► softmax ─►  10%, 85%, 5%
   Q(" is") · K(" is")  = score 0.6  ─┘             (weights, sum = 100%)

 OUTPUT: new vector for " is" = weighted blend of all Values

   0.10 × V("The")  +  0.85 × V(" cat")  +  0.05 × V(" is")
                         │
                         ▼
        ┌──────────────────────────────┐
        │  new " is" vector (4096 dims)  │   ← now "knows" about "cat"
        └──────────────────────────────┘

 (This whole process repeats for "The" and " cat" too — every token
  builds its own new vector by attending to all the others.)
```

Modern models use **multi-head attention** — this same process runs in parallel many times (e.g., 32 or 64 "heads"), each potentially capturing a different kind of relationship (grammar, meaning, position, etc.).

### 3b) Feed-Forward Network (FFN)

After attention mixes information *between* tokens, each token's vector individually passes through a small neural network (usually two linear layers with a non-linear activation in between) that further transforms the representation.

```
Output = W₂ · activation(W₁ · x + b₁) + b₂
```

This runs **separately on each token's vector** — unlike attention, tokens do not exchange information here.

```
 INPUT: one token's vector, e.g. the new " is" vector from attention

   [4096 numbers]
        │
        ▼
   ┌─────────┐   Linear layer W₁: expands  4096 → 16384
   │   W₁    │   (gives the network more room to compute in)
   └────┬────┘
        ▼
   [16384 numbers]
        │
        ▼
   ┌─────────┐   Non-linear activation (e.g. GELU)
   │  act.   │   (lets the model learn non-linear patterns,
   └────┬────┘    not just straight-line combinations)
        ▼
   [16384 numbers]
        │
        ▼
   ┌─────────┐   Linear layer W₂: projects back  16384 → 4096
   │   W₂    │
   └────┬────┘
        ▼
 OUTPUT: [4096 numbers]  ← same shape as input, but refined

 (The same 3 sub-steps run independently for "The" and " cat" too,
  each using the exact same W₁ / W₂ weights.)
```

### 3c) Residual Connections + Normalization

Each sub-step's output is added back to its input (a "residual connection") and normalized, which keeps training stable across many stacked layers.

```
x = LayerNorm(x + Attention(x))
x = LayerNorm(x + FFN(x))
```

This whole block (Attention → FFN → Norms) repeats **N times** (once per layer), with each layer refining the representation further — earlier layers tend to capture syntax and local patterns, deeper layers capture more abstract meaning and long-range context.

```
Input embeddings → Layer 1 → Layer 2 → ... → Layer N → Final hidden states
```

The key thing to notice: **the shape never changes.** Every layer takes in 3 vectors of 4096 numbers and returns 3 vectors of 4096 numbers — only the *content* of the numbers gets refined each time.

```
 INPUT                       OUTPUT               becomes INPUT of next layer
Shape [3, 4096]           Shape [3, 4096]

┌───────────────┐   ┌────────────────────────┐   ┌───────────────┐
│  "The"         │   │   LAYER 1               │   │  "The"  (v1)   │
│  " cat"        │──►│   3a) Self-Attention    │──►│  " cat" (v1)   │
│  " is"         │   │   3b) FFN               │   │  " is"  (v1)   │
│  (embeddings)  │   │   3c) Residual + Norm   │   │                │
└───────────────┘   └────────────────────────┘   └───────┬───────┘
                                                            ▼
                     ┌────────────────────────┐   ┌───────────────┐
                     │   LAYER 2               │   │  "The"  (v2)   │
                     │   (identical structure, │──►│  " cat" (v2)   │
                     │    its own weights)     │   │  " is"  (v2)   │
                     └────────────────────────┘   └───────┬───────┘
                                                            ▼
                              ⋮  repeats for Layers 3 … N-1 ⋮
                                                            ▼
                     ┌────────────────────────┐   ┌───────────────┐
                     │   LAYER N (last one)    │   │  "The"  (vN)   │
                     │                         │──►│  " cat" (vN)   │
                     └────────────────────────┘   │  " is"  (vN)   │
                                                    └───────────────┘
                                                    Shape [3, 4096]
                                                    = "Final hidden states"
                                                    (used in Step 4 next)
```

---

## Step 4 — The Final Hidden State

After the last Transformer layer, we take the hidden state vector corresponding to the **last token** in the sequence (`" is"`), since that represents the model's understanding of the entire sentence so far, ready to predict what comes next.

```
Final hidden state for " is" → [1.02, -0.44, 0.78, ..., 0.09]   (still 4096 numbers)
```

```
 INPUT                                    OUTPUT (only last token kept)
┌────────┬────────┬────────┐
│ vector │ vector │ vector │   pick last   ┌──────────────────────────┐
│ "The"  │ " cat" │ " is"  │ ────────────► │ [1.02, -0.44, ..., 0.09] │
└────────┴────────┴────────┘   token       └──────────────────────────┘
   (all 3 tokens'                            "summary" of whole sentence
    final vectors)                             so far, ready to predict
```

---

## Step 5 — The LM Head (Unembedding)

This final vector is multiplied by one last large matrix, the **LM Head** (shape `[hidden_size × vocab_size]`), which converts it into one raw score — a **logit** — for *every single token in the vocabulary*.

```
hidden_state (1 × 4096)  ×  LM_Head (4096 × 50000)  =  logits (1 × 50000)
```

```
 INPUT                          OUTPUT (one score per vocab word)
┌──────────────────┐  × LM_Head ┌─────────────┬────────┐
│ hidden state      │ ─────────►│ "sleeping"   │  4.8   │
│ (4096 numbers)    │  matrix   ├─────────────┼────────┤
└──────────────────┘  (4096 ×  │ "hungry"     │  3.1   │
                        50000)  ├─────────────┼────────┤
                                │ "outside"    │  1.9   │
                                ├─────────────┼────────┤
                                │ "purple"     │ -2.4   │
                                ├─────────────┼────────┤
                                │ "running"    │  0.7   │
                                ├─────────────┼────────┤
                                │ ... 49,995   │  ...   │  (rest of vocab)
                                │ more words   │        │
                                └─────────────┴────────┘
                                  raw, unbounded LOGITS
```

Example (simplified to 5 candidate words):

```
"sleeping" → logit: 4.8
"hungry"   → logit: 3.1
"outside"  → logit: 1.9
"purple"   → logit: -2.4
"running"  → logit: 0.7
```

These numbers are **not probabilities** — they're raw, unbounded scores. Higher means the model considers that token more likely to come next, but they don't sum to anything meaningful yet.

---

## Step 6 — Softmax (and optionally Temperature)

The logits are converted into a proper probability distribution using **softmax**:

```
P(x_i) = exp(z_i) / Σ exp(z_j)
```

If a **temperature (T)** is applied, the logits are divided by T *before* the softmax:

```
P(x_i) = exp(z_i / T) / Σ exp(z_j / T)
```

- T < 1 → sharper distribution (more confident/greedy-like)
- T > 1 → flatter distribution (more randomness)

Result (with T = 1):

```
"sleeping" → 61%
"hungry"   → 14%
"outside"  → 4%
"purple"   → 0.1%
"running"  → 2%
(...remaining probability spread across the other ~49,995 vocabulary tokens)
```

**Important:** softmax only calculates the distribution — it does not pick anything by itself.

```
 INPUT (logits)                    OUTPUT (probabilities, sum = 100%)
┌─────────────┬────────┐          ┌─────────────┬────────┐
│ "sleeping"   │  4.8   │          │ "sleeping"   │  61%   │  ██████████████
│ "hungry"     │  3.1   │ softmax  │ "hungry"     │  14%   │  ███
│ "outside"    │  1.9   │ ───────► │ "outside"    │   4%   │  █
│ "purple"     │ -2.4   │  (/ T)   │ "purple"     │ 0.1%   │  ▏
│ "running"    │  0.7   │          │ "running"    │   2%   │  ▏
│ ...          │  ...   │          │ ...          │  ~19%  │  ████ (rest of vocab)
└─────────────┴────────┘          └─────────────┴────────┘
```

---

## Step 7 — Decoding (the actual choice)

Now the model must pick one token. This is a separate, configurable step:

| Method | What happens |
|---|---|
| **Greedy** | Always pick the highest-probability token (`"sleeping"`, 61%) |
| **Sampling** | Randomly draw from the distribution, weighted by probability — usually combined with **top-k** (only consider the k highest-probability tokens) and/or **top-p** (only consider the smallest set of tokens whose cumulative probability ≥ p) to avoid picking absurd low-probability tokens |

Suppose sampling is used and `"sleeping"` is drawn (most likely outcome given 61%):

```
Chosen token: "sleeping"
```

```
 INPUT (probability distribution)          OUTPUT (single chosen token)
┌─────────────┬────────┐
│ "sleeping"   │  61%   │──┐
│ "hungry"     │  14%   │──┤   GREEDY: always pick max
│ "outside"    │   4%   │──┼──────────────────────────►  "sleeping"
│ "purple"     │ 0.1%   │──┤
│ "running"    │   2%   │──┘
└─────────────┴────────┘
                                            SAMPLING: weighted random draw
                                            (top-k / top-p filter first)
                            ─────────────►  "sleeping"  (61% of the time)
                                            "hungry"    (14% of the time)
                                            ... etc.
```

---

## Step 8 — Append and Repeat (Autoregression)

The chosen token is appended to the sequence, and the **entire process restarts** from Step 2 (embedding) — but now the input is `"The cat is sleeping"`, and the model predicts the *next* token after that.

```
"The cat is"            → predict → "sleeping"
"The cat is sleeping"   → predict → "."
"The cat is sleeping."  → predict → <end of sequence>
```

```
 INPUT                          OUTPUT (appended, fed back as new input)
┌──────────────┐   full model   ┌───────────┐
│"The cat is"  │ ─────pipeline─►│"sleeping" │
└──────┬───────┘   (steps 2-7)  └─────┬─────┘
       │                              │
       │        append token          │
       └──────────────◄───────────────┘
       ▼
┌────────────────────┐  full model   ┌───────┐
│"The cat is sleeping"│─────pipeline─►│  "."  │
└──────────┬──────────┘  (steps 2-7) └───┬───┘
           │                             │
           └─────────────◄───────────────┘
           ▼
┌─────────────────────┐ full model  ┌─────────┐
│"The cat is sleeping."│────pipeline►│ <end>   │
└──────────────────────┘ (steps 2-7)└─────────┘
        loop stops — full sentence generated
```

This one-token-at-a-time loop is why LLMs are called **autoregressive** models — each new token depends on all the tokens generated before it.

---

## Full Pipeline Summary

```
Text
  ↓ tokenization
Tokens (IDs)
  ↓ embedding + positional encoding
Vectors
  ↓ N × [Self-Attention → FFN → Residual/Norm]  (Transformer layers)
Final hidden state
  ↓ × LM Head matrix
Logits (one score per vocabulary word)
  ↓ softmax (optionally scaled by temperature)
Probabilities
  ↓ decoding (greedy / sampling with top-k, top-p)
Chosen token
  ↓ append to sequence, repeat
Full generated sentence
```
