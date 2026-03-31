# Latent Reasoner Architecture Design

## Problem Statement

Given a transformer processing `input <think> ... reasoning tokens ... </think> answer`,
we capture the hidden state at `<think>` (h_start) and `</think>` (h_end). The latent
reasoner must learn the mapping `h_start → h_end` — replacing N tokens of chain-of-thought
with a fixed latent computation module.

## Design Constraints

| Constraint | Detail |
|-----------|--------|
| Input/output | Single hidden-state vector (h_start → h_end) |
| Dimensionality | GPT-2: 768d, LLaMA-7B: 4096d — must handle both |
| Cross-model | Architecture should work across model families |
| Training stability | Gradient must flow cleanly through N iterative steps |
| Inference cost | Must be cheaper than generating N reasoning tokens |

## Architecture Candidates

We recommend three candidates ordered by complexity. All three share a common
interface: `(h_start: Tensor[B, D]) → (h_end: Tensor[B, D])`.

---

### Candidate 1: Looped Residual GRU (Recommended First Experiment)

**Core idea**: A GRU cell applied iteratively, refining h_start over N steps with
shared weights. The most direct analogy to "thinking step by step" — each application
is one "reasoning step" in latent space.

**Architecture**:
```
h_0 = project_in(h_start)           # D_model → D_latent
for t in 1..N:
    h_t = GRU(h_{t-1}, h_0)         # h_0 as persistent "query"
    h_t = h_t + h_{t-1}             # residual connection
    h_t = LayerNorm(h_t)
h_end = project_out(h_N)            # D_latent → D_model
```

**Why GRU over LSTM**: Fewer parameters (2 gates vs 3), empirically matches LSTM
on short sequences, and the hidden state is the output (no separate cell state to
manage), simplifying the residual path.

**Key design choices**:
- **Shared weights** across all N steps (single GRU cell reused). Dramatically
  reduces parameters. Unique-per-step weights can be tried later but shared is
  the right default — Graves 2016 (ACT) showed shared weights work well for
  learned iteration.
- **h_0 as GRU input at every step**: The original hidden state acts as a
  persistent "problem statement" that each refinement step attends to.
- **Residual + LayerNorm**: Critical for gradient flow through N steps.
  Without residual connections, gradients vanish by step ~8.

**Dimensionality handling**:
- `project_in`: Linear(D_model, D_latent) + LayerNorm
- `project_out`: Linear(D_latent, D_model)
- D_latent is a hyperparameter (e.g., 512 or 1024), decoupled from D_model.
  This is how we achieve cross-model compatibility: all models project into the
  same latent space.

**Adaptive computation (phase 2)**:
Add a halting unit per step: `p_halt = sigmoid(Linear(h_t))`. Accumulate halting
probabilities; stop when cumulative probability exceeds threshold (ACT mechanism,
Graves 2016). Adds a ponder cost penalty to the loss: `L_ponder = τ · N_mean`.

**Training stability**:
- Gradient clipping (max_norm=1.0) is essential
- Learning rate warmup over first 1000 steps
- N ≤ 16 for initial experiments (gradient flow validated empirically)
- Monitor `‖h_N - h_0‖ / ‖h_0‖` — should grow then plateau, not explode

**Estimated parameters** (D_latent=512):
- project_in: D_model × 512 + 512
- GRU cell: 3 × (512 × 512 + 512 × 512 + 512) = ~1.6M
- project_out: 512 × D_model + D_model
- Total for GPT-2 (768d): ~2.4M params
- Total for LLaMA-7B (4096d): ~5.8M params

**Pros**: Natural iterative refinement; shared weights = parameter efficient;
well-understood gradient dynamics; easy to add adaptive halting later.

**Cons**: Sequential computation (N forward passes); may plateau in expressiveness
for complex reasoning.

---

### Candidate 2: Latent Transformer Block

**Core idea**: Project h_start into a small set of K latent tokens, run them through
a miniature transformer (2–4 layers), then read out h_end. Reasoning happens via
self-attention among latent tokens — parallel "deliberation."

**Architecture**:
```
tokens = broadcast(project_in(h_start), K)   # [B, K, D_latent]
tokens = tokens + positional_encoding(K)
for layer in transformer_layers:
    tokens = layer(tokens)                     # standard pre-norm transformer
h_end = project_out(mean_pool(tokens))         # [B, D_model]
```

**Broadcast strategy**: The simplest approach projects h_start to K × D_latent and
reshapes. Alternatively, use K learned "slot" vectors added to the projected input
(like Perceiver latent array).

**Key design choices**:
- **K = 8–16 latent tokens**: Small enough to be fast, large enough to represent
  multi-faceted reasoning. K is analogous to "reasoning steps" but computed in parallel.
- **2–4 transformer layers**: More layers = deeper reasoning. Start with 2.
- **Pre-norm (LayerNorm before attention/FFN)**: More stable than post-norm for
  small models.
- **Mean pooling for readout**: Simpler than learned readout attention, sufficient
  for initial experiments. Can upgrade to cross-attention readout later.

**Dimensionality handling**: Same projection strategy as Candidate 1. D_latent is
independent of D_model.

**Training stability**:
- Standard transformer training practices apply
- No recurrence = no vanishing gradient through steps
- Attention weights provide interpretability (which latent tokens attend to what)

**Estimated parameters** (D_latent=512, K=8, 2 layers):
- project_in: D_model × (K × 512)
- Each transformer layer: ~4 × 512² = ~1M (attn + FFN)
- project_out: 512 × D_model
- Total for GPT-2: ~6M params
- Total for LLaMA-7B: ~36M params (dominated by projection)

Note: projection cost is high for large models. Could use a bottleneck projection
(D_model → 256 → K × D_latent) to reduce this.

**Pros**: Parallel computation (no sequential steps); self-attention = natural
"deliberation"; interpretable via attention maps; proven architecture.

**Cons**: Higher parameter count; quadratic in K (but K is small); less natural
analogy to iterative refinement; harder to add adaptive computation.

---

### Candidate 3: Deep Residual MLP (Baseline)

**Core idea**: A stack of residual MLP blocks. No recurrence, no attention — just
depth. This is the simplest possible architecture and serves as the critical baseline.
If the GRU or Transformer can't beat this, the iterative/attention mechanisms aren't
contributing.

**Architecture**:
```
h = project_in(h_start)              # D_model → D_latent
for block in residual_blocks:
    h = h + block(h)                  # block = LN → Linear → GELU → Linear
h_end = project_out(h)               # D_latent → D_model
```

**Key design choices**:
- **6–12 residual blocks**: Each block is `LayerNorm → Linear(D, 4D) → GELU → Linear(4D, D)`.
  Standard transformer FFN architecture, just stacked.
- **No weight sharing**: Each block has unique weights (unlike Candidate 1).
  This is deliberate — the baseline should have maximum expressiveness per parameter.
- **GELU activation**: Matches transformer internals. Smooth gradient everywhere.

**Dimensionality handling**: Same projection layers as other candidates.

**Training stability**:
- Residual connections make this extremely stable even at 12 blocks
- Pre-norm (LayerNorm before each block) prevents representation collapse
- Simplest gradient dynamics of all three candidates

**Estimated parameters** (D_latent=512, 8 blocks):
- Each block: 2 × (512 × 2048) + 2048 + 512 = ~2.1M
- 8 blocks: ~17M
- Projections: ~1–4M depending on D_model
- Total for GPT-2: ~18M
- Total for LLaMA-7B: ~21M

**Pros**: Dead simple to implement (< 50 LOC); extremely stable training; strong
baseline — if this works, we know the problem is learnable.

**Cons**: Fixed computation path (no adaptivity); more parameters than GRU for
comparable reasoning depth; no obvious interpretability.

---

## Comparison Matrix

| Property | Looped GRU | Latent Transformer | Residual MLP |
|----------|-----------|-------------------|-------------|
| Parameters (GPT-2) | ~2.4M | ~6M | ~18M |
| Parameters (LLaMA-7B) | ~5.8M | ~36M | ~21M |
| Iterative refinement | Yes (N steps) | No (parallel) | No (fixed) |
| Adaptive depth | Easy (ACT) | Hard | No |
| Training stability | Moderate (needs clip+LN) | Good | Excellent |
| Interpretability | Step-by-step h trajectory | Attention maps | None |
| Implementation complexity | Low | Medium | Very low |
| Inference cost | N × GRU forward | 1 transformer forward | 1 MLP forward |
| Analogy to CoT | Strong (iterative) | Moderate (parallel deliberation) | Weak |

## Recommended Experiment Plan

### Phase 1: Baseline validation (Residual MLP)

1. Implement Candidate 3 (Residual MLP) first — it's the simplest and establishes
   whether the h_start → h_end mapping is learnable at all.
2. Train on GPT-2 hidden states from a math reasoning dataset (GSM8K or similar).
3. Metric: MSE(predicted_h_end, actual_h_end) + downstream task accuracy when
   predicted h_end is injected back into the model.

### Phase 2: Iterative reasoning (Looped GRU)

1. Implement Candidate 1 (Looped GRU) with N=8 steps, shared weights.
2. Compare against MLP baseline at matched parameter count (adjust D_latent).
3. Key question: does iterative refinement improve over a single deep pass?
4. Visualize the h trajectory across steps — is it doing meaningful refinement?

### Phase 3: Parallel deliberation (Latent Transformer)

1. Implement Candidate 2 (Latent Transformer) with K=8 tokens, 2 layers.
2. Compare against both baselines.
3. Key question: does multi-token parallel reasoning capture something the
   single-vector approaches miss?

### Phase 4: Adaptive computation

1. Add ACT halting to the best-performing iterative architecture (likely GRU).
2. Measure: does the model learn to use more steps for harder problems?
3. This is the key result for the "latent CoT" hypothesis.

## Cross-Model Compatibility Strategy

The projection layers (project_in, project_out) are the only model-specific
components. The latent reasoner core operates in a fixed D_latent space.

**Strategy**: Train with data from multiple models simultaneously.

```
GPT-2 (768d) ──→ project_in_gpt2 ──→ ┌─────────────────┐ ──→ project_out_gpt2 ──→ (768d)
                                       │ Shared Latent    │
LLaMA (4096d) ──→ project_in_llama ──→│ Reasoner Core    │──→ project_out_llama ──→ (4096d)
                                       │ (D_latent = 512) │
Mistral (4096d)──→ project_in_mist ──→└─────────────────┘ ──→ project_out_mist ──→ (4096d)
```

Each model family gets its own projection pair. The core reasoner is shared.
If the Platonic Representation Hypothesis holds, the shared core should find
a common reasoning structure across model families.

## Training Stability Recommendations

1. **Gradient clipping** (max_norm=1.0): Essential for GRU, recommended for all.
2. **Learning rate**: 1e-4 with cosine schedule, 1000 steps warmup.
3. **Batch size**: Start with 32, scale up if stable.
4. **Loss**: MSE on h_end, optionally + downstream cross-entropy on model output.
5. **Monitoring**: Track `‖h_pred - h_true‖₂`, gradient norms per layer,
   and (for GRU) the h trajectory norm across steps.
6. **Regularization**: Dropout 0.1 within the latent reasoner. No weight decay
   on LayerNorm parameters.

## Open Questions for Future Research

1. **Which layer?** Should we capture h_start/h_end at the last layer, a middle
   layer, or multiple layers? Start with the residual stream at the last layer
   (most semantically rich), but multi-layer approaches may work better.
2. **Bidirectional training**: Can we train h_end → h_start (reasoning "backwards")
   to validate the representation quality?
3. **Curriculum**: Should we start with short CoT chains and increase length?
4. **Distillation vs reconstruction**: Is MSE on h_end the right loss, or should
   we use a learned discriminator (adversarial) or contrastive loss?
