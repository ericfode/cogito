# Cogito v0.1 — Experimental Protocol

> **Goal**: Validate the core hypothesis — can a latent reasoner learn to map
> `h_start → h_end` (hidden states at `<think>` and `</think>` boundaries),
> replacing explicit chain-of-thought tokens with latent-space computation?

---

## 1. Training Data

### 1.1 Task Selection

We use four reasoning benchmarks spanning distinct reasoning types to ensure the
latent reasoner isn't overfitting to a single reasoning modality:

| Benchmark   | Reasoning Type       | Format                       | Train Size | Test Size | Typical CoT Length |
|-------------|---------------------|------------------------------|------------|-----------|-------------------|
| **GSM8K**   | Multi-step arithmetic | Natural language math problem → numeric answer | 7,473 | 1,319 | 3–8 steps, ~100–200 tokens |
| **ARC-Challenge** | Science/commonsense | Multiple-choice science questions (grade-school level) | 1,119 | 1,172 | 2–5 steps, ~50–150 tokens |
| **StrategyQA** | Multi-hop boolean | Yes/no questions requiring implicit decomposition | 2,290 | 490 (dev) | 2–4 hops, ~60–120 tokens |
| **LogiQA**  | Formal logic        | Reading comprehension with logical reasoning, 4-choice | 7,376 | 651 | 2–6 steps, ~80–180 tokens |

**Rationale**: GSM8K tests sequential numeric computation. ARC tests
retrieval-augmented scientific reasoning. StrategyQA tests implicit
decomposition (the question doesn't tell you what to decompose). LogiQA tests
formal deductive reasoning. Together they cover arithmetic, scientific,
strategic, and logical reasoning.

**Combined training pool**: ~18,258 examples (before CoT augmentation).

### 1.2 Chain-of-Thought Trace Sources

For each benchmark, we need high-quality CoT traces to serve as the "reasoning"
between `<think>` and `</think>` tokens. Sources:

1. **GSM8K**: Ships with human-written step-by-step solutions. Use directly.
   These are gold-standard traces.

2. **ARC-Challenge**: No official CoT. Generate using a strong teacher model
   (e.g., GPT-4o or Claude 3.5 Sonnet) with the prompt template:
   ```
   Question: {question}
   Choices: {choices}
   Think step by step, then give the answer letter.
   ```
   Filter: keep only traces that produce the correct answer.

3. **StrategyQA**: Official dataset includes decomposition and evidence
   paragraphs. Convert to CoT format:
   `Decomposition step 1 → evidence → ... → therefore yes/no.`

4. **LogiQA**: No official CoT. Generate with teacher model (same approach as
   ARC). Filter for correctness.

**Quality control for generated traces**:
- Reject traces where the final answer is wrong
- Reject traces shorter than 20 tokens (likely guessing, not reasoning)
- Reject traces longer than 512 tokens (diminishing returns, noise)
- Sample-and-verify: manually inspect 50 traces per benchmark for coherence

**Expected yield**: ~15,000–17,000 usable (question, CoT, answer) triples after
filtering.

### 1.3 Hidden State Capture Engineering

For each (question, CoT, answer) triple, we need to capture hidden states from
each base model at the reasoning boundaries.

**Token protocol**:
```
{question} <think> {CoT trace} </think> {answer}
```

We add `<think>` and `</think>` as special tokens to each model's tokenizer
(resize embedding layer accordingly). These tokens serve as sentinel positions
for hidden state capture.

**Capture procedure** (per model, per example):
1. Tokenize the full sequence including `<think>` and `</think>` markers
2. Run a forward pass through the model (inference mode, no grad)
3. Record the hidden state at **every layer** at the `<think>` token position → `h_start[layer]`
4. Record the hidden state at **every layer** at the `</think>` token position → `h_end[layer]`
5. Store as `(h_start, h_end)` pairs per layer

**Storage format**: HDF5 files organized as:
```
data/
  hidden_states/
    {model_name}/
      {benchmark}/
        layer_{L}.h5    # shape: (N, 2, hidden_dim)  — [start, end] per example
```

**Storage estimate** (per model at one layer, N=17,000, float16):
- GPT-2 Small (768-dim): 17,000 × 2 × 768 × 2 bytes ≈ 50 MB
- LLaMA-3.2-3B (3072-dim): 17,000 × 2 × 3072 × 2 bytes ≈ 200 MB
- Capturing all layers across all models: ~50–100 GB total (manageable on disk)

**Implementation notes**:
- Use `torch.no_grad()` and `model.eval()` throughout
- Process in batches (batch size depends on model; start with 8, reduce if OOM)
- Use `output_hidden_states=True` in HuggingFace `.forward()` call
- Pad sequences per-batch; record attention masks to ensure `<think>`/`</think>`
  positions are correct despite padding

---

## 2. Base Models

### 2.1 Model Selection

Maximum architecture diversity, minimum compute. All models must be:
- Open-weight and available on HuggingFace
- Small enough to run inference on a single A100 (80GB)
- Decoder-only transformers (consistent hidden state semantics)

| Model | Params | Hidden Dim | Layers | VRAM (fp16 inference) | License |
|-------|--------|-----------|--------|----------------------|---------|
| GPT-2 Small | 124M | 768 | 12 | ~0.5 GB | MIT |
| GPT-2 Medium | 355M | 1024 | 24 | ~1.4 GB | MIT |
| GPT-2 Large | 774M | 1280 | 36 | ~3.0 GB | MIT |
| Pythia-70M | 70M | 512 | 6 | ~0.3 GB | Apache 2.0 |
| Pythia-160M | 160M | 768 | 12 | ~0.6 GB | Apache 2.0 |
| Pythia-410M | 410M | 1024 | 24 | ~1.6 GB | Apache 2.0 |
| Pythia-1B | 1.0B | 2048 | 16 | ~4.0 GB | Apache 2.0 |
| Pythia-1.4B | 1.4B | 2048 | 24 | ~5.5 GB | Apache 2.0 |
| Phi-2 | 2.7B | 2560 | 32 | ~5.5 GB | MIT |
| Phi-3-mini (3.8B) | 3.8B | 3072 | 32 | ~7.6 GB | MIT |
| LLaMA-3.2-1B | 1.2B | 2048 | 16 | ~2.5 GB | Llama 3.2 Community |
| LLaMA-3.2-3B | 3.2B | 3072 | 28 | ~6.5 GB | Llama 3.2 Community |
| Gemma-2-2B | 2.6B | 2304 | 26 | ~5.2 GB | Gemma license |

**13 models total**, spanning 70M–3.8B parameters, 512–3072 hidden dims, 6–36 layers.

**Architecture diversity**:
- GPT-2: Original transformer decoder, learned positional embeddings
- Pythia: EleutherAI, rotary positional embeddings, same training data (The Pile)
  across all sizes → isolates scale effects
- Phi-2/3: Microsoft, high-quality curated training data ("textbook quality")
- LLaMA-3.2: Meta, GQA attention, RoPE, trained on large-scale web data
- Gemma-2: Google DeepMind, sliding window + global attention hybrid

### 2.2 Why These Models

- **GPT-2 family**: Baseline architecture, fast, well-understood. The control group.
- **Pythia family**: Same architecture and training data across scales. Lets us
  isolate the effect of model size on hidden state reasoning structure.
- **Phi-2/3**: Disproportionately capable for their size due to data quality.
  Tests whether training data quality affects latent reasoning structure.
- **LLaMA-3.2**: State-of-the-art small models. Tests whether modern training
  recipes produce different hidden state dynamics.
- **Gemma-2**: Different attention pattern (sliding window). Tests architectural
  variation beyond positional encoding.

### 2.3 VRAM Budget

All 13 models sum to ~36 GB for fp16 inference weights. An A100 (80GB) can
comfortably hold the largest model (Phi-3-mini at ~7.6 GB) with room for
activations and hidden state buffers. Models will be loaded one at a time for
hidden state capture.

---

## 3. Baselines

Five baselines to contextualize the latent reasoner's performance:

### 3.1 No CoT (Baseline B0)

**Method**: Direct prediction — model sees question, produces answer with no
reasoning tokens.

**Purpose**: Lower bound. How well can each model answer without any reasoning?

**Implementation**: Standard few-shot prompting (5-shot) with format:
```
Q: {question}
A: {answer}
```

### 3.2 Full Token CoT (Baseline B1 — Upper Bound)

**Method**: Standard chain-of-thought — model generates full reasoning trace in
natural language tokens.

**Purpose**: Upper bound. This is the performance ceiling that latent reasoning
aims to match.

**Implementation**: Few-shot prompting with CoT format:
```
Q: {question}
<think>{CoT trace}</think>
A: {answer}
```
Evaluate the answer token(s) after `</think>`.

### 3.3 Random Hidden State (Baseline B2 — Ablation)

**Method**: Replace the predicted `h_end` with a random vector sampled from the
empirical distribution of `h_end` vectors (same mean and covariance).

**Purpose**: Ablation control. If the latent reasoner's predicted `h_end` isn't
better than a random draw from the right distribution, it hasn't learned
meaningful structure.

**Implementation**:
1. Compute empirical mean `μ` and covariance `Σ` of all `h_end` vectors in the
   training set
2. Sample `h_random ~ N(μ, Σ)` for each test example
3. Inject at `</think>` position, measure downstream accuracy

### 3.4 Linear Projection (Baseline B3 — Minimal Learner)

**Method**: Train a simple affine transformation `h_end = W·h_start + b`.

**Purpose**: Tests how much of the `h_start → h_end` mapping is linearly
recoverable. If a linear probe captures most of the signal, the reasoning
transform may be simpler than expected.

**Implementation**:
- Train `W ∈ R^{d×d}`, `b ∈ R^d` via MSE loss on `(h_start, h_end)` pairs
- Per-model, per-layer training (no cross-model sharing)
- Optimizer: Adam, lr=1e-3, 50 epochs, batch size 256
- Regularization: weight decay 1e-4

### 3.5 Coconut Reproduction (Baseline B4 — Prior Art)

**Method**: Reproduce the core mechanism from "Chain of Continuous Thought"
(Meta, 2024). Coconut replaces discrete CoT tokens with continuous "thought"
embeddings, trained end-to-end with the language model.

**Key differences from Cogito**: Coconut trains the continuous thoughts jointly
with the LM; Cogito trains an external module on frozen hidden states. Coconut
is model-specific; Cogito aims for cross-model transfer.

**Reproduction scope** (simplified, matching our compute budget):
1. Take a base model (start with GPT-2 Medium as reference)
2. Replace CoT tokens with K learnable continuous embedding vectors
3. Train with the "breadth-first" curriculum from the paper:
   - Phase 1: Full CoT supervision
   - Phase 2: Replace first reasoning token with continuous thought
   - Phase 3: Replace more tokens progressively
4. Evaluate on same benchmarks as Cogito

**Simplified reproduction**: We reproduce the mechanism, not the full paper.
We use K=4 continuous thoughts (paper uses variable), single model, GSM8K only
for this baseline. Full Coconut replication is out of scope for v0.1.

---

## 4. Evaluation Metrics

### 4.1 Task Accuracy (Primary)

**Metric**: Exact-match accuracy on the answer portion.

| Benchmark | Answer Format | Evaluation |
|-----------|---------------|------------|
| GSM8K | Numeric | Extract final number, compare to gold |
| ARC-Challenge | Letter (A/B/C/D) | Compare predicted letter to gold |
| StrategyQA | Yes/No | Compare predicted boolean to gold |
| LogiQA | Letter (A/B/C/D) | Compare predicted letter to gold |

**Reporting**: Accuracy ± 95% confidence interval (Wilson score interval).
Report per-benchmark and macro-average across benchmarks.

**Comparison table** (one row per method):

| Method | GSM8K | ARC | StrategyQA | LogiQA | Average |
|--------|-------|-----|------------|--------|---------|
| B0: No CoT | | | | | |
| B1: Full CoT | | | | | |
| B2: Random h | | | | | |
| B3: Linear | | | | | |
| B4: Coconut | | | | | |
| **Cogito (ours)** | | | | | |

### 4.2 Hidden State Prediction Quality

**Metric**: Cosine similarity between predicted `h_end` and true `h_end`.

```
cos_sim(h_pred, h_true) = (h_pred · h_true) / (||h_pred|| × ||h_true||)
```

Report:
- Mean cosine similarity ± std across test examples
- Per-layer cosine similarity (to identify which layers are most predictable)
- Per-benchmark breakdown (to see if some reasoning types transfer better)
- Distribution plot (histogram of cosine similarities)

**Secondary metrics**:
- MSE between predicted and true `h_end` (scale-sensitive complement to cosine)
- R² score of the prediction (variance explained)

### 4.3 Cross-Model Generalization

**Protocol**: Leave-one-model-out evaluation.

1. Train the latent reasoner on hidden states from N-1 models
2. Evaluate on the held-out model
3. Repeat for each model as the held-out

**Key questions**:
- Does training on more architectures improve held-out performance?
- Which model families transfer best to each other?
- Is there a "universal" component that transfers, plus a model-specific residual?

**Reporting**: N×N transfer matrix (train-on-rows, test-on-columns) for both
accuracy and cosine similarity. Highlight diagonal (in-distribution) vs.
off-diagonal (transfer) performance.

**Dimensionality alignment**: Different models have different hidden dimensions.
To enable cross-model training, project all hidden states to a shared dimension
`d_shared` using per-model learned linear projections:
```
h_shared = P_model · h_original    where P_model ∈ R^{d_shared × d_model}
```
`d_shared = 1024` (median hidden dim across our model set).

### 4.4 Compute Efficiency

**Metrics**:
- **Token savings**: Ratio of CoT tokens eliminated vs. full CoT
  `savings = 1 - (latent_compute_flops / full_cot_flops)`
- **Wall-clock speedup**: End-to-end inference time ratio
  `speedup = time_full_cot / time_cogito`
- **Quality-efficiency Pareto**: Plot accuracy vs. compute for all methods
- **Training cost**: GPU-hours to train the latent reasoner

**Measurement**: Run inference 3× per method, report median wall-clock time.
Use `torch.cuda.Event` for GPU timing. Exclude data loading.

---

## 5. Latent Reasoner Architecture (v0.1)

The latent reasoner itself is the primary experimental variable. For v0.1, we
evaluate three architectures (from simplest to most expressive):

### 5.1 MLP Reasoner
```
h_end = MLP(h_start)
      = W3 · GELU(W2 · GELU(W1 · h_start + b1) + b2) + b3
```
- 3 layers, hidden dim = 4 × d_model (following transformer FFN convention)
- ~12M params for d_model=1024
- Fastest to train, tests whether the mapping is a simple function

### 5.2 Recurrent Reasoner
```
for t in range(K):  # K "thinking steps"
    h = GRU(h, h_start)   # h_start as recurring input
h_end = h
```
- K=4 steps (matches Coconut's thought token count)
- GRU cell with hidden dim = d_model
- ~8M params for d_model=1024
- Tests whether iterative refinement helps

### 5.3 Small Transformer Reasoner
```
h_end = TransformerDecoder(
    [h_start, t1, t2, ..., tK],  # h_start + K learnable thought tokens
    num_layers=4, d_model=d_shared, nhead=8
)[-1]  # Take final position as h_end
```
- 4 layers, 8 heads, K=4 thought tokens
- ~25M params for d_model=1024
- Most expressive, tests whether attention over thought positions helps

### 5.4 Training Details (All Architectures)

**Loss**: MSE between predicted and true `h_end`, optionally with cosine
similarity regularization:
```
L = MSE(h_pred, h_true) + λ(1 - cos_sim(h_pred, h_true))
λ = 0.1
```

**Optimizer**: AdamW, lr=3e-4, weight decay=0.01, cosine schedule with warmup
(5% of steps)

**Batch size**: 128 (effective, with gradient accumulation if needed)

**Epochs**: 100 (with early stopping, patience=10, monitoring validation cosine
similarity)

**Data split**: 80% train, 10% validation, 10% test (stratified by benchmark)

**Per-layer vs. best-layer**: Initially train separate reasoners per layer. After
identifying which layer(s) work best (expect middle-to-late layers), focus
compute on those layers.

---

## 6. Feasibility Analysis

### 6.1 Can the First Experiment Run on a Single A100 in <24h?

**Phase 1: Hidden State Capture** (~8 hours)

| Step | Estimate |
|------|----------|
| Load each model (13 models) | ~2 min each, ~30 min total |
| Forward pass per model (17K examples, batch=8, ~200 tokens avg) | ~30 min per small model, ~90 min per 3B model |
| Total inference time | ~6–8 hours |
| Storage I/O | Overlapped with compute |

**Phase 2: Latent Reasoner Training** (~4 hours)

| Step | Estimate |
|------|----------|
| MLP reasoner (single model, single layer) | ~15 min |
| MLP reasoner (all models, best layer) | ~3 hours |
| GRU/Transformer reasoners | ~1 hour each (parallel with MLP) |

**Phase 3: Evaluation** (~2 hours)

| Step | Estimate |
|------|----------|
| Inject h_pred into each model, measure accuracy | ~30 min per model |
| But we evaluate only 3–4 representative models initially | ~2 hours |

**Total estimated: ~14 hours on single A100 (80GB)**

This fits within the 24h budget with margin for debugging and reruns.

### 6.2 Critical Path Optimizations

1. **Start with one model**: Run the full pipeline on GPT-2 Medium first (fast,
   well-understood). This validates the entire pipeline in ~2 hours before
   committing to the full model sweep.

2. **Layer selection**: Capture all layers but initially train the reasoner on
   only layers {L/4, L/2, 3L/4, L} (quartiles). Expand to all layers only if
   results are promising.

3. **Benchmark prioritization**: Start with GSM8K only (clearest CoT, most
   studied). Add other benchmarks in Phase 2 if the core loop works.

4. **Batch inference**: Use `torch.compile()` and flash attention where
   available for the larger models.

### 6.3 Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Models too weak for benchmarks (esp. GPT-2) | Expected. Low baseline accuracy is fine — we measure the *delta* from NoCoT to Cogito, not absolute accuracy |
| CoT traces too long for small context windows | Truncate CoT to 256 tokens; filter examples where CoT < context_length - question_length - 20 |
| Hidden state capture OOM on larger models | Reduce batch size; use gradient checkpointing (even in eval, for activation memory) |
| Cross-model alignment fails | Start with within-model experiments; cross-model is Phase 2 stretch goal |
| Coconut baseline too expensive to reproduce | Simplified reproduction (single model, single benchmark) is acceptable |

---

## 7. Experiment Phasing

### Phase 1: Single-Model Proof of Concept (Day 1)

**Scope**: GPT-2 Medium, GSM8K only, MLP reasoner.

1. Generate/collect CoT traces for GSM8K
2. Capture hidden states from GPT-2 Medium
3. Train MLP reasoner (per-layer)
4. Evaluate: cosine similarity and task accuracy
5. Compare against B0 (No CoT) and B3 (Linear)

**Success criterion**: Cosine similarity > 0.7 at best layer AND measurable
accuracy improvement over No CoT.

### Phase 2: Multi-Model Expansion (Day 2–3)

**Scope**: All 13 models, GSM8K, MLP + GRU reasoners.

1. Capture hidden states across all models
2. Train per-model MLP reasoners, identify best layers
3. Train cross-model reasoner with dimensional alignment
4. Evaluate cross-model transfer matrix

**Success criterion**: Cross-model transfer shows > 50% of within-model
performance for at least 3 model pairs.

### Phase 3: Full Protocol (Day 4–5)

**Scope**: All models, all benchmarks, all baselines, all reasoner architectures.

1. Generate CoT traces for ARC, StrategyQA, LogiQA
2. Run full baseline suite (B0–B4)
3. Train and evaluate all reasoner architectures
4. Compute all evaluation metrics
5. Produce final results tables and analysis

---

## 8. Implementation Checklist

```
[ ] Data pipeline
    [ ] CoT trace collection/generation for all 4 benchmarks
    [ ] Quality filtering pipeline
    [ ] Tokenization with <think>/</think> special tokens
    [ ] Hidden state capture script (batched, all models)
    [ ] HDF5 storage and loading utilities

[ ] Models
    [ ] HuggingFace model loading with tokenizer patching
    [ ] Hidden state extraction hook (all layers)
    [ ] Dimensional alignment projections for cross-model

[ ] Latent Reasoner
    [ ] MLP architecture
    [ ] GRU architecture
    [ ] Small Transformer architecture
    [ ] Training loop with MSE + cosine loss
    [ ] Early stopping and checkpointing

[ ] Baselines
    [ ] B0: No CoT evaluation
    [ ] B1: Full CoT evaluation
    [ ] B2: Random hidden state injection
    [ ] B3: Linear projection training and evaluation
    [ ] B4: Simplified Coconut reproduction

[ ] Evaluation
    [ ] Exact-match accuracy per benchmark
    [ ] Cosine similarity computation
    [ ] Cross-model transfer matrix
    [ ] Compute efficiency measurement
    [ ] Confidence intervals and statistical tests

[ ] Infrastructure
    [ ] GPU memory profiling
    [ ] Experiment tracking (W&B or simple CSV logs)
    [ ] Reproducibility: fixed seeds, config files
```

---

## Appendix A: Key Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| d_shared | 1024 | Median hidden dim across model set |
| K (thought steps) | 4 | Matches Coconut; enough for multi-step reasoning |
| MLP hidden multiplier | 4× | Standard transformer FFN ratio |
| Learning rate | 3e-4 | Standard for Adam on small models |
| Batch size | 128 | Balances GPU utilization and gradient noise |
| Max CoT length | 512 tokens | Covers 95%+ of traces across benchmarks |
| Epochs | 100 | With early stopping (patience=10) |
| λ (cosine loss weight) | 0.1 | Regularization; MSE is primary |
| Train/val/test split | 80/10/10 | Stratified by benchmark |
| Few-shot examples | 5 | Standard for baseline evaluations |

## Appendix B: Expected Outcomes and Decision Points

| Outcome | Interpretation | Next Step |
|---------|---------------|-----------|
| Cosine sim > 0.9, accuracy ≈ Full CoT | Hidden states are highly predictable; latent reasoning works | Scale to larger models, harder tasks |
| Cosine sim 0.5–0.9, accuracy between NoCoT and FullCoT | Partial signal; reasoner captures some but not all reasoning | Investigate which reasoning types transfer; try deeper reasoners |
| Cosine sim < 0.5 | Hidden state mapping is fundamentally hard | Pivot: try different layer selection, different loss functions, or reconsider hypothesis |
| Cross-model transfer works | Universal reasoning structure exists | Major finding; investigate what's shared |
| Cross-model transfer fails | Reasoning structure is model-specific | Still valuable; focus on per-model latent reasoning |
| Linear baseline ≈ Cogito | Mapping is mostly linear | Simplify approach; investigate what the nonlinear component captures |
