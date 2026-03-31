# Cross-Model Representation Alignment

Research survey for the Cogito project. **Core question**: if we train a latent
reasoner (h_start → h_end) on one model (e.g., GPT-2), can representation
alignment make it work on other architectures (Pythia, LLaMA)?

## Executive Summary

**The answer is: yes, with constraints.** Three converging lines of evidence
support cross-model transfer of learned latent modules:

1. **Convergent representations** — Different neural networks trained on similar
   data develop increasingly similar internal representations (Platonic
   Representation Hypothesis). This convergence strengthens with model scale.

2. **Linear alignment suffices** — Representations across models are related by
   approximately linear (often affine or orthogonal) transformations. Simple
   learned maps achieve 90-95% accuracy recovery in stitching experiments.

3. **Features are linearly encoded** — The linear representation hypothesis from
   mechanistic interpretability shows that concepts occupy linear directions in
   hidden space, making cross-model alignment a linear algebra problem.

**Practical implication for Cogito**: A latent reasoner trained in Model A's
space can operate in Model B's space via:
```
h_start_B → [align_B→A] → [reasoner_A] → [align_A→B] → h_end_B
```
where the alignment maps are small affine transforms learned from paired activations.

**Key constraints**: (a) models must be large enough for representational
convergence, (b) shared tokenizer dramatically simplifies alignment, (c) expect
~5-10% performance degradation from stitching.

---

## 1. The Platonic Representation Hypothesis

**Paper**: Huh, Cheung, Wang, Isola (MIT). ICML 2024. arXiv:2405.07987

### Core Claims

Neural network representations are **converging toward a shared statistical
model of reality** — the "platonic representation." This convergence:
- Increases with model scale and capability
- Extends across modalities (vision and language models align)
- Is driven by task pressure, capacity, and simplicity bias

### Key Evidence

- **78 vision models** evaluated on Places-365 using mutual k-NN alignment:
  higher-performing models cluster more tightly in representation space.
- **Cross-modal convergence**: Linear relationship between language-vision
  alignment and language modeling capability (measured as 1 − bits-per-byte).
  More capable LLMs align better with more capable vision models.
- **HELIX** (Gorbett & Jana, Columbia, March 2026): Independent frontier models
  (GPT, Gemini, Qwen, Mistral, Cohere) show CKA similarity **0.595–0.881**. A
  single affine map suffices to align their internal representations.

### Three Driving Forces

| Pressure | Mechanism |
|----------|-----------|
| **Multitask scaling** | Fewer representations satisfy N tasks than M < N tasks |
| **Capacity** | Larger models converge to same global optima |
| **Simplicity bias** | Deep networks prefer simple fits → converge to simplest faithful model |

### Limitations

- Alignment scores reach ~0.16 on mutual k-NN — partial convergence only
- **Aristotelian critique** (Jia et al., 2025, arXiv:2602.14486): After
  null-calibration for network scale, global convergence largely disappears.
  **Local neighborhood similarity** (which points are nearest neighbors) retains
  significance. Representations converge in local relational structure, not
  toward a single global ideal.
- Domain gaps persist (robotics, specialized tasks)
- Sociological bias: human design choices may steer toward human-like
  representations

### Cogito Relevance

The hypothesis directly predicts that GPT-2, Pythia, and LLaMA develop
compatible representations, with alignment improving as models scale. For small
models (GPT-2 117M), convergence may be weaker — the latent reasoner should
target the largest feasible models in each family.

---

## 2. Similarity Metrics

### Comparison Table

| Metric | Invariances | Strengths | Weaknesses |
|--------|-------------|-----------|------------|
| **CKA** (Kornblith 2019) | Orthogonal transforms, isotropic scaling | Highest test-retest reliability; works with different-sized layers | Doesn't produce explicit alignment |
| **CCA/SVCCA** (Raghu 2017) | All invertible linear transforms | Measures subspace overlap | Too permissive; numerically unstable at high dim |
| **PWCCA** (Morcos 2018) | Weighted CCA | Addresses unimportant-direction problem | Still overly invariant |
| **Procrustes** | Orthogonal transforms | Geometrically interpretable; directly gives alignment | Requires same dimensionality |
| **Mutual k-NN** | Topological/geometric | Captures neighborhood structure | Less sensitive to global transforms |

### Recommendations for Cogito

- **Use CKA** as the primary comparison metric when evaluating whether two
  models' representations are compatible
- **Use Procrustes distance** when you need the actual alignment transform
  (directly usable as a stitching layer)
- **Use mutual k-NN** as a complementary topological check
- Always compare against null baselines (Ding et al., 2021) — a similarity
  score is meaningless without context

---

## 3. Model Stitching

### Foundational Work

**Lenc & Vedaldi (2015)**: Take first k layers of Model A + remaining layers
of Model B, train a linear "stitching layer" between them. If performance is
preserved, the representations at layer k are **functionally equivalent** up to
a linear transform.

**Bansal et al. (NeurIPS 2021)**: Extended stitching across architectures:
- ResNet ↔ ViT stitching works surprisingly well
- A **1×1 convolution** or linear layer suffices as a stitching layer
- Monotonic layer correspondence: early→early, late→late
- Performance degrades gracefully — substantial accuracy retained even with
  imperfect stitches
- Stitching is a **stricter** measure than CKA/CCA (tests functional
  equivalence, not just structural similarity)

### Cross-Architecture LLM Stitching

**Transferring Linear Features Across Language Models** (2025, arXiv:2506.06609):
- Affine maps between residual streams enable transfer of sparse autoencoders,
  linear probes, and steering vectors
- Tested on Pythia (70M–160M), GPT-2 (Small–Medium), Gemma-2 (2B–9B)
- Initialization via stitching saves **30-50% FLOPs** for SAE training on
  larger models
- **Critical limitation: all tested models shared the same tokenizer**

### Cogito Relevance

Model stitching is the **exact architecture** needed for latent reasoner
transfer. The reasoner module IS a stitching component — it maps h_start to
h_end. Cross-model deployment adds alignment layers on each side:

```
Model B layer k → [affine align B→A] → [reasoner trained on A] → [affine align A→B] → Model B layer k
```

The stitching literature says linear/affine maps suffice, and the alignment can
be learned from relatively few paired examples.

---

## 4. Alignment Methods

### Procrustes Alignment

Find orthogonal R minimizing ‖X − YR‖_F. Closed-form via SVD:
R = VU^T where UΣV^T = SVD(Y^T X).

- Used extensively in cross-lingual embedding alignment (Conneau et al., 2018)
- Directly applicable: collect activations on shared data, compute alignment
- Fast, closed-form, no gradient descent needed

### Optimal Transport

**Singh et al. (2020)** — Model Fusion via Optimal Transport (OT-Fusion):
- Treats neurons as distributions, finds transport plan mapping neurons across
  models
- Handles many-to-one mappings, respects neuron importance
- Computationally more expensive than Procrustes

### Relative Representations (Zero-Shot Stitching)

**Moschella et al., ICLR 2023 (Oral)**: Represent each point by its similarity
to fixed **anchor points**:
```
relative(x) = [sim(x, a_1), sim(x, a_2), ..., sim(x, a_k)]
```

This creates a coordinate system invariant to random init, isometries, and
rescalings. Enables **zero-shot** model stitching — no retraining needed.

**Maiorca et al. (NeurIPS 2023)**: Translations between latent spaces can be
computed via closed-form algebraic procedures (Procrustes/affine maps),
achieving **90-95% recovery** of native accuracy. Even cross-modal stitching
(text encoder + vision decoder) works.

### Contrastive Alignment

CLIP-style training (Radford et al., 2021) learns shared spaces across
modalities via InfoNCE loss. Could be used to train cross-model alignment
from paired activations, but requires more data than Procrustes.

### Cogito Recommendation

**Start with Procrustes alignment** — it's closed-form, fast, and well-tested.
If Procrustes is insufficient (models too different), try learned affine maps
(small matrix + bias, trained with MSE on paired activations). Reserve OT and
contrastive methods for the hardest cases (cross-tokenizer alignment).

---

## 5. The Linear Representation Hypothesis

### Core Claim

Features and concepts are encoded as **linear directions** in neural network
hidden states. The presence of concept C in input x is captured by ⟨h(x), v_C⟩
for some direction v_C.

### Key Evidence

| Paper | Finding |
|-------|---------|
| Mikolov et al. (2013) | Word2Vec: king − man + woman ≈ queen |
| Elhage et al. (2022, Anthropic) | Superposition: networks pack more features than dimensions using nearly-orthogonal directions |
| Nanda et al. (2023) | Othello-GPT: board state is a linear function of hidden activations |
| Park et al. (2024) | Concepts (sentiment, factuality) = linear subspaces in LLMs; causally validated via intervention |
| Burns et al. (2023) | Truth/falsehood linearly represented; unsupervised probe finds "truth direction" |
| Zou et al. (2023) | Representation Engineering: linear "reading vectors" for honesty, harm; adding them steers behavior |

### Why Linear?

1. **Simplicity bias**: SGD converges to linear encodings (easiest to learn/use)
2. **Linear readout pressure**: Downstream layers read via matrix multiply →
   optimal encoding is linear
3. **Superposition**: For sparse features, nearly-orthogonal directions are
   information-theoretically optimal in a linear readout regime

### Implications for Cross-Model Alignment

If features are linear in both models, alignment reduces to finding a linear map
between feature direction sets. This is why CKA, Procrustes, and linear
stitching layers work. **Superposition complicates things**: different models may
use different superposition structures (different nearly-orthogonal bases), so a
simple rotation may not suffice — the full affine map is needed.

---

## 6. Cross-Architecture Transfer: What Works and What Doesn't

### The Tokenizer Problem

**The single biggest obstacle** for cross-family transfer.

Different tokenizers mean:
- Different sequence lengths for the same input
- No position-wise correspondence of hidden states
- No direct logit-space comparison

**Solutions**:
- **Universal Logit Distillation** (Boizard et al., 2024): Uses optimal
  transport (Wasserstein distance) to map between vocabularies. Achieves parity
  with text-based distillation using half the data.
- **Dual-Space Knowledge Distillation** (EMNLP 2024): Transfers through both
  teacher and student representation spaces.
- **Simplest path**: Work within tokenizer families (GPT-2/Pythia share
  tokenizer; LLaMA family shares tokenizer).

### Neural Incompatibility

**ACL 2025**: Direct parametric transfer between LLMs of different scales is
fundamentally limited. However, **activation-based** (behavioral) alignment
sidesteps this — align how layers *behave*, not how weights *look*.

**SemAlign** (2025): Latent semantic alignment before parameter update predicts
stable cross-scale transfer.

### Transfer Bimodality

Structural features (syntax, position) transfer reliably. Semantic features
either transfer well or fail completely — no graceful degradation. This
bimodality matters for Cogito: the latent reasoner likely captures structural
reasoning patterns (which transfer well) rather than factual knowledge (which
doesn't).

### Feasibility Ranking for Cogito

| Scenario | Feasibility | Evidence |
|----------|-------------|---------|
| Same family, different scale (e.g., GPT-2 Small→Medium) | **High** | 30-50% FLOPs savings via stitching (2025) |
| Same tokenizer, different architecture (e.g., GPT-2→Pythia) | **Moderate-High** | Shared tokenizer eliminates biggest obstacle; CKA 0.6-0.88 |
| Different family AND tokenizer (e.g., GPT-2→LLaMA) | **Moderate** | Requires OT-based tokenizer alignment + representation alignment |
| Different modality (vision→language) | **Low** | Platonic hypothesis says it should work eventually; no demonstrations for latent modules |

---

## 7. Recommended Approach for Cogito

### Phase 1: Within-Family Transfer (Proof of Concept)

1. Train latent reasoner on GPT-2 Small (117M)
2. Collect paired activations on shared data from GPT-2 Small and GPT-2 Medium
3. Compute Procrustes alignment (closed-form, fast)
4. Evaluate: reasoner + alignment maps on GPT-2 Medium
5. Measure: CKA before/after alignment, task accuracy, reasoning quality

### Phase 2: Cross-Family, Same Tokenizer

GPT-2 and Pythia share the same tokenizer (GPT-2 BPE). This eliminates the
tokenizer obstacle while testing cross-architecture transfer.

1. Compute alignment between GPT-2 and Pythia-410M activations
2. Transfer reasoner with alignment wrappers
3. Compare against reasoner trained natively on Pythia

### Phase 3: Cross-Tokenizer Transfer

1. Implement tokenizer alignment via optimal transport (ULD approach)
2. Align GPT-2 family → LLaMA family representations
3. Transfer reasoner with both tokenizer and representation alignment
4. This is the hardest case; expect significant performance degradation

### Measurement Protocol

At each phase, measure:
- **CKA** between source and target representations (before/after alignment)
- **Procrustes distance** to quantify alignment quality
- **Mutual k-NN** for local structure preservation
- **Task accuracy** with stitched reasoner vs. native reasoner
- **Reasoning quality** on specific benchmarks (GSM8K, etc.)

---

## Key References

### Convergent Representations
- Huh et al. (2024). The Platonic Representation Hypothesis. ICML 2024. arXiv:2405.07987
- Jia et al. (2025). Revisiting the Platonic Representation Hypothesis: An Aristotelian View. arXiv:2602.14486
- Gorbett & Jana (2026). HELIX: Secure Linear Alignment of LLMs. arXiv:2603.18908

### Similarity Metrics
- Kornblith et al. (2019). Similarity of Neural Network Representations Revisited. ICML 2019 (CKA)
- Raghu et al. (2017). SVCCA. NeurIPS 2017
- Morcos et al. (2018). Insights on Representational Similarity (PWCCA)
- Ding et al. (2021). Grounding Representation Similarity Through Statistical Testing

### Model Stitching
- Lenc & Vedaldi (2015). Understanding Image Representations by Measuring Equivariance and Equivalence
- Bansal et al. (2021). Revisiting Model Stitching to Compare Neural Representations. NeurIPS 2021
- Transferring Linear Features Across LMs with Model Stitching (2025). arXiv:2506.06609

### Alignment Methods
- Moschella et al. (2023). Relative Representations Enable Zero-Shot Latent Space Communication. ICLR 2023
- Maiorca et al. (2023). Latent Space Translation via Semantic Alignment. NeurIPS 2023
- Singh et al. (2020). Model Fusion via Optimal Transport
- Ainsworth et al. (2023). Git Re-Basin: Merging Models Modulo Permutation Symmetries

### Linear Representation Hypothesis
- Elhage et al. (2022). Toy Models of Superposition. Anthropic
- Park et al. (2024). The Linear Representation Hypothesis and the Geometry of LLMs
- Burns et al. (2023). Discovering Latent Knowledge Without Supervision (CCS)
- Zou et al. (2023). Representation Engineering

### Cross-Architecture Transfer
- Boizard et al. (2024). Universal Logit Distillation. arXiv:2402.12030
- SemAlign (2025). Beyond Neural Incompatibility. arXiv:2510.24208
- Neural Incompatibility (ACL 2025). arXiv:2505.14436
