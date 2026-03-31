# Research: Optimal Layer Selection for Latent Reasoning

> **Bead:** co-1zj | **Date:** 2026-03-31 | **Status:** Complete
>
> This determines WHERE we attach the latent reasoner in the Cogito architecture.

## Executive Summary

**Recommendation: Extract representations at ~60–75% of model depth (the "two-thirds
heuristic") from the full residual stream (post-layer, post-MLP).** This is where
representations are maximally compressed while retaining task-relevant reasoning
information, before the final layers specialize for next-token prediction.

For a 32-layer model (e.g., LLaMA-7B): **layers 20–24**.
For a 12-layer model (e.g., GPT-2 small): **layers 7–9**.

The optimal layer scales approximately linearly with depth across architectures.

For Cogito specifically, we should consider:
1. **Static selection** at ~2/3 depth as the default
2. **Tuned lens convergence** as a principled measurement tool
3. **Multi-layer extraction** (tapping 2–3 layers in the 60–80% range) for richer signal
4. **Input-dependent selection** as a future optimization

---

## Table of Contents

1. [Information Compression by Layer](#1-information-compression-by-layer)
2. [Middle vs Late Layers](#2-middle-vs-late-layers)
3. [Residual Stream vs Attention vs MLP](#3-residual-stream-vs-attention-vs-mlp)
4. [Layer Selection Across Architectures](#4-layer-selection-across-architectures)
5. [Practical Measurement Methods](#5-practical-measurement-methods)
6. [Dynamic / Input-Dependent Layer Selection](#6-dynamic--input-dependent-layer-selection)
7. [Prior Art: Layer Choices in Related Systems](#7-prior-art-layer-choices-in-related-systems)
8. [Recommendations for Cogito](#8-recommendations-for-cogito)
9. [References](#9-references)

---

## 1. Information Compression by Layer

### 1.1 Probing Classifiers

Probing studies train lightweight classifiers on frozen layer representations to measure
what information each layer encodes.

**Key findings:**

| Layer Region | % Depth | What's Encoded | Sources |
|---|---|---|---|
| Early (0–20%) | Layers 0–2 | Surface features, POS, tokenization | Tenney 2019, Jawahar 2019 |
| Lower-middle (20–50%) | Layers 3–6 | Syntactic structure, dependency trees | Hewitt & Manning 2019 |
| **Upper-middle (50–80%)** | **Layers 6–9** | **Semantic composition, reasoning, transfer** | **Liu 2019, Tenney 2019** |
| Final (80–100%) | Layers 10–12 | Output formatting, pretraining objective | All probing studies |

**Critical insight:** The best layer for transfer is generally NOT the last layer (Liu et al.
2019). Final layers degrade on probing tasks because they specialize for the pretraining
objective.

Hewitt & Manning (2019) showed syntax trees are linearly encoded in BERT's representation
space, with strongest encoding at layers 6–8 of BERT-base-12.

### 1.2 CKA and SVCCA

Representation similarity analyses reveal a characteristic pattern across transformer layers:

- **Early layers** are similar to each other (slow change)
- **Middle layers** show maximum divergence — the most transformation happens here
- **Upper layers** converge again (settling on output representation)

This creates a **sigmoid/hourglass pattern** in CKA similarity matrices (Kornblith et al.
2019). The middle layers are where representations undergo the most radical transformation.

Wu et al. (2020) found that **middle layers across different architectures are more similar
to each other** than early or late layers — evidence for a universal representation "sweet
spot."

SVCCA analysis (Raghu et al. 2017) shows intrinsic dimensionality peaks in middle layers,
then decreases as the network compresses into task-relevant subspaces.

### 1.3 Information Bottleneck Theory

The Information Bottleneck framework (Shwartz-Ziv & Tishby 2017) predicts that optimal
representations maximize I(T;Y) (mutual information with the target) while minimizing
I(T;X) (mutual information with the input) — i.e., maximum compression of irrelevant input
information.

Applied to transformers:

- **Lower layers** preserve token identity (high I(T;X))
- **Upper layers** compress token identity, amplify task-relevant features (Voita et al. 2019)
- The **compression funnel peaks at ~60–75% depth** — where irrelevant information is
  maximally discarded while task-relevant signal is retained

This is consistent with probing and CKA findings: the upper-middle layers are the
information bottleneck.

### 1.4 Residual Stream Perspective

The residual stream interpretation (Elhage et al. 2021, Anthropic) views the transformer
as a communication channel:

- Each attention head and MLP block reads from and writes to the residual stream additively
- Information **accumulates** across layers — no single layer contains the full answer
- The residual stream at the model's midpoint contains the richest accumulated state:
  syntactic and semantic features are present but output specialization hasn't narrowed
  the representation

Key mechanistic findings:
- **Induction heads** (in-context learning) emerge in early-to-middle layers (Olsson et al. 2022)
- **MLP layers** progressively promote specific vocabulary items: broad semantic categories
  early → specific tokens late (Geva et al. 2022)
- **Automated circuit discovery** (Conmy et al. 2023) shows critical reasoning computation
  is distributed across attention heads in middle-to-upper layers

---

## 2. Middle vs Late Layers

### 2.1 Evidence for Middle Layer Superiority

The evidence for middle layers being most transferable is extensive:

- **Fine-tuning dynamics** (Merchant et al. 2020): Lower and middle layers change least
  during fine-tuning → they encode task-general features. Upper layers change most → they
  specialize.
- **Linguistic regions** (Zhang et al. 2024): Middle layers form a "linguistic region"
  responsible for core language understanding.
- **Truthfulness probing** (Azaria & Mitchell 2023): Optimal probing for truthfulness at
  ~60–70% depth in LLaMA models.
- **Inference-Time Intervention** (Li et al. 2024): Interventions at 60–70% depth most
  effective for steering truthfulness.
- **CCS** (Burns et al. 2023): Contrast-consistent search for latent knowledge works best
  at 60–75% depth.

### 2.2 Late Layer Specialization

Late layers serve a fundamentally different function:

- **Logit lens** (nostalgebraist 2020): Only the last few layers produce confident
  next-token predictions when projected through the unembedding matrix
- **MLP as key-value memory** (Geva et al. 2022): Late MLP layers promote specific tokens,
  collapsing rich representations into predictions
- **Robustness** (Lad et al. 2024): Early/middle layers can be quantized or perturbed with
  modest loss; corrupting the last few layers is catastrophic

### 2.3 The "Two-Thirds Depth" Heuristic

No single paper proposes this as a universal rule, but converging evidence supports it:

- Probing accuracy peaks at 58–75% depth across models (Liu 2019)
- CKA output similarity peaks at ~2/3 depth (Raghu et al. 2021)
- Truthfulness/knowledge probes optimal at 60–70% depth (Azaria & Mitchell 2023, Li et al. 2024)
- Tuned lens convergence at ~60–70% depth across Pythia scales (Belrose et al. 2023)

**The heuristic scales linearly with model depth across architectures.**

### 2.4 The Pruning Paradox

Layer pruning studies reveal an apparent paradox:

- **ShortGPT** (Men et al. 2024): Middle layers have the lowest Block Influence scores
  individually (any one can be removed)
- **Gromov et al. (2024)**: Up to 50% of later layers can be pruned with minimal perplexity loss

**Resolution:** Middle layers are **individually redundant** but **collectively essential**.
They perform distributed, iterative refinement. Any single middle layer can be removed, but
removing many degrades performance catastrophically. This is actually favorable for Cogito:
it means the representation at this depth is robust — tapping a specific layer isn't
fragile.

---

## 3. Residual Stream vs Attention vs MLP

### Where to Tap Within a Layer

| Tap Point | Best For | Character |
|---|---|---|
| **Full residual stream (post-layer)** | **General-purpose extraction** | Most stable; accumulates all information |
| Post-attention (before MLP) | Syntactic/positional information | Token mixing done, knowledge not yet injected |
| Post-MLP (before next attention) | Knowledge-dependent tasks | Factual knowledge injected |
| Individual attention heads | Interpretability, specific features | Sparse, specialized |
| MLP intermediate activations | Feature extraction, SAE analysis | High-dimensional, sparse |

### Depth-Dependent Roles

- **Early attention (layers 0–3):** Basic contextualization, nearby token interactions
- **Middle attention (layers 4–12 of 32):** Syntactic specialization — dependency parsing, agreement
- **Late attention (layers 12–24):** Longer-range semantic dependencies
- **Early MLP:** Token embedding enrichment
- **Middle MLP:** Semantic feature encoding
- **Late MLP:** Factual knowledge retrieval and output token promotion

### Recommendation for Cogito

**Tap the full residual stream (post-layer, after both attention and MLP).** This gives the
most complete, stable representation. The residual stream is a sum of all prior
contributions and is the natural "working memory" of the transformer.

---

## 4. Layer Selection Across Architectures

### Optimal Layers by Model Family

| Architecture | Total Layers | Optimal Range (60–75%) | Evidence |
|---|---|---|---|
| GPT-2 Small | 12 | 7–9 | Logit lens, probing |
| GPT-2 Medium | 24 | 15–18 | Logit lens |
| GPT-2 Large | 36 | 22–27 | Logit lens |
| GPT-2 XL | 48 | 30–36 | Logit lens |
| LLaMA-7B | 32 | 20–24 | Azaria & Mitchell 2023 |
| LLaMA-13B | 40 | 25–30 | Scaled heuristic |
| LLaMA-33B | 60 | 38–45 | Scaled heuristic |
| LLaMA-65B/70B | 80 | 50–60 | Li et al. 2024 ITI |
| Pythia (all scales) | 6–36 | 60–70% of depth | Belrose et al. 2023 |
| Phi-1/2/3 | 24–32 | 16–24 | Architectural similarity |

**Key finding: The optimal layer scales linearly with depth.** The proportional depth
(~60–75%) remains roughly constant across model families and scales. This means Cogito
can use a simple formula: `optimal_layer = int(n_layers * 0.67)` as a default, regardless
of the target architecture.

---

## 5. Practical Measurement Methods

### 5.1 Tuned Lens (Recommended Primary Method)

The tuned lens (Belrose et al. 2023) trains a per-layer affine transformation to project
intermediate representations into vocabulary space. It reveals where models converge to
their final predictions.

**Why it's best for Cogito:**
- Gives per-input convergence information (not just static averages)
- More accurate than raw logit lens (handles representation subspace differences)
- The convergence point naturally identifies where "reasoning is done"
- Easy to implement: one affine transform per layer, trained on a small calibration set

**Practical use:** The layer where tuned lens prediction matches the final output is where
the model has "finished thinking" — later layers are formatting, not reasoning.

### 5.2 Probing Accuracy Curves

Train linear probes at each layer for the target task. The saturation point identifies
minimum useful depth. Different tasks peak at different layers.

### 5.3 Layer Ablation / Block Influence

Block Influence (BI) from ShortGPT: measure cosine similarity between layer input and
output. High similarity = low influence = redundant layer. The inverse gives importance.

### 5.4 Gradient-Based Importance

Compute gradient of loss w.r.t. layer output. Unlike Fisher information (static), this
gives per-input importance. Most useful for dynamic layer selection.

### 5.5 CKA Phase Transition Detection

Plot CKA similarity between adjacent layers. Sharp drops mark phase transitions — these
are the layers introducing genuinely new information. Useful for identifying the boundary
between "reasoning" and "output formatting" phases.

---

## 6. Dynamic / Input-Dependent Layer Selection

### 6.1 Early Exit Methods

| Method | Mechanism | Speedup | Best For |
|---|---|---|---|
| DeeBERT (Xin 2020) | Entropy threshold | ~40% FLOPs | Classification |
| PABEE (Zhou 2020) | Patience (k consecutive agreements) | ~40% FLOPs | Classification |
| SkipDecode (Del Corro 2023) | Token-level layer skipping | 2–5x | Batched generation |
| CALM (Schuster 2022) | Consistency-based confidence | ~3x | Generation with guarantees |
| **LayerSkip (Meta 2024)** | **Progressive dropout + self-speculative** | **~2x** | **Most practical for LLMs** |

LayerSkip is the most deployment-ready: it trains with progressive layer dropout, adds
early-exit loss at every layer, and uses self-speculative decoding at inference (early
layers draft, full model verifies).

### 6.2 Mixture-of-Depths (MoD)

Raposo et al. (2024): Per-token, per-layer routing. A learned router decides which tokens
get full computation at each layer. Tokens that skip a layer pass through unchanged via
the residual stream.

Key properties:
- Function words and predictable tokens tend to be routed around (skipped)
- Content words and surprising tokens get full computation
- Matches standard transformer performance at 50% fewer FLOPs
- Requires training from scratch

### 6.3 Adaptive Computation Time / Universal Transformers

- ACT (Graves 2016): Halting units accumulate probability; computation stops when cumulative
  probability exceeds 1. Per-input depth emerges naturally.
- Universal Transformers (Dehghani et al. 2019): Weight-shared layers + ACT. Turing-complete
  but limited capacity vs distinct layers.

### 6.4 Implications for Cogito

Dynamic layer selection is a future optimization. For the initial Cogito design:
- **Start with static selection** at ~2/3 depth
- **Measure with tuned lens** to validate the choice per-model
- **Consider multi-layer tapping** (e.g., layers at 50%, 67%, 80%) to capture information
  at multiple abstraction levels
- **Future work:** Input-dependent selection using tuned lens convergence as a signal

---

## 7. Prior Art: Layer Choices in Related Systems

### 7.1 Coconut (Meta, 2024)

Hao et al. "Training Large Language Models to Reason in a Continuous Latent Space"

- Uses the **last hidden state** (final layer output, before LM head)
- Feeds it back as input embedding for the next reasoning step
- Creates a recurrent loop through the full transformer depth
- **Rationale:** Simplicity; end-to-end training adapts what the last layer produces

**Note:** This is at odds with probing literature. Coconut works because the model is
trained end-to-end to produce useful "thought vectors" from the last layer, effectively
repurposing it from "predict next token" to "produce reasoning state." This requires
retraining. For Cogito's approach of training a latent reasoner h_start → h_end without
modifying the base model, middle layers are more appropriate.

### 7.2 Quiet-STaR (Stanford, 2024)

Zelikman et al. "Language Models Can Teach Themselves to Think Before Speaking"

- Generates internal rationale tokens through the full stack
- Mixing at the last layer between post-rationale and original hidden states
- No layer-specific selection; benefits are largest on reasoning tasks

### 7.3 Pause Tokens (Google DeepMind, 2024)

Goyal et al. "Think before you speak: Training Language Models With Pause Tokens"

- Benefit increases with model depth — more layers to propagate through
- Extra computation is distributed across all layers
- Middle layers benefit most from the extra computation time
- Diminishing returns beyond 1–10 pause tokens

### 7.4 Key Difference for Cogito

Coconut, Quiet-STaR, and Pause Tokens all modify the base model's forward pass. Cogito's
approach of training an external latent reasoner on frozen representations is fundamentally
different. We need representations that are **naturally informative** without retraining the
base model — which points clearly to the upper-middle layers (60–75% depth) where
representations are richest.

---

## 8. Recommendations for Cogito

### 8.1 Primary Recommendation

**Extract from the full residual stream at ~67% depth (2/3 of total layers).**

| Model | Formula | Specific Layer |
|---|---|---|
| Any N-layer model | `layer = int(N * 0.67)` | — |
| GPT-2 Small (12L) | `int(12 * 0.67)` | Layer 8 |
| LLaMA-7B (32L) | `int(32 * 0.67)` | Layer 21 |
| Pythia-410M (24L) | `int(24 * 0.67)` | Layer 16 |
| Phi-2 (32L) | `int(32 * 0.67)` | Layer 21 |

### 8.2 Validation Protocol

Before committing to a layer for a given model:

1. **Tuned lens analysis**: Train per-layer affine transforms, measure convergence per
   input. The convergence point distribution should peak near the chosen layer.
2. **Probing accuracy curve**: Train linear probes at each layer for a reasoning task
   (e.g., GSM8K-derived labels). The curve should plateau at or before the chosen layer.
3. **Layer ablation**: Verify that removing the chosen layer has higher impact than removing
   adjacent layers.

### 8.3 Multi-Layer Extraction (Recommended Enhancement)

Rather than a single layer, consider extracting from **2–3 layers** in the 50–80% range:

```
h_start = concat(residual[layer_50%], residual[layer_67%], residual[layer_80%])
```

This captures information at multiple abstraction levels:
- 50% depth: Syntactic structure, compositional semantics
- 67% depth: Full semantic composition, reasoning features (primary)
- 80% depth: Near-output but pre-specialization

The latent reasoner can learn to weight these appropriately.

### 8.4 What NOT to Do

- **Don't use the last layer** (unless retraining the base model like Coconut)
- **Don't use the first few layers** (too shallow, only surface features)
- **Don't tap attention outputs alone** (miss MLP knowledge injection)
- **Don't assume one layer works for all models** (validate with tuned lens)

### 8.5 Future Work: Dynamic Layer Selection

Once the basic system works:
1. **Per-input layer selection** using tuned lens convergence as the signal
2. **Learned routing** (MoD-style) within the latent reasoner
3. **Multi-scale extraction** with attention over layers

---

## 9. References

### Probing and Representation Analysis
- Tenney et al. (2019). "BERT Rediscovers the Classical NLP Pipeline." ACL.
- Jawahar et al. (2019). "What Does BERT Learn about the Structure of Language?"
- Liu et al. (2019). "Linguistic Knowledge and Transferability of Contextual Representations." NAACL.
- Hewitt & Manning (2019). "A Structural Probe for Finding Syntax in Word Representations." NAACL.
- Wu et al. (2020). "Similarity Analysis of Contextual Word Representation Models."
- Merchant et al. (2020). "What Happens To BERT Embeddings During Fine-tuning?" ACL.
- Hernandez et al. (2021). "Low-Complexity Probing via Finding Subnetworks."
- Azaria & Mitchell (2023). "The Internal State of an LLM Knows When It's Lying." EMNLP.
- Burns et al. (2023). "Discovering Latent Knowledge in Language Models Without Supervision." ICLR.
- Zhang et al. (2024). "Unveiling Linguistic Regions in Large Language Models."

### Representation Similarity
- Raghu et al. (2017). "SVCCA: Singular Vector Canonical Correlation Analysis." NeurIPS.
- Kornblith et al. (2019). "Similarity of Neural Network Representations Revisited." ICML.
- Raghu et al. (2021). "Do Vision Transformers See Like Convolutional Neural Networks?" NeurIPS.

### Information Theory
- Shwartz-Ziv & Tishby (2017). "Opening the Black Box of Deep Neural Networks via Information."
- Voita et al. (2019). "The Bottom-up Evolution of Representations in the Transformer."
- Guan et al. (2024). "Dissecting the Interplay of Attention Paths in a Statistical Mechanics Theory of Transformers."

### Mechanistic Interpretability
- Elhage et al. (2021). "A Mathematical Framework for Transformer Circuits." Anthropic.
- Geva et al. (2021). "Transformer Feed-Forward Layers Are Key-Value Memories." EMNLP.
- Geva et al. (2022). "Transformer FFN Layers Build Predictions by Promoting Concepts." EMNLP.
- Olsson et al. (2022). "In-context Learning and Induction Heads." Anthropic.
- Conmy et al. (2023). "Towards Automated Circuit Discovery for Mechanistic Interpretability."
- Dar et al. (2023). "Analyzing Transformers in Embedding Space."

### Layer Pruning
- Men et al. (2024). "ShortGPT: Layers in Large Language Models are More Redundant Than You Think."
- Gromov et al. (2024). "The Unreasonable Ineffectiveness of the Deeper Layers."
- Lad et al. (2024). "The Remarkable Robustness of LLMs."

### Lens Methods
- nostalgebraist (2020). "Interpreting GPT: the logit lens." Blog post.
- Belrose et al. (2023). "Eliciting Latent Predictions from Transformers with the Tuned Lens." NeurIPS.

### Adaptive Computation
- Graves (2016). "Adaptive Computation Time for Recurrent Neural Networks."
- Dehghani et al. (2019). "Universal Transformers." ICLR.
- Schwartz et al. (2020). "The Right Tool for the Job: Matching Model and Instance Complexities." ACL.
- Xin et al. (2020). "DeeBERT: Dynamic Early Exiting for Accelerating BERT Inference." ACL.
- Zhou et al. (2020). "PABEE: Pre-trained Model-based Early Exiting." ACL.
- Fan et al. (2020). "Reducing Transformer Depth on Demand with Structured Dropout." ICLR.
- Schuster et al. (2022). "Confident Adaptive Language Modeling." NeurIPS.
- Del Corro et al. (2023). "SkipDecode: Autoregressive Skip Decoding with Batched Verification."
- Raposo et al. (2024). "Mixture-of-Depths: Dynamically Allocating Compute." arXiv.
- Elhoushi et al. (2024). "LayerSkip: Enabling Early Exit and Self-Speculative Decoding." Meta.

### Latent Reasoning
- Hao et al. (2024). "Training LLMs to Reason in a Continuous Latent Space." (Coconut, Meta FAIR).
- Zelikman et al. (2024). "Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking."
- Goyal et al. (2024). "Think before you speak: Training Language Models With Pause Tokens." DeepMind.

### Truthfulness and Intervention
- Li et al. (2024). "Inference-Time Intervention: Eliciting Truthful Answers." NeurIPS.
- Michel et al. (2019). "Are Sixteen Heads Really Better than One?" NeurIPS.
