# Latent Reasoner: Architectural Design & Training Strategy

## 1. Architecture Candidates

### 1.1 Recurrent (GRU/LSTM)
- **Concept**: Reasoning as a temporal sequence of hidden state updates.
- **Pros**: Can handle variable-step "internal reasoning" time; relatively low parameter count.
- **Cons**: Vanishing gradients; sequential nature might be slower than parallel architectures during training.

### 1.2 Small Transformer
- **Concept**: Treat the hidden state transformation as a self-attention process.
- **Pros**: State-of-the-art for sequence modeling; can capture complex dependencies within the latent representation.
- **Cons**: Might be overkill for a single state transformation; fixed-step by default.

### 1.3 Deep Residual MLP
- **Concept**: A deep stack of residual blocks mapping $h_{start} \to h_{end}$.
- **Pros**: Extremely fast inference; easy to implement and optimize.
- **Cons**: Fixed computation depth; no temporal/sequential inductive bias.

### 1.4 Neural ODE
- **Concept**: Model the transition as a continuous-time differential equation $dh/dt = f(h(t), t)$.
- **Pros**: Theoretically sound for continuous state transitions; allows for variable "reasoning depth" at inference by adjusting the ODE solver's time step.
- **Cons**: Training can be computationally expensive and unstable.

### 1.5 Diffusion-style Iterative Refinement
- **Concept**: Treat $h_{start}$ as a noisy version of $h_{end}$ and refine it over several steps.
- **Pros**: Proven success in generative modeling; can produce high-fidelity latent representations.
- **Cons**: Computationally expensive due to iterative nature.

---

## 2. The Projection Problem: Cross-Architecture Generalization

To generalize across models with different dimensions ($d_{GPT2}=768$, $d_{Pythia}=2048$, $d_{LLaMA}=4096$), we propose a **Universal Latent Reasoning Space (ULRS)**.

### 2.1 Universal Latent Reasoning Space (ULRS)
- **Fixed Dimension**: $d_{ULRS} = 1024$ (or similar).
- **Architecture**:
    - **Encoder ($E_m$):** A model-specific linear layer (or small MLP) that maps $h \in \mathbb{R}^{d_m}$ to $z \in \mathbb{R}^{d_{ULRS}}$.
    - **Latent Reasoner ($L$):** Operates entirely within the ULRS: $z_{start} \to z_{end}$.
    - **Decoder ($D_m$):** A model-specific layer that maps $z \in \mathbb{R}^{d_{ULRS}}$ back to $h \in \mathbb{R}^{d_m}$.

### 2.2 Shared Bottleneck
- The Latent Reasoner $L$ is shared across all models.
- Only $E_m$ and $D_m$ are model-specific.
- This forces the reasoner to learn a representation that is model-agnostic.

---

## 3. Training Strategy

### 3.1 Joint Training
- Train on datasets composed of $(h_{start}, h_{end})$ pairs from multiple models simultaneously.
- Loss is the sum of losses across all models in the batch.

### 3.2 Curriculum Learning
1. **Phase 1: Reconstruction**: Train encoders/decoders to perfectly reconstruct hidden states ($h \to z \to h$).
2. **Phase 2: Simple Reasoning**: Train on short CoT sequences (e.g., 1-2 tokens).
3. **Phase 3: Complex Reasoning**: Train on long CoT sequences.

### 3.3 Contrastive Losses
- Use InfoNCE or similar to ensure that $z_{end} = L(z_{start})$ is closer to the true encoded $z_{end}$ than to other states in the batch.
- This helps avoid "state collapse" where the reasoner predicts a mean state.

---

## 4. Training Objectives

### 4.1 MSE on $h_{end}$
- $\mathcal{L} = \| D_m(L(E_m(h_{start}))) - h_{end} \|^2$.
- Simple, but sensitive to outliers.

### 4.2 Cosine Similarity
- Maximize the cosine similarity between predicted and target states.
- Good for capturing the "direction" of reasoning in latent space.

### 4.3 Downstream Task Accuracy (Frozen/E2E)
- Plug $h_{pred\_end}$ back into the original transformer and measure the accuracy of the next-token prediction (the "Answer").
- This is the ultimate metric of success.

---

## 5. Information-Theoretic Analysis

### 5.1 CoT Information Injection
- **Hypothesis**: The "reasoning tokens" $T = [t_1, t_2, \dots, t_n]$ act as a "scratchpad" that injects information into the latent state.
- **Metric**: Mutual Information $I(h_{end}; T | h_{start})$.
- **Compressibility**: Can we represent the same "reasoning delta" $\Delta h = h_{end} - h_{start}$ with fewer bits than the tokens themselves?

### 5.2 Latent Space Rank
- Analyze the effective rank of the $\Delta h$ vectors. If it's low-dimensional, latent reasoning is highly feasible.

---

## 6. Compute Feasibility

### 6.1 Minimum Viable Scale (MVC)
- **Small Models**: Use GPT-2 Small (124M) and Pythia-70M/160M for initial experiments.
- **Hardware**: Training should be possible on a single NVIDIA A100 or even a consumer RTX 3090/4090.
- **Data**: Synthetic reasoning tasks (e.g., GSM8K, MATH) where CoT is explicit.

---

## 7. Research Roadmap

1. **Data Collection**: Extract $(h_{start}, h_{end})$ pairs from GPT-2 and Pythia on GSM8K.
2. **Baseline**: Train a simple Residual MLP reasoner on a single model.
3. **Multi-Model**: Implement ULRS and train on both GPT-2 and Pythia.
4. **Validation**: Test if the reasoner can zero-shot transfer its "reasoning logic" between models (e.g., train on GPT-2, test if it helps Pythia).
