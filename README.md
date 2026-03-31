# Cogito

**Non-linguistic reasoning in neural network latent space.**

Can a transformer reason without words? This project investigates whether a
"latent reasoner" module can learn to perform chain-of-thought-equivalent
computation in a neural network's internal representation space — and whether
that reasoning capability generalizes across architectures.

## Core Hypothesis

Given a transformer processing `input <think> ... reasoning tokens ... </think> answer`:
1. Capture hidden state at `<think>` (h_start) and `</think>` (h_end)
2. Train a latent reasoner: h_start → h_end in zero tokens
3. Train across multiple architectures to find universal structure
4. At inference, replace token-level CoT with latent computation

## Research Questions

1. **Which layer?** Where in a transformer is the representation most amenable to latent reasoning?
2. **What architecture?** What should the latent reasoner look like (recurrent, attention, MLP)?
3. **Does it generalize?** Can a reasoner trained on multiple models transfer to unseen architectures?
4. **What is the representation?** Is there a universal "reasoning structure" in latent space?

## Prior Art

- Coconut (Meta, 2024) — Chain of Continuous Thought
- Quiet-STaR — Self-Taught Reasoner
- Pause Tokens — learning to pause before answering
- JEPA (LeCun) — Joint Embedding Predictive Architecture
- Platonic Representation Hypothesis — convergent representations across models
