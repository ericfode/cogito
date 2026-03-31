"""Latent reasoner architectures for predicting h_end from h_start.

Key design decisions:
- GRU with proper residual connections (h_t += h_{t-1}, not h_0)
- h_0 serves as persistent query/context, but state evolves iteratively
- Optional ACT (Adaptive Computation Time) for variable-depth reasoning
- LayerNorm after each step for gradient stability
"""
from tinygrad import Tensor
from tinygrad.nn import Linear, LayerNorm

class GRUCell:
  """Minimal GRU cell. Two gates (update, reset) + candidate."""
  def __init__(self, input_dim: int, hidden_dim: int):
    self.W_z = Linear(input_dim + hidden_dim, hidden_dim)
    self.W_r = Linear(input_dim + hidden_dim, hidden_dim)
    self.W_h = Linear(input_dim + hidden_dim, hidden_dim)

  def __call__(self, x: Tensor, h: Tensor) -> Tensor:
    xh = x.cat(h, dim=-1)
    z = self.W_z(xh).sigmoid()
    r = self.W_r(xh).sigmoid()
    xrh = x.cat(r * h, dim=-1)
    h_tilde = self.W_h(xrh).tanh()
    return (1 - z) * h + z * h_tilde

class LoopedResidualGRU:
  """Looped Residual GRU reasoner.

  Architecture:
    h_0 = project_in(h_start)          # project to latent dim
    h = h_0
    for t in 1..N:
        h_new = GRU(h_0, h)            # h_0 as persistent query, h evolves
        h = LayerNorm(h_new + h)       # residual to PREVIOUS state
    h_end = project_out(h)             # project back to model dim

  The GRU weights are shared across all N steps (Universal Transformer style).
  h_0 is fed as input at every step — it's the "question" that the GRU
  iteratively refines an answer to. The residual connection is to h_{t-1},
  so each step builds on the previous one.
  """
  def __init__(self, model_dim: int, latent_dim: int = 512, n_steps: int = 8):
    self.n_steps = n_steps
    self.project_in = Linear(model_dim, latent_dim)
    self.gru = GRUCell(latent_dim, latent_dim)
    self.norm = LayerNorm(latent_dim)
    self.project_out = Linear(latent_dim, model_dim)

  def __call__(self, h_start: Tensor) -> Tensor:
    """Predict h_end from h_start. Shape: (B, D_model) -> (B, D_model)."""
    h_0 = self.project_in(h_start)
    h = h_0
    for _ in range(self.n_steps):
      h_new = self.gru(h_0, h)
      h = self.norm(h_new + h)    # residual to previous state
    return self.project_out(h)

class LoopedResidualGRUWithACT:
  """GRU reasoner with Adaptive Computation Time.

  Learns a per-example halting probability at each step. Harder examples
  get more computation. Based on Graves 2016 (ACT) + Universal Transformers.

  The ponder cost is added to the loss: tau * mean_steps, encouraging
  the model to halt early when it can.
  """
  def __init__(self, model_dim: int, latent_dim: int = 512, max_steps: int = 16,
               halt_threshold: float = 0.99):
    self.max_steps = max_steps
    self.halt_threshold = halt_threshold
    self.project_in = Linear(model_dim, latent_dim)
    self.gru = GRUCell(latent_dim, latent_dim)
    self.norm = LayerNorm(latent_dim)
    self.halt_linear = Linear(latent_dim, 1)  # predicts halting probability
    self.project_out = Linear(latent_dim, model_dim)

  def __call__(self, h_start: Tensor) -> tuple[Tensor, Tensor]:
    """Returns (h_pred, ponder_cost).

    ponder_cost: mean number of steps used (differentiable via remainders).
    Add tau * ponder_cost to the main loss.
    """
    B = h_start.shape[0]
    h_0 = self.project_in(h_start)
    h = h_0

    # ACT accumulators
    cum_halt = Tensor.zeros(B, 1)        # cumulative halt probability
    remainders = Tensor.zeros(B, 1)      # remainder for halted examples
    n_updates = Tensor.zeros(B, 1)       # number of updates per example
    h_acc = Tensor.zeros_like(h)         # accumulated weighted hidden state

    for t in range(self.max_steps):
      h_new = self.gru(h_0, h)
      h = self.norm(h_new + h)

      # Halting probability for this step
      p = self.halt_linear(h).sigmoid()  # (B, 1)

      # Which examples are still running?
      still_running = (cum_halt < 1.0).float()

      # Which would halt this step?
      new_halted = ((cum_halt + p) >= self.halt_threshold).float() * still_running

      # For newly halted: remainder = 1 - cum_halt
      # For still running: add p to cumulative
      still_running_after = still_running * (1.0 - new_halted)
      p_used = new_halted * (1.0 - cum_halt) + still_running_after * p

      cum_halt = cum_halt + p_used
      n_updates = n_updates + still_running
      h_acc = h_acc + p_used * h

    # Final output: weighted sum of states
    return self.project_out(h_acc), n_updates.mean()


class MultiScaleProjection:
  """Captures hidden states from multiple layers and concatenates them.

  Instead of tapping a single layer at 67% depth, we tap at
  50%, 67%, and 80% and concatenate. This gives the reasoner access to
  multiple abstraction levels.
  """
  def __init__(self, model_dim: int, n_layers: int):
    self.layers = [
      int(n_layers * 0.50),  # mid-level features
      int(n_layers * 0.67),  # sweet spot
      int(n_layers * 0.80),  # near-final semantic
    ]
    # Project concatenated multi-scale features down to model_dim
    self.proj = Linear(model_dim * len(self.layers), model_dim)

  def __call__(self, hidden_states: list[Tensor]) -> Tensor:
    """hidden_states: list of (B, T, D) from each captured layer."""
    concat = hidden_states[0].cat(*hidden_states[1:], dim=-1)
    return self.proj(concat)


class LinearBaseline:
  """Baseline: h_end = W @ h_start + b."""
  def __init__(self, model_dim: int):
    self.proj = Linear(model_dim, model_dim)

  def __call__(self, h_start: Tensor) -> Tensor:
    return self.proj(h_start)
