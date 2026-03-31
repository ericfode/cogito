"""Training loop for the latent reasoner.

Phase 1: Single model (LLaMA 3.2 1B), GSM8K, Looped Residual GRU.
Tracked with Weave for experiment observability.
"""
from tinygrad import Tensor, dtypes
from tinygrad.nn.optim import AdamW
from tinygrad.nn.state import get_parameters
import numpy as np
import weave
import os, json, time

from cogito.reasoner import LoopedResidualGRU, LinearBaseline

def cosine_similarity(a: Tensor, b: Tensor) -> Tensor:
  """Cosine similarity between batched vectors."""
  a_norm = a / (a.pow(2).sum(-1, keepdim=True).sqrt() + 1e-8)
  b_norm = b / (b.pow(2).sum(-1, keepdim=True).sqrt() + 1e-8)
  return (a_norm * b_norm).sum(-1)

def loss_fn(h_pred: Tensor, h_true: Tensor, cos_weight: float = 0.1) -> Tensor:
  """MSE + cosine similarity loss.

  From experimental protocol: L = MSE(h_pred, h_true) + λ(1 - cos_sim(h_pred, h_true))
  """
  mse = (h_pred - h_true).pow(2).mean()
  cos_loss = (1.0 - cosine_similarity(h_pred, h_true)).mean()
  return mse + cos_weight * cos_loss

@weave.op()
def train_epoch(reasoner, opt, params, h_starts, h_ends, train_idx, batch_size, cos_weight):
  """Single training epoch."""
  Tensor.training = True
  np.random.shuffle(train_idx)
  train_losses = []

  for i in range(0, len(train_idx), batch_size):
    batch_idx = train_idx[i:i + batch_size]
    h_start = Tensor(h_starts[batch_idx].astype(np.float32))
    h_end = Tensor(h_ends[batch_idx].astype(np.float32))

    h_pred = reasoner(h_start)
    loss = loss_fn(h_pred, h_end, cos_weight)

    opt.zero_grad()
    loss.backward()
    # gradient clipping
    for p in params:
      if p.grad is not None:
        grad_norm = p.grad.pow(2).sum().sqrt()
        scale = Tensor.where(grad_norm > 1.0, 1.0 / grad_norm, Tensor(1.0))
        p.grad = p.grad * scale
    opt.step()

    train_losses.append(loss.numpy().item())

  Tensor.training = False
  return float(np.mean(train_losses))

@weave.op()
def validate(reasoner, h_starts, h_ends, val_idx, batch_size, cos_weight):
  """Validation pass."""
  val_losses, val_cos, val_mse = [], [], []
  for i in range(0, len(val_idx), batch_size):
    batch_idx = val_idx[i:i + batch_size]
    h_start = Tensor(h_starts[batch_idx].astype(np.float32))
    h_end = Tensor(h_ends[batch_idx].astype(np.float32))

    h_pred = reasoner(h_start)
    vl = loss_fn(h_pred, h_end, cos_weight)
    cs = cosine_similarity(h_pred, h_end).mean()
    mse = (h_pred - h_end).pow(2).mean()
    val_losses.append(vl.numpy().item())
    val_cos.append(cs.numpy().item())
    val_mse.append(mse.numpy().item())

  return {
    "loss": float(np.mean(val_losses)),
    "cosine_similarity": float(np.mean(val_cos)),
    "mse": float(np.mean(val_mse)),
  }

@weave.op()
def train_reasoner(
    h_starts: np.ndarray,  # (N, dim)
    h_ends: np.ndarray,    # (N, dim)
    model_dim: int,
    latent_dim: int = 512,
    n_steps: int = 8,
    batch_size: int = 128,
    epochs: int = 100,
    lr: float = 3e-4,
    patience: int = 10,
    cos_weight: float = 0.1,
    save_dir: str = "checkpoints",
):
  """Train the Looped Residual GRU reasoner on captured hidden states."""
  N = h_starts.shape[0]
  assert h_starts.shape == h_ends.shape == (N, model_dim)

  # Train/val/test split: 80/10/10
  perm = np.random.permutation(N)
  n_train = int(0.8 * N)
  n_val = int(0.1 * N)
  train_idx = perm[:n_train]
  val_idx = perm[n_train:n_train + n_val]
  test_idx = perm[n_train + n_val:]

  print(f"Split: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")

  reasoner = LoopedResidualGRU(model_dim, latent_dim, n_steps)
  params = get_parameters(reasoner)
  n_params = sum(p.numel() for p in params)
  print(f"Reasoner params: {n_params:,}")

  opt = AdamW(params, lr=lr, weight_decay=0.01)

  best_val_loss = float("inf")
  patience_counter = 0
  os.makedirs(save_dir, exist_ok=True)

  history = []

  for epoch in range(epochs):
    t0 = time.time()

    avg_train = train_epoch(reasoner, opt, params, h_starts, h_ends, train_idx, batch_size, cos_weight)
    val_metrics = validate(reasoner, h_starts, h_ends, val_idx, batch_size, cos_weight)

    epoch_time = time.time() - t0
    avg_val = val_metrics["loss"]
    avg_cos = val_metrics["cosine_similarity"]
    avg_mse = val_metrics["mse"]

    print(f"Epoch {epoch+1:3d} | train_loss={avg_train:.4f} val_loss={avg_val:.4f} "
          f"cos_sim={avg_cos:.4f} mse={avg_mse:.4f} [{epoch_time:.1f}s]")

    history.append({
      "epoch": epoch + 1, "train_loss": avg_train,
      "val_loss": avg_val, "val_cos_sim": avg_cos, "val_mse": avg_mse,
      "epoch_time_s": epoch_time,
    })

    # Early stopping
    if avg_val < best_val_loss:
      best_val_loss = avg_val
      patience_counter = 0
      state = {k: v.numpy() for k, v in get_state_dict_from_obj(reasoner).items()}
      np.savez(f"{save_dir}/best_reasoner.npz", **state)
    else:
      patience_counter += 1
      if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

  # --- Test ---
  print("\n--- Test Results ---")
  test_metrics = validate(reasoner, h_starts, h_ends, test_idx, batch_size, cos_weight)
  print(f"Test loss: {test_metrics['loss']:.4f}")
  print(f"Test cos_sim: {test_metrics['cosine_similarity']:.4f}")
  print(f"Test MSE: {test_metrics['mse']:.4f}")

  return {
    "history": history,
    "test": test_metrics,
    "best_val_loss": best_val_loss,
    "n_params": n_params,
    "stopped_epoch": len(history),
  }


def get_state_dict_from_obj(obj, prefix="") -> dict[str, Tensor]:
  """Extract state dict from an arbitrary object tree."""
  result = {}
  for k, v in obj.__dict__.items():
    name = f"{prefix}{k}" if prefix else k
    if isinstance(v, Tensor):
      result[name] = v
    elif hasattr(v, "__dict__") and not callable(v):
      result.update(get_state_dict_from_obj(v, f"{name}."))
    elif isinstance(v, list):
      for i, item in enumerate(v):
        if hasattr(item, "__dict__"):
          result.update(get_state_dict_from_obj(item, f"{name}.{i}."))
  return result
