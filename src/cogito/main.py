"""Cogito v0.1 — Latent Reasoning Experiment.

Usage:
  python -m cogito.main capture   # Capture hidden states from LLaMA
  python -m cogito.main train     # Train latent reasoner on captured states
  python -m cogito.main eval      # Evaluate reasoner + baselines
  python -m cogito.main all       # Run full pipeline
"""
import argparse, os, sys
import numpy as np
from tinygrad import Tensor, getenv
import weave

MODEL_URL = "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q6_K.gguf"
DATA_DIR = "data/hidden_states/llama3.2-1b"
CHECKPOINT_DIR = "checkpoints"

def load_model():
  """Load LLaMA 3.2 1B from GGUF."""
  from tinygrad.apps.llm import Transformer, SimpleTokenizer
  print("Loading LLaMA 3.2 1B...")
  model, kv = Transformer.from_gguf(Tensor.from_url(MODEL_URL), max_context=512, realize=True)
  tok = SimpleTokenizer.from_gguf_kv(kv)
  print(f"Loaded: {len(model.blk)} layers, dim={model.blk[0].attn_q.weight.shape[1]}")
  return model, tok

@weave.op()
def cmd_capture(args):
  """Capture hidden states from GSM8K through LLaMA."""
  from cogito.data import load_gsm8k, capture_dataset
  model, tok = load_model()
  examples = load_gsm8k("train", max_examples=args.max_examples)
  print(f"Loaded {len(examples)} GSM8K examples")

  h_starts, h_ends = capture_dataset(
    model, tok, examples,
    max_seq_len=args.max_seq_len,
    save_path=DATA_DIR,
    multiscale=args.multiscale,
  )
  return {"n_pairs": len(h_starts), "dim": h_starts.shape[1], "multiscale": args.multiscale}

@weave.op()
def cmd_train(args):
  """Train the latent reasoner on captured hidden states."""
  from cogito.train import train_reasoner

  h_starts = np.load(f"{DATA_DIR}/h_starts.npy")
  h_ends = np.load(f"{DATA_DIR}/h_ends.npy")
  model_dim = h_starts.shape[1]
  print(f"Loaded {h_starts.shape[0]} pairs, dim={model_dim}")

  result = train_reasoner(
    h_starts, h_ends,
    model_dim=model_dim,
    latent_dim=args.latent_dim,
    n_steps=args.n_steps,
    batch_size=args.batch_size,
    epochs=args.epochs,
    lr=args.lr,
    save_dir=CHECKPOINT_DIR,
  )
  return result

@weave.op()
def cmd_eval(args):
  """Evaluate reasoner against baselines."""
  from cogito.train import cosine_similarity, loss_fn
  from cogito.reasoner import LoopedResidualGRU, LinearBaseline

  h_starts = np.load(f"{DATA_DIR}/h_starts.npy")
  h_ends = np.load(f"{DATA_DIR}/h_ends.npy")
  model_dim = h_starts.shape[1]
  N = h_starts.shape[0]

  # Use last 10% as test
  n_test = max(int(0.1 * N), 1)
  test_starts = Tensor(h_starts[-n_test:].astype(np.float32))
  test_ends = Tensor(h_ends[-n_test:].astype(np.float32))

  results = {}

  # B0: Random hidden state
  random_h = Tensor.randn(*test_ends.shape) * test_ends.std().numpy().item()
  cos_random = cosine_similarity(random_h, test_ends).mean().numpy().item()
  mse_random = (random_h - test_ends).pow(2).mean().numpy().item()
  results["B0_random_h"] = {"cos_sim": cos_random, "mse": mse_random}

  # B1: Copy h_start (identity baseline)
  cos_copy = cosine_similarity(test_starts, test_ends).mean().numpy().item()
  mse_copy = (test_starts - test_ends).pow(2).mean().numpy().item()
  results["B1_copy_h_start"] = {"cos_sim": cos_copy, "mse": mse_copy}

  # B2: Linear projection (train quickly)
  print("Training linear baseline...")
  linear = LinearBaseline(model_dim)
  from tinygrad.nn.optim import AdamW
  from tinygrad.nn.state import get_parameters
  opt = AdamW(get_parameters(linear), lr=1e-3)
  train_starts = Tensor(h_starts[:-n_test].astype(np.float32))
  train_ends = Tensor(h_ends[:-n_test].astype(np.float32))
  for ep in range(50):
    h_pred = linear(train_starts)
    loss = loss_fn(h_pred, train_ends)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if (ep + 1) % 10 == 0:
      print(f"  Linear ep {ep+1}: loss={loss.numpy().item():.4f}")

  linear_pred = linear(test_starts)
  cos_linear = cosine_similarity(linear_pred, test_ends).mean().numpy().item()
  mse_linear = (linear_pred - test_ends).pow(2).mean().numpy().item()
  results["B2_linear_proj"] = {"cos_sim": cos_linear, "mse": mse_linear}

  # Cogito reasoner (load best checkpoint)
  ckpt_path = f"{CHECKPOINT_DIR}/best_reasoner.npz"
  if os.path.exists(ckpt_path):
    print("Loading trained reasoner...")
    reasoner = LoopedResidualGRU(model_dim, args.latent_dim, args.n_steps)
    ckpt = np.load(ckpt_path)
    from tinygrad.nn.state import get_state_dict
    sd = get_state_dict(reasoner)
    for k in sd:
      if k in ckpt:
        sd[k].replace(Tensor(ckpt[k]))

    cogito_pred = reasoner(test_starts)
    cos_cogito = cosine_similarity(cogito_pred, test_ends).mean().numpy().item()
    mse_cogito = (cogito_pred - test_ends).pow(2).mean().numpy().item()
    results["cogito_gru"] = {"cos_sim": cos_cogito, "mse": mse_cogito}

  # Print results table
  print("\n" + "=" * 60)
  print(f"{'Method':<20} {'Cos Sim':>10} {'MSE':>10}")
  print("-" * 60)
  for method, metrics in results.items():
    print(f"{method:<20} {metrics['cos_sim']:>10.4f} {metrics['mse']:>10.4f}")
  print("=" * 60)

  return results

@weave.op()
def cmd_all(args):
  """Run full pipeline: capture -> train -> eval."""
  capture_result = cmd_capture(args)
  train_result = cmd_train(args)
  eval_result = cmd_eval(args)
  return {"capture": capture_result, "train": train_result, "eval": eval_result}

def main():
  parser = argparse.ArgumentParser(description="Cogito v0.1 — Latent Reasoning Experiment")
  parser.add_argument("command", choices=["capture", "train", "eval", "all"])
  parser.add_argument("--max-examples", type=int, default=500, help="Max GSM8K examples to capture")
  parser.add_argument("--max-seq-len", type=int, default=512, help="Max sequence length")
  parser.add_argument("--latent-dim", type=int, default=512, help="Latent reasoner hidden dim")
  parser.add_argument("--n-steps", type=int, default=8, help="GRU reasoning steps")
  parser.add_argument("--batch-size", type=int, default=128, help="Training batch size")
  parser.add_argument("--epochs", type=int, default=100, help="Max training epochs")
  parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
  parser.add_argument("--multiscale", action="store_true", help="Multi-scale capture (50%/67%/80% layers)")
  args = parser.parse_args()

  weave.init("cogito")
  {"capture": cmd_capture, "train": cmd_train, "eval": cmd_eval, "all": cmd_all}[args.command](args)
  weave.finish()

if __name__ == "__main__":
  main()
