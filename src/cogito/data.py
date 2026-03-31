"""Data loading and formatting for hidden state capture.

Loads GSM8K, formats examples with <think>...</think> markers,
and runs them through the model to capture hidden state pairs.
"""
from tinygrad import Tensor
from tinygrad.apps.llm import Transformer, SimpleTokenizer
import numpy as np
import os

from cogito.capture import forward_with_capture, get_capture_layer, get_multiscale_layers

THINK_START = "<think>"
THINK_END = "</think>"

def load_gsm8k(split: str = "train", max_examples: int | None = None) -> list[dict]:
  """Load GSM8K dataset. Returns list of {question, answer, cot}."""
  from datasets import load_dataset
  ds = load_dataset("openai/gsm8k", "main", split=split)
  examples = []
  for item in ds:
    answer_text = item["answer"]
    parts = answer_text.split("####")
    cot = parts[0].strip()
    final_answer = parts[1].strip() if len(parts) > 1 else answer_text
    examples.append({"question": item["question"], "cot": cot, "answer": final_answer})
    if max_examples and len(examples) >= max_examples:
      break
  return examples

def format_sequence(question: str, cot: str, answer: str) -> str:
  """Format a single example with think markers."""
  return f"{question} {THINK_START} {cot} {THINK_END} {answer}"

def find_marker_positions(token_ids: list[int], tokenizer, marker: str) -> int | None:
  """Find the token position where a marker starts in the sequence."""
  text_so_far = ""
  for i, tid in enumerate(token_ids):
    text_so_far = tokenizer.decode(token_ids[:i+1])
    if marker in text_so_far:
      pre_marker = text_so_far[:text_so_far.index(marker)]
      for j in range(i, -1, -1):
        if tokenizer.decode(token_ids[:j]).rstrip() == pre_marker.rstrip():
          return j
      return i
  return None

def capture_dataset(
    model: Transformer,
    tokenizer,
    examples: list[dict],
    capture_layer: int | list[int] | None = None,
    max_seq_len: int = 512,
    save_path: str = "data/hidden_states",
    multiscale: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
  """Run examples through model and capture (h_start, h_end) pairs.

  Args:
    multiscale: if True, capture from 3 layers (50%, 67%, 80%) and concatenate.
                This gives 3x model_dim features.

  Returns:
    h_starts: (N, dim) or (N, 3*dim) numpy array
    h_ends: same shape
  """
  if multiscale:
    capture_layers = get_multiscale_layers(model)
    print(f"Multi-scale capture at layers {capture_layers}/{len(model.blk)}")
  elif capture_layer is None:
    capture_layers = [get_capture_layer(model)]
    print(f"Capturing at layer {capture_layers[0]}/{len(model.blk)}")
  elif isinstance(capture_layer, int):
    capture_layers = [capture_layer]
    print(f"Capturing at layer {capture_layer}/{len(model.blk)}")
  else:
    capture_layers = capture_layer
    print(f"Capturing at layers {capture_layers}/{len(model.blk)}")

  h_starts, h_ends = [], []
  skipped = 0

  for i, ex in enumerate(examples):
    seq = format_sequence(ex["question"], ex["cot"], ex["answer"])
    token_ids = tokenizer.encode(seq)

    if len(token_ids) > max_seq_len:
      skipped += 1
      continue

    think_pos = find_marker_positions(token_ids, tokenizer, THINK_START)
    end_think_pos = find_marker_positions(token_ids, tokenizer, THINK_END)

    if think_pos is None or end_think_pos is None:
      skipped += 1
      continue

    # Forward pass — single example, no grad
    tokens = Tensor([token_ids], dtype="int32")
    Tensor.no_grad = True
    _, hidden_dict = forward_with_capture(model, tokens, capture_layers)
    Tensor.no_grad = False

    # Extract and optionally concatenate across layers
    start_parts, end_parts = [], []
    for layer_idx in sorted(hidden_dict.keys()):
      h = hidden_dict[layer_idx]
      start_parts.append(h[0, think_pos].numpy())
      end_parts.append(h[0, end_think_pos].numpy())

    h_start = np.concatenate(start_parts) if len(start_parts) > 1 else start_parts[0]
    h_end = np.concatenate(end_parts) if len(end_parts) > 1 else end_parts[0]

    h_starts.append(h_start)
    h_ends.append(h_end)

    if (i + 1) % 100 == 0:
      print(f"  Captured {i+1}/{len(examples)} ({skipped} skipped)")

  print(f"Done: {len(h_starts)} pairs captured, {skipped} skipped")

  h_starts_arr = np.stack(h_starts).astype(np.float32)
  h_ends_arr = np.stack(h_ends).astype(np.float32)

  os.makedirs(save_path, exist_ok=True)
  np.save(f"{save_path}/h_starts.npy", h_starts_arr)
  np.save(f"{save_path}/h_ends.npy", h_ends_arr)
  # Save metadata
  import json
  meta = {"capture_layers": capture_layers, "multiscale": multiscale,
          "n_pairs": len(h_starts_arr), "dim": h_starts_arr.shape[1],
          "model": "llama3.2-1b", "max_seq_len": max_seq_len}
  with open(f"{save_path}/metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

  print(f"Saved to {save_path}: h_starts {h_starts_arr.shape}, h_ends {h_ends_arr.shape}")
  return h_starts_arr, h_ends_arr
