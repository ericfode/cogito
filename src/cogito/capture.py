"""Hidden state capture from tinygrad's built-in LLaMA models.

Hooks into the Transformer.forward() to extract residual stream at target layers.
Uses GGUF models directly — no custom model code needed.
"""
from tinygrad import Tensor
from tinygrad.apps.llm import Transformer

def forward_with_capture(model: Transformer, tokens: Tensor,
                         capture_layers: list[int] | int) -> tuple[Tensor, dict[int, Tensor]]:
  """Run forward pass, return (logits, {layer_idx: hidden_states}).

  Args:
    model: tinygrad Transformer loaded from GGUF
    tokens: (B, T) int tensor
    capture_layers: layer index or list of layer indices to capture

  Returns:
    logits: (B, T, vocab_size)
    captured: dict mapping layer index -> (B, T, dim) residual stream
  """
  if isinstance(capture_layers, int):
    capture_layers = [capture_layers]
  capture_set = set(capture_layers)

  x = model.token_embd(tokens)
  captured = {}
  for i, block in enumerate(model.blk):
    x = block(x, 0)  # start_pos=0, full sequence, no KV cache
    if i in capture_set:
      captured[i] = x
  logits = model.output(model.output_norm(x))
  return logits, captured

def get_capture_layer(model: Transformer) -> int:
  """67% depth heuristic for optimal hidden state capture."""
  return int(len(model.blk) * 0.67)

def get_multiscale_layers(model: Transformer) -> list[int]:
  """Return layers at 50%, 67%, 80% depth for multi-scale capture."""
  n = len(model.blk)
  return [int(n * 0.50), int(n * 0.67), int(n * 0.80)]

def capture_hidden_states(model: Transformer, tokens: Tensor,
                          think_positions: list[int], end_think_positions: list[int],
                          capture_layers: list[int] | int) -> tuple[Tensor, Tensor]:
  """Capture h_start and h_end from a batch of sequences.

  For multi-scale: captures are concatenated along the feature dimension.

  Args:
    model: loaded Transformer
    tokens: (B, T) token ids
    think_positions: position of <think> token per batch element
    end_think_positions: position of </think> token per batch element
    capture_layers: which layer(s) to extract from

  Returns:
    h_start: (B, dim * n_layers) hidden state at <think> position
    h_end: (B, dim * n_layers) hidden state at </think> position
  """
  if isinstance(capture_layers, int):
    capture_layers = [capture_layers]

  _, hidden_dict = forward_with_capture(model, tokens, capture_layers)
  B = tokens.shape[0]

  h_starts, h_ends = [], []
  for b in range(B):
    # Concatenate across captured layers for multi-scale
    start_parts, end_parts = [], []
    for layer_idx in sorted(hidden_dict.keys()):
      hidden = hidden_dict[layer_idx]
      start_parts.append(hidden[b, think_positions[b]])
      end_parts.append(hidden[b, end_think_positions[b]])

    h_starts.append(Tensor.cat(*start_parts) if len(start_parts) > 1 else start_parts[0])
    h_ends.append(Tensor.cat(*end_parts) if len(end_parts) > 1 else end_parts[0])

  h_start = Tensor.stack(*h_starts)
  h_end = Tensor.stack(*h_ends)
  return h_start, h_end
