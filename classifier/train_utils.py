import torch


def get_best_system_device() -> str:
  """Detect the best available device on this machine."""
  device = "cpu"
  if torch.cuda.is_available():
    device = "cuda"
  elif torch.backends.mps.is_available():
    device = "mps"
  return device